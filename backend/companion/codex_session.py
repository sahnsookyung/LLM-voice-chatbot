"""Codex session adapter. Host owns tasks/history; Ari owns only presentation/audio."""

import asyncio
import time
import uuid
from pathlib import Path

from .codex_audio import CodexAudio
from .codex_compat import codex_home, host_id
from .codex_history import CodexTimeline, display_batches, tool_catalog_summary
from .codex_rpc import CodexError
from .codex_rtc import CodexRTC
from .codex_types import AvatarCue, approval_result
from .diagnostics import Timings, memory_snapshot
from .protocol import decode_microphone


class CodexSession:
    def __init__(
        self, rpc, store, send, send_audio, *, validation_thread=None, rtc_factory=CodexRTC
    ):
        self.rpc, self.store, self.send = rpc, store, send
        self.thread = None
        self.active_turn = None
        self.turn_revision = 0
        self.timeline = CodexTimeline()
        self.cursor = None
        self.voice_state = "off"
        self.voice_unavailable_reason = None
        self.voice_id = store.setting("codex_voice", "")
        self.realtime_id = None
        self.owns_voice = False
        self.capture = None
        self.sequence = self.offset = 0
        self.input_queue = asyncio.Queue(maxsize=64)
        self.input_worker = None
        self.event_worker = None
        self.refresh_task = None
        self.requests = {}
        self.audio = CodexAudio(self.event, send_audio)
        self.settings = {}
        self.voices = []
        self.models = []
        self.closed = False
        self.submission_uncertain = False
        self.avatar_visible = True
        self.avatar_mode = "illustration"
        self.reduced_motion = False
        self.cue_expiry = None
        self.voice_epoch = 0
        self.voice_start = None
        self.voice_start_sent = False
        self.suppress_audio = False
        self.history_lock = asyncio.Lock()
        self.history_loading = False
        self.task_cursor = None
        self.archived = False
        self.read_only = False
        self.automation_items = set()
        self.timings = Timings()
        self.buffered_history_events = None
        self.history_session = None
        self.history_position = -1
        self.voice_revision = 0
        self.pending_cue = None
        self.validation_thread = validation_thread
        self.rtc_factory = rtc_factory
        self.rtc = None

    async def event(self, kind, **values):
        if self.closed:
            return
        if kind == "codex_audio_failed":
            asyncio.create_task(self.end_voice())
        if not self.closed:
            await self.send({"version": 1, "type": kind, "provider": "codex", **values})

    async def open(self):
        if Path(self.rpc.identity.get("codexHome", "")).resolve() != codex_home():
            raise CodexError("Codex home differs from the shared desktop home.")
        self.event_worker = asyncio.create_task(self._events())
        self.refresh_task = asyncio.create_task(self._metrics())
        await self.event(
            "ready",
            backend="codex",
            conversation_id="",
            microphone_versions=[2],
            character_id=self.store.setting("character", "ari"),
        )
        await self.catalog()
        selected = self.validation_thread or self.store.setting("codex_thread", "")
        if not self.validation_thread and self.store.setting("codex_host", host_id()) != host_id():
            selected = ""
            await self.event(
                "notice", message="The Codex home changed. Choose a task on this host."
            )
        if selected:
            try:
                await self.select(selected)
            except CodexError:
                if self.rpc.closed:
                    raise
                await self.event(
                    "notice", message="The previous Codex task is unavailable. Choose a task."
                )

    async def _metrics(self):
        while True:
            await self.event("metrics", **memory_snapshot())
            await asyncio.sleep(5)

    async def catalog(self, more=False):
        params = {"limit": 100, "sortKey": "updated_at", "archived": self.archived}
        if more and self.task_cursor:
            params["cursor"] = self.task_cursor
        if self.validation_thread:
            result = await self.rpc.call(
                "thread/read", {"threadId": self.validation_thread, "includeTurns": False}
            )
            threads = {"data": [result["thread"]]}
        else:
            threads = await self.rpc.call("thread/list", params)
        self.task_cursor = threads.get("nextCursor")
        result = await self.rpc.call("thread/realtime/listVoices", {})
        catalog = result.get("voices", {})
        # V3 currently uses the host's V2 voice catalog; readiness testing must verify it.
        self.voices = catalog.get("v2", [])
        if self.voice_id not in self.voices:
            self.voice_id = ""  # Let V3 choose its actual default; V2 names are only candidates.
        await self.event(
            "codex_catalog",
            threads=threads.get("data", []),
            next_cursor=self.task_cursor,
            append=more,
            archived=self.archived,
            voices=self.voices,
            voice=self.voice_id,
            voice_catalog_status="candidate voices; verified only when a V3 session starts",
        )
        for method, name, params in [
            ("model/list", "models", {}),
            ("mcpServerStatus/list", "tools", {}),
            ("project/list", "projects", {"limit": 100}),
            (
                "plugin/installed",
                "plugins",
                {"cwds": [self.thread["cwd"]] if self.thread else None},
            ),
            ("account/read", "account", {"refreshToken": False}),
            ("account/rateLimits/read", "limits", {}),
        ]:
            try:
                value = await self.rpc.call(method, params)
                if name == "models":
                    self.models = value.get("data", [])
                if name == "tools":
                    value = tool_catalog_summary(value)
                if name == "account":
                    account = value.get("account") or {}
                    value = {
                        "signed_in": bool(account),
                        "type": account.get("type"),
                        "plan": account.get("planType"),
                    }
                await self.event("codex_capability", name=name, value=value)
            except CodexError:
                await self.event("codex_capability", name=name, unavailable=True)

    async def select(self, identifier):
        if self.validation_thread and identifier != self.validation_thread:
            raise CodexError("Validation is restricted to its designated task.")
        if self.thread and identifier == self.thread["id"]:
            await self.refresh_history()
            return
        self.submission_uncertain = True
        await self.end_voice()
        if self.thread:
            await self.rpc.call("thread/unsubscribe", {"threadId": self.thread["id"]})
        self.requests.clear()
        self.automation_items.clear()
        await self.event("codex_requests_clear")
        await self.event("codex_automation", active=False)
        metadata = await self.rpc.call(
            "thread/read", {"threadId": identifier, "includeTurns": False}
        )
        self.read_only = self.archived
        result = (
            metadata
            if self.read_only
            else await self.rpc.call(
                "thread/resume", {"threadId": identifier, "excludeTurns": True}
            )
        )
        self.thread = result["thread"]
        self.settings = {
            key: result.get(key)
            for key in (
                "model",
                "reasoningEffort",
                "approvalPolicy",
                "approvalsReviewer",
                "activePermissionProfile",
                "disabledPluginIds",
                "cwd",
            )
        }
        self.store.set_setting("codex_thread", identifier)
        self.store.set_setting("codex_host", host_id())
        self.timeline = CodexTimeline()
        self.cursor = None
        self.active_turn = None
        self.submission_uncertain = False
        await self.refresh_history()
        await self.event("codex_selected", thread=self.thread, settings=self.settings)

    async def refresh_history(self, more=False):
        if not self.thread:
            return
        # Entries are chronological within each page, but pages arrive newest
        # first. Older pages must not overwrite current voice ownership or evict
        # recent replies. Positions, not arrival order, define the display window.
        async with self.history_lock:
            self.history_loading = True
            self.buffered_history_events = []
            await self.state()
            if not more:
                self.timeline = CodexTimeline()
                self.cursor = None
                self.active_turn = None
                self.history_session = None
                self.history_position = -1
            voice_revision = self.voice_revision
            history_session = self.history_session
            history_position = self.history_position
            try:
                history_ready = False
                seen_cursors = set()
                for _ in range(500):
                    params = {"threadId": self.thread["id"], "limit": 20}
                    if self.cursor:
                        params["cursor"] = self.cursor
                    page = await self.rpc.call("thread/timeline/list", params)
                    self.timeline.load(page.get("data", []))
                    entries = sorted(page.get("data", []), key=lambda entry: entry["position"])
                    page_session = page.get("activeRealtimeSessionAtPageStart")
                    for entry in entries:
                        item = entry.get("item", {})
                        if entry.get("type") == "realtime":
                            if item.get("type") == "realtimeSessionStarted":
                                page_session = item.get("realtimeSessionId")
                            elif item.get("type") == "realtimeSessionClosed":
                                page_session = None
                        elif entry.get("type") == "turnStarted":
                            self.active_turn = entry.get("turn_id")
                        elif (
                            entry.get("type") == "turnCompleted"
                            and entry.get("turn_id") == self.active_turn
                        ):
                            self.active_turn = None
                    page_position = entries[-1]["position"] if entries else -1
                    if page_position >= history_position:
                        history_session, history_position = page_session, page_position
                    self.cursor = page.get("nextCursor")
                    if not self.cursor:
                        break
                    if self.cursor in seen_cursors:
                        raise CodexError(
                            "Codex repeated a history cursor. Reconnect before continuing."
                        )
                    seen_cursors.add(self.cursor)
                self.history_loading = bool(self.cursor)
                if voice_revision == self.voice_revision:
                    self.history_session, self.history_position = history_session, history_position
                    if not self.owns_voice and (not self.voice_start or self.voice_start.done()):
                        self.voice_state = "in_desktop" if self.history_session else "off"
                self.submission_uncertain = bool(self.cursor)
                for index, rows in enumerate(display_batches(self.timeline.snapshot())):
                    await self.event(
                        "codex_history",
                        messages=rows,
                        append=index > 0,
                        more=bool(self.cursor),
                        truncated=self.timeline.truncated,
                    )
                history_ready = True
            except BaseException:
                self.submission_uncertain = True
                raise
            finally:
                queued, self.buffered_history_events = self.buffered_history_events, None
                for message in queued:
                    # Persisted completed items already contain these text deltas.
                    identifier = message.get("params", {}).get("itemId")
                    if (
                        message["method"].endswith("/delta")
                        and identifier in self.timeline.completed
                    ):
                        continue
                    await self.handle_event(message)
                if history_ready and not self.cursor:
                    self.submission_uncertain = True
                    await self.reconcile_task_state()
                    self.submission_uncertain = False
                await self.state()

    async def reconcile_task_state(self):
        # The persisted timeline may end at the previous completed turn. Obtain
        # the live turn separately without downloading its tool/image contents.
        revision = self.turn_revision
        result = await self.rpc.call(
            "thread/turns/list",
            {
                "threadId": self.thread["id"],
                "limit": 1,
                "sortDirection": "desc",
                "itemsView": "notLoaded",
            },
        )
        turns = result.get("data")
        if not isinstance(turns, list):
            raise CodexError("Cannot verify current task state. Refresh before submitting.")
        if revision == self.turn_revision:
            self.active_turn = (
                turns[0]["id"] if turns and turns[0].get("status") == "inProgress" else None
            )

    async def state(self):
        await self.event(
            "codex_state",
            voice=self.voice_state,
            voice_unavailable_reason=self.voice_unavailable_reason,
            task="working" if self.active_turn else "idle",
            host_id=host_id(),
            thread_id=self.thread["id"] if self.thread else None,
            turn_id=self.active_turn,
            submission_uncertain=self.submission_uncertain or self.history_loading,
            read_only=self.read_only,
            validation=bool(self.validation_thread),
        )

    async def text(self, text):
        if not self.thread:
            raise CodexError("Select a Codex task first.")
        if self.read_only:
            raise CodexError(
                "Archived history is read-only. Restore the task in Codex before continuing."
            )
        if self.submission_uncertain or self.history_loading:
            raise CodexError("Refresh canonical history before sending again.")
        if self.voice_state in {"in_desktop", "unknown", "starting"}:
            raise CodexError(
                "Voice ownership is not available in Ari. Check Codex desktop or refresh history."
            )
        try:
            if self.owns_voice:
                self.suppress_audio = False
            # This method handles explicit composer input only. Voice transcripts
            # use the host's automatic delegation and never call text(). Realtime
            # appendText only adds context: it does not reliably submit a typed
            # request or persist it as a user message in the task timeline.
            if self.active_turn:
                await self.rpc.call(
                    "turn/steer",
                    {
                        "threadId": self.thread["id"],
                        "expectedTurnId": self.active_turn,
                        "clientUserMessageId": str(uuid.uuid4()),
                        "input": [{"type": "text", "text": text, "text_elements": []}],
                    },
                )
            else:
                await self.rpc.call(
                    "turn/start",
                    {
                        "threadId": self.thread["id"],
                        "clientUserMessageId": str(uuid.uuid4()),
                        "input": [{"type": "text", "text": text, "text_elements": []}],
                    },
                )
        except (CodexError, asyncio.CancelledError):
            self.submission_uncertain = True
            await self.state()
            raise

    async def start_voice(self, capture):
        if self.voice_unavailable_reason:
            raise CodexError(self.voice_unavailable_reason)
        if not self.thread or self.read_only or self.history_loading or self.submission_uncertain:
            raise CodexError("Select a task and reconcile its history before starting voice.")
        if self.voice_state in {"in_desktop", "unknown"}:
            raise CodexError(
                "End voice in Codex desktop, then refresh history to hand voice to Ari."
            )
        if self.voice_start and not self.voice_start.done():
            raise CodexError("Voice is already starting.")
        self.capture = str(capture)
        self.sequence = self.offset = 0
        if self.owns_voice:
            await self._listen()
            return
        self.voice_epoch += 1
        epoch = self.voice_epoch
        self.voice_state = "starting"
        self.voice_start_sent = False
        self.suppress_audio = False
        await self.state()

        async def prepare():
            began = time.monotonic()
            params = {
                "threadId": self.thread["id"],
                "version": "v3",
                "outputModality": "audio",
                "includeStartupContext": True,
                "clientManagedHandoffs": False,
                "backendReasoningStatus": False,
                "delegationAckFiller": False,
                "codexResponseHandoffMode": "thinking",
                "prompt": (
                    "Speak naturally through Ari's animated avatar in the user's language, "
                    "including English, Korean and Japanese. Preserve project instructions "
                    "and task permissions. Reply naturally to conversation and questions. "
                    "For delegated tasks, treat tool calls and routine progress as background "
                    "activity. Do not read commands, tool arguments, tool output, or progress "
                    "logs aloud unless the user explicitly asks for those details. Give a "
                    "concise outcome summary grounded in the task's actual results, including "
                    "failures or unfinished work; never announce success before it is confirmed. "
                    "Ask when user input or approval is needed, without granting it yourself. "
                    "Do not speak control tags. When available, use ari_avatar_present for "
                    "appropriate presentation cues; it does not execute task actions."
                ),
            }
            if self.voice_id:
                params["voice"] = self.voice_id
            try:
                rtc = self.rtc_factory(
                    lambda pcm: self._rtc_audio(epoch, pcm),
                    lambda message: self._rtc_failed(epoch, message),
                )
                self.rtc = rtc
                params["transport"] = {"type": "webrtc", "sdp": await rtc.offer()}
                if epoch != self.voice_epoch:
                    return
                self.voice_start_sent = True
                await self.rpc.call("thread/realtime/start", params, timeout=30)
                await rtc.wait_ready()
                if epoch != self.voice_epoch:
                    return  # End voice already sent a stop after this ordered start.
                self.owns_voice = True
                self.timings.since("voice_startup", began)
                await self._listen()
            except (CodexError, OSError, RuntimeError, ValueError, TimeoutError):
                if epoch == self.voice_epoch:
                    await self.end_voice()
                    await self.event(
                        "error",
                        message="Voice startup was not confirmed. Refresh history and check Codex before retrying.",
                    )
            finally:
                await self.state()

        self.voice_start = asyncio.create_task(prepare())

    async def _rtc_audio(self, epoch, pcm):
        if epoch != self.voice_epoch or self.suppress_audio or self.closed:
            return
        if not self.owns_voice and self.voice_state != "starting":
            return
        # V3 is a continuous full-duplex track, not one RTP stream per phrase.
        # Rotate the playback identity after every interruption, never on a
        # transcript boundary (text completion is not audible completion).
        item = self.audio.item_id or f"rtc-{epoch}-{uuid.uuid4()}"
        await self.audio.feed_pcm(pcm, 24000, 1, item)
        if self.pending_cue:
            cue, self.pending_cue = self.pending_cue, None
            await self.present(cue)

    async def _rtc_failed(self, epoch, message):
        if epoch != self.voice_epoch or self.closed:
            return
        await self.end_voice()
        await self.event("notice", message=message)

    async def _listen(self):
        self.voice_state = "listening" if self.capture else "muted"
        if self.rtc:
            self.rtc.set_muted(not self.capture)
        if not self.input_worker or self.input_worker.done():
            self.input_worker = asyncio.create_task(self._send_input())
        await self.state()
        if self.capture:
            await self.event("listening", enabled=True, capture_id=self.capture)

    async def input_audio(self, packet):
        capture, sequence, offset, pcm = decode_microphone(packet)
        if not self.owns_voice or capture != self.capture:
            return
        if sequence != self.sequence or offset != self.offset:
            await self.end_voice()
            raise CodexError("Microphone audio arrived out of order; voice stopped.")
        self.sequence += 1
        self.offset += len(pcm) // 2
        try:
            self.input_queue.put_nowait((capture, pcm))
        except asyncio.QueueFull as error:
            await self.end_voice()
            raise CodexError(
                "Microphone delivery fell behind; voice stopped without dropping words silently."
            ) from error

    async def _send_input(self):
        try:
            while True:
                capture, pcm = await self.input_queue.get()
                try:
                    if self.owns_voice and capture == self.capture:
                        if not self.rtc:
                            raise CodexError("Codex voice media is not connected.")
                        self.rtc.feed(pcm)
                finally:
                    self.input_queue.task_done()
        except asyncio.CancelledError:
            raise
        except (CodexError, ValueError, KeyError, TypeError, OSError, RuntimeError):
            await self.end_voice()
            await self.event(
                "notice",
                message="Codex microphone delivery failed. Voice stopped; task status is unchanged.",
            )

    async def mute(self):
        self.capture = None
        if self.rtc:
            self.rtc.set_muted(True)
        while not self.input_queue.empty():
            self.input_queue.get_nowait()
            self.input_queue.task_done()
        if self.owns_voice:
            self.voice_state = "muted"
        await self.event("listening", enabled=False)
        await self.state()

    async def end_voice(self):
        previous_state = self.voice_state
        owned = self.owns_voice or (self.voice_state == "starting" and self.voice_start_sent)
        if self.voice_start and not self.voice_start_sent:
            self.voice_start.cancel()
        self.voice_epoch += 1
        self.pending_cue = None
        uncertain = False
        self.owns_voice = False
        await self.mute()
        rtc, self.rtc = self.rtc, None
        if rtc:
            await rtc.close()
        if self.input_worker and self.input_worker is not asyncio.current_task():
            self.input_worker.cancel()
            await asyncio.gather(self.input_worker, return_exceptions=True)
        self.input_worker = None
        await self.audio.clear()
        if owned and self.thread and not self.rpc.closed:
            try:
                await self.rpc.call(
                    "thread/realtime/stop", {"threadId": self.thread["id"]}, timeout=5
                )
            except CodexError:
                uncertain = True
                await self.event(
                    "notice",
                    message="Local audio stopped. Codex voice shutdown was not acknowledged; check the desktop before starting voice again.",
                )
        self.realtime_id = None
        self.voice_state = (
            "unknown"
            if uncertain
            else previous_state
            if not owned and previous_state in {"in_desktop", "unknown"}
            else "off"
        )
        await self.state()

    async def stop_task(self):
        self.suppress_audio = True
        self.pending_cue = None
        await self.audio.clear()
        if self.active_turn and self.thread and not self.read_only:
            await self.rpc.call(
                "turn/interrupt", {"threadId": self.thread["id"], "turnId": self.active_turn}
            )
        await self.event(
            "notice",
            message="Playback stopped. Waiting for Codex to confirm task interruption; completed actions remain completed.",
        )

    async def command(self, command):
        action = command.action
        if self.validation_thread and action in {"new", "archived", "settings"}:
            raise CodexError(
                "Task creation, archive browsing and settings changes are disabled during validation."
            )
        if action == "refresh":
            await self.catalog()
            await self.refresh_history()
        elif action == "tasks_more":
            if self.task_cursor:
                await self.catalog(more=True)
        elif action == "archived":
            self.archived = not self.archived
            await self.catalog()
        elif action == "select":
            await self.select(command.thread_id)
        elif action == "new":
            if (
                not command.cwd
                or not Path(command.cwd).is_absolute()
                or not Path(command.cwd).is_dir()
            ):
                raise ValueError("Choose an existing absolute project folder.")
            result = await self.rpc.call(
                "thread/start",
                {
                    "cwd": command.cwd,
                    "projectId": command.project_id,
                    "ephemeral": False,
                    "historyMode": "paginated",
                    "dynamicTools": [
                        {
                            "type": "function",
                            "name": "ari_avatar_present",
                            "description": "Presentation-only expression/gesture for Ari. No external effects. Never include cue tags in speech.",
                            "inputSchema": AvatarCue.model_json_schema(),
                        }
                    ],
                },
            )
            self.archived = False
            await self.select(result["thread"]["id"])
            await self.catalog()
        elif action == "history_more":
            await self.refresh_history(more=True)
        elif action == "voice_end":
            await self.end_voice()
        elif action == "stop_task":
            await self.stop_task()
        elif action == "voice_select":
            if self.owns_voice:
                raise ValueError("End voice before changing its voice.")
            if command.voice and command.voice not in self.voices:
                raise ValueError("Voice not offered by this Codex host.")
            self.voice_id = command.voice
            self.store.set_setting("codex_voice", self.voice_id)
        elif action == "settings":
            if not self.thread or self.read_only:
                raise ValueError("Select an active task first.")
            changes = {"threadId": self.thread["id"]}
            if command.model is not None:
                model = next(
                    (
                        m
                        for m in self.models
                        if m.get("id") == command.model or m.get("model") == command.model
                    ),
                    None,
                )
                if not model:
                    raise ValueError("Model not offered by the host.")
                changes["model"] = command.model
            else:
                model = next(
                    (m for m in self.models if m.get("model") == self.settings.get("model")), None
                )
            if command.effort is not None:
                if not model or command.effort not in {
                    e.get("reasoningEffort") for e in model.get("supportedReasoningEfforts", [])
                }:
                    raise ValueError("Reasoning effort not supported by the selected model.")
                changes["effort"] = command.effort
            if command.disabled_plugins is not None:
                # Current schema explicitly warns this does not filter capabilities.
                raise ValueError(
                    "Plugin restrictions are not enforced by this host build. Manage plugins in Codex desktop."
                )
            await self.rpc.call("thread/settings/update", changes)
        elif action in {"approval", "answer"}:
            entry = self.requests.get(command.request_id)
            if not entry:
                raise ValueError("This request was already resolved or is no longer active.")
            if entry["params"].get("threadId") != self.thread["id"]:
                raise ValueError("Request belongs to another task.")
            if action == "answer":
                if entry["method"] != "item/tool/requestUserInput":
                    raise ValueError("This request is not a question.")
                questions = {q["id"] for q in entry["params"].get("questions", [])}
                if set(command.answers) != questions:
                    raise ValueError("Answer every question in this request.")
                result = {
                    "answers": {key: {"answers": values} for key, values in command.answers.items()}
                }
            else:
                result = approval_result(entry["method"], command.decision, entry["params"])
            # Remove before await so a double click cannot submit twice.
            self.requests.pop(command.request_id)
            await self.rpc.respond(entry["id"], result)
            await self.event("codex_request_resolved", request_id=command.request_id)

    async def _events(self):
        try:
            while True:
                message = await self.rpc.events.get()
                await self.handle_event(message)
        except asyncio.CancelledError:
            raise
        except (CodexError, ValueError, KeyError, TypeError, OSError, RuntimeError):
            self.submission_uncertain = True
            await self.end_voice()
            await self.event(
                "error",
                message="Codex event processing stopped. Reconnect and reload history before continuing.",
            )

    async def handle_event(self, message):
        method, p = message.get("method", ""), message.get("params", {})
        if method == "account/updated":
            # A confirmed account change permits a new explicit attempt. Never
            # restart the microphone or replay the failed request automatically.
            self.voice_unavailable_reason = None
            await self.state()
            return
        if method == "ari/disconnected":
            self.owns_voice = False
            self.voice_state = "off"
            await self.end_voice()
            self.submission_uncertain = True
            self.requests.clear()
            await self.event(
                "codex_disconnected",
                message=p.get("reason")
                or "Codex disconnected. Task submission may have succeeded; reconnect to inspect history before retrying.",
            )
            return
        if not self.thread or p.get("threadId") != self.thread["id"]:
            # Shared-host requests may also be visible to another client. A rejection
            # here could win the response race and deny someone else's tool call.
            return
        if (
            self.buffered_history_events is not None
            and "id" not in message
            and (
                method
                in {
                    "item/started",
                    "item/completed",
                    "item/agentMessage/delta",
                    "turn/started",
                    "turn/completed",
                }
                or method.startswith("thread/realtime/item/")
            )
        ):
            if len(self.buffered_history_events) >= 256:
                raise CodexError("History reconciliation overflowed. Reconnect before continuing.")
            self.buffered_history_events.append(message)
            return
        if "id" in message:
            if method == "item/tool/call" and p.get("tool") == "ari_avatar_present":
                if p.get("turnId") != self.active_turn:
                    raise CodexError("Avatar cue does not belong to the active task turn.")
                await self.present(p.get("arguments", {}))
                await self.rpc.respond(
                    message["id"],
                    {
                        "contentItems": [{"type": "inputText", "text": "Avatar cue accepted."}],
                        "success": True,
                    },
                )
            elif method in {
                "item/commandExecution/requestApproval",
                "item/fileChange/requestApproval",
                "item/permissions/requestApproval",
                "item/tool/requestUserInput",
                "mcpServer/elicitation/request",
            }:
                if len(self.requests) >= 32:
                    raise CodexError("Too many pending approvals.")
                identifier = str(message["id"])
                self.requests[identifier] = message
                await self.event("codex_request", request_id=identifier, method=method, params=p)
            else:
                # Do not race the desktop with an unsupported-tool error. Its
                # own tool bridge must handle the call; live routing is a gate.
                await self.event(
                    "notice",
                    message="A Codex interaction requires the desktop client. Open the task in Codex; Ari has not answered or approved it.",
                )
            return
        if method == "serverRequest/resolved":
            identifier = str(p.get("requestId"))
            self.requests.pop(identifier, None)
            await self.event("codex_request_resolved", request_id=identifier)
        elif method in {
            "item/started",
            "item/completed",
            "thread/realtime/item/started",
            "thread/realtime/item/completed",
        }:
            item = p.get("item", {})
            realtime = method.startswith("thread/realtime")
            if (
                realtime
                and item.get("type") == "transcriptSegment"
                and item.get("role") == "user"
                and method.endswith("started")
            ):
                self.pending_cue = None
                self.suppress_audio = False
                await self.audio.clear()
            row = self.timeline.upsert(
                item, realtime=realtime, completed=method.endswith("completed")
            )
            if row:
                await self.event("codex_item", item=row)
            if item.get("type") in {"mcpToolCall", "commandExecution"}:
                if method.endswith("started"):
                    self.automation_items.add(item.get("id"))
                else:
                    self.automation_items.discard(item.get("id"))
                await self.event("codex_automation", active=bool(self.automation_items))
        elif method in {"item/agentMessage/delta", "thread/realtime/item/transcript/delta"}:
            row = self.timeline.delta(p.get("itemId"), p.get("delta", ""))
            if row:
                await self.event("codex_item", item=row)
        elif method == "thread/realtime/sdp":
            if self.rtc and self.voice_state == "starting":
                await self.rtc.answer(p.get("sdp"))
        elif method in {"thread/realtime/outputAudio/delta", "thread/realtime/itemAdded"}:
            # Sideband may reflect media/legacy completion events. Only the RTC
            # track owns playback; text completion must not close its resampler.
            return
        elif method == "thread/realtime/started":
            self.voice_revision += 1
            self.realtime_id = p.get("realtimeSessionId")
            self.history_session = self.realtime_id
            if self.voice_state != "starting" and not self.owns_voice:
                self.voice_state = "in_desktop"
            await self.state()
        elif method in {"thread/realtime/closed", "thread/realtime/error"}:
            self.voice_revision += 1
            self.history_session = None
            if (
                method.endswith("error")
                and p.get("message", "").strip().lower()
                == "realtime conversation requires api key auth"
            ):
                # Map this observed error to a fixed explanation; never display
                # arbitrary provider error bodies, prompts, or credentials.
                self.voice_unavailable_reason = (
                    "This Codex host rejected voice with an API-key authentication requirement. "
                    "Ari uses WebRTC with the existing sign-in; this host could not start it. "
                    "Typed tasks still work."
                )
            self.owns_voice = False
            self.voice_state = "off"
            await self.end_voice()
            if method.endswith("error"):
                await self.event(
                    "notice",
                    message=self.voice_unavailable_reason
                    or "Codex voice is unavailable or failed. Task work is separate; check Codex desktop.",
                )
        elif method == "turn/started":
            self.turn_revision += 1
            self.active_turn = p.get("turn", {}).get("id")
            await self.state()
        elif method == "turn/completed":
            if p.get("turn", {}).get("id") == self.active_turn:
                self.turn_revision += 1
                self.active_turn = None
                self.requests.clear()
                await self.event("codex_requests_clear")
                self.automation_items.clear()
                await self.event("codex_automation", active=False)
            await self.state()
        elif method == "thread/settings/updated":
            self.settings.update(p)
            await self.event("codex_settings", settings=self.settings)

    async def present(self, arguments):
        cue = AvatarCue.model_validate(arguments)
        if not self.owns_voice:
            return
        if not self.audio.turn:
            self.pending_cue = arguments
            return
        await self.event(
            "expression",
            turn_id=self.audio.turn,
            sample_offset=self.audio.sent,
            expression=cue.emotion,
        )
        if (
            cue.action != "none"
            and cue.emotion not in {"concerned", "annoyed", "sleepy"}
            and self.avatar_visible
            and self.avatar_mode == "model"
            and not self.reduced_motion
        ):
            await self.event(
                "gesture", turn_id=self.audio.turn, gesture=cue.action, wait_for_audio=True
            )
        if self.cue_expiry:
            self.cue_expiry.cancel()
        turn = self.audio.turn

        async def expire():
            await asyncio.sleep(5)
            if self.audio.turn == turn:
                await self.event(
                    "expression", turn_id=turn, sample_offset=self.audio.sent, expression="neutral"
                )

        self.cue_expiry = asyncio.create_task(expire())

    async def close(self):
        if self.closed:
            return
        # The native socket can already be gone. Teardown must not depend on
        # delivering presentation events before releasing media and host subscriptions.
        self.closed = True
        await self.end_voice()
        for task in (self.event_worker, self.refresh_task, self.cue_expiry, self.voice_start):
            if task:
                task.cancel()
        await asyncio.gather(
            *(
                t
                for t in (self.event_worker, self.refresh_task, self.cue_expiry, self.voice_start)
                if t
            ),
            return_exceptions=True,
        )
        if self.thread and not self.rpc.closed:
            try:
                await self.rpc.call(
                    "thread/unsubscribe", {"threadId": self.thread["id"]}, timeout=3
                )
            except CodexError:
                pass
        await self.rpc.close()
