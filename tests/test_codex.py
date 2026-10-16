"""Protocol fixtures, not evidence of live desktop/account compatibility."""

import asyncio
import base64
import json
import uuid

import httpx
import numpy as np
import pytest
from companion.app import create_app
from companion.codex_audio import CodexAudio
from companion.codex_compat import ACCEPTANCE_GATES, accepted_report
from companion.codex_history import CodexTimeline, display_user_text, project_item
from companion.codex_rpc import CodexError, CodexRPC
from companion.codex_session import CodexSession
from companion.codex_types import AvatarCue, CodexCommand, approval_result
from companion.config import Config
from companion.protocol import decode_audio, microphone_packet
from companion.store import Store
from scipy.signal import resample_poly


class FakeRPC:
    def __init__(self):
        self.calls = []
        self.replies = {"thread/turns/list": {"data": []}}
        self.events = asyncio.Queue()
        self.closed = False
        self.identity = {}
        self.responses = []

    async def call(self, method, params, **kwargs):
        self.calls.append((method, params))
        reply = self.replies.get(method, {})
        if callable(reply):
            return await reply(params)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    async def respond(self, identifier, result=None, error=None):
        self.responses.append((identifier, result, error))

    async def close(self):
        self.closed = True


class FakeRTC:
    def __init__(self, audio, failed):
        self.audio, self.failed = audio, failed
        self.closed = False
        self.muted = True
        self.packets = []
        self.answers = []
        self.ready = asyncio.Event()
        self.ready.set()

    async def offer(self):
        return "fixture SDP"

    async def answer(self, sdp):
        self.answers.append(sdp)

    async def wait_ready(self):
        await self.ready.wait()
        if self.closed:
            raise CodexError("closed")

    def set_muted(self, muted):
        self.muted = muted
        if muted:
            self.packets.clear()

    def feed(self, pcm):
        assert not self.muted and not self.closed
        self.packets.append(pcm)

    async def close(self):
        self.closed = True
        self.ready.set()


@pytest.fixture
async def session(tmp_path):
    events, packets = [], []

    async def event(e):
        events.append(e)

    async def audio(p):
        packets.append(p)

    store = Store(tmp_path / "history.sqlite")
    value = CodexSession(FakeRPC(), store, event, audio, rtc_factory=FakeRTC)
    value.thread = {"id": "task-1", "cwd": str(tmp_path)}
    value.test_events, value.test_packets = events, packets
    yield value
    await value.close()
    store.close()


async def test_voice_handoff_does_not_start_duplicate_task(session):
    await session.start_voice(uuid.uuid4())
    await session.voice_start
    await session.handle_event(
        {
            "method": "thread/realtime/item/completed",
            "params": {
                "threadId": "task-1",
                "item": {
                    "id": "spoken",
                    "type": "transcriptSegment",
                    "role": "user",
                    "text": "오늘도 hello、こんにちは",
                },
            },
        }
    )
    methods = [m for m, _ in session.rpc.calls]
    assert methods == ["thread/realtime/start"]
    assert session.rpc.calls[0][1]["clientManagedHandoffs"] is False
    assert session.rpc.calls[0][1]["backendReasoningStatus"] is False
    assert session.rpc.calls[0][1]["delegationAckFiller"] is False
    assert session.rpc.calls[0][1]["codexResponseHandoffMode"] == "thinking"
    assert "model" not in session.rpc.calls[0][1]
    assert session.store.conversations() == []


@pytest.mark.parametrize("active_turn", [None, "already-running"])
async def test_typed_message_during_voice_submits_once_to_selected_task(session, active_turn):
    await session.start_voice(uuid.uuid4())
    await session.voice_start
    session.active_turn = active_turn
    await session.text("오늘도 hello、こんにちは")
    method, params = session.rpc.calls[-1]
    assert method == ("turn/steer" if active_turn else "turn/start")
    assert params["threadId"] == "task-1"
    assert params["input"][0]["text"] == "오늘도 hello、こんにちは"
    assert len(session.rpc.calls) == 2
    if active_turn:
        assert params["expectedTurnId"] == active_turn


@pytest.mark.parametrize(
    "item,expected",
    [
        ({"type": "commandExecution", "command": "ls"}, "activity"),
        ({"type": "mcpToolCall", "tool": "search"}, "activity"),
        ({"type": "agentMessage", "phase": "commentary", "text": "Checking…"}, "activity"),
        ({"type": "agentMessage", "phase": "final_answer", "text": "Done."}, "conversation"),
        (
            {"type": "agentMessage", "phase": "partial_answer", "text": "First result"},
            "conversation",
        ),
        ({"type": "agentMessage", "text": "Legacy reply"}, "conversation"),
        ({"type": "agentMessage", "phase": "future_phase", "text": "Reply"}, "conversation"),
        (
            {
                "type": "agentMessage",
                "phase": "commentary",
                "text": "Which folder?",
                "questions": [{}],
            },
            "conversation",
        ),
        ({"type": "userMessage", "content": [{"type": "text", "text": "Run ls"}]}, "conversation"),
        ({"type": "transcriptSegment", "role": "assistant", "text": "All done."}, "conversation"),
    ],
)
def test_conversation_projection_uses_metadata_not_text(item, expected):
    row = project_item({"id": "item", **item})
    assert row["presentation"] == expected
    assert row["id"] == "item"


def test_structured_question_reply_displays_only_human_answer():
    payload = (
        "<send_user_message_question_reply>"
        + json.dumps(
            [
                {
                    "questionItemId": "question-1",
                    "question": "Approve?",
                    "answer": "Approve these phrases and destination",
                },
                {"questionItemId": "question-2", "question": "Language?", "answer": "한국어"},
            ]
        )
        + "</send_user_message_question_reply>"
    )
    item = {"id": "user", "type": "userMessage", "content": [{"type": "text", "text": payload}]}
    row = project_item(item)
    assert row["text"] == "Approve these phrases and destination\n\n한국어"
    assert row["presentation"] == "conversation"
    assert item["content"][0]["text"] == payload  # Canonical history is untouched.


def test_voice_delegation_is_background_activity_without_raw_envelope():
    payload = (
        "<realtime_delegation>\n<input>안녕 Ari &amp; Codex</input>\n"
        "<transcript_delta>user: 안녕 Ari &amp; Codex</transcript_delta>\n"
        "</realtime_delegation>"
    )
    item = {"id": "handoff", "type": "userMessage", "content": [{"type": "text", "text": payload}]}
    row = project_item(item)
    assert row["presentation"] == "activity"
    assert row["text"] == "Voice request sent to Codex: 안녕 Ari & Codex"
    assert item["content"][0]["text"] == payload
    # Discussing this syntax or sending malformed envelopes must stay visible.
    for text in ("Explain " + payload, "<realtime_delegation>hello</realtime_delegation>"):
        ordinary = project_item(
            {
                "id": "ordinary",
                "type": "userMessage",
                "content": [
                    {"type": "text", "text": text},
                ],
            }
        )
        assert ordinary["presentation"] == "conversation" and ordinary["text"] == text


@pytest.mark.parametrize(
    "text",
    [
        "<send_user_message_question_reply>invalid JSON</send_user_message_question_reply>",
        '<send_user_message_question_reply>[{"answer":"hello"}]</send_user_message_question_reply>',
        "Please explain <send_user_message_question_reply> to me.",
    ],
)
def test_reply_projection_preserves_ordinary_or_malformed_text(text):
    assert display_user_text(text) == text


async def test_progress_and_tools_stay_disposable_activity_without_audio(session):
    items = [
        {"id": "progress", "type": "agentMessage", "phase": "commentary", "text": "Checking"},
        {"id": "tool", "type": "commandExecution", "command": "ls", "status": "completed"},
        {
            "id": "answer",
            "type": "agentMessage",
            "phase": "final_answer",
            "text": "Found 39 entries.",
        },
    ]
    for item in items:
        await session.handle_event(
            {"method": "item/completed", "params": {"threadId": "task-1", "item": item}}
        )
    live = [e["item"] for e in session.test_events if e["type"] == "codex_item"]
    assert [r["presentation"] for r in live] == ["activity", "activity", "conversation"]
    reloaded = CodexTimeline()
    reloaded.load([{"type": "item", "position": i, "item": item} for i, item in enumerate(items)])
    assert reloaded.snapshot() == live
    assert session.test_packets == []
    assert session.store.conversations() == []
    assert not session.rpc.calls  # Display filtering never creates a summarization task.


async def test_voice_prompt_requests_outcomes_without_routine_tool_narration(session):
    await session.start_voice(uuid.uuid4())
    await session.voice_start
    prompt = session.rpc.calls[0][1]["prompt"]
    assert "Do not read commands, tool arguments, tool output, or progress logs aloud" in prompt
    assert "unless the user explicitly asks" in prompt
    assert "never announce success before it is confirmed" in prompt
    assert "user input or approval is needed" in prompt


async def test_typed_input_steers_active_turn_without_starting_another(session):
    session.active_turn = "already-running"
    await session.text("Change the direction")
    method, params = session.rpc.calls[-1]
    assert method == "turn/steer" and params["expectedTurnId"] == "already-running"
    session.rpc.replies["turn/steer"] = CodexError("turn changed")
    with pytest.raises(CodexError):
        await session.text("race with completion")
    assert session.submission_uncertain
    assert not any(method == "turn/start" for method, _ in session.rpc.calls)


async def test_late_turn_completion_keeps_new_turn_approval_and_tools(session):
    session.active_turn = "new"
    session.requests["request"] = {"id": "request"}
    session.automation_items.add("tool")
    await session.handle_event(
        {"method": "turn/completed", "params": {"threadId": "task-1", "turn": {"id": "old"}}}
    )
    assert session.active_turn == "new" and session.requests and session.automation_items


def test_late_text_delta_cannot_duplicate_persisted_completed_message():
    timeline = CodexTimeline()
    timeline.upsert({"id": "message", "type": "agentMessage", "text": "Done."}, completed=True)
    assert timeline.delta("message", "Done.") is None
    assert timeline.snapshot()[0]["text"] == "Done."


async def test_queued_text_cannot_move_to_another_task_and_stop_discards_it(session):
    from companion.codex_api import CodexCommands
    from companion.protocol import Command

    commands = CodexCommands(session)
    try:
        commands.submit(Command(type="text", text="for old task"))
        session.thread = {"id": "new-task"}
        await commands.queue.join()
        assert not any(method == "turn/start" for method, _ in session.rpc.calls)
        commands.submit(Command(type="text", text="cancel before dispatch"))
        commands.discard_pending()
        await commands.queue.join()
        assert not any(method == "turn/start" for method, _ in session.rpc.calls)
    finally:
        await commands.close()


async def test_validation_scope_rejects_other_tasks_and_preference_mutations(session):
    from companion.codex_validation import TransientCodexSettings

    original = session.store
    original.set_setting("codex_voice", "marin")
    session.store = TransientCodexSettings(original, "task-1")
    session.validation_thread = "task-1"
    with pytest.raises(CodexError):
        await session.select("other-task")
    for action in ("new", "archived", "settings"):
        with pytest.raises(CodexError):
            await session.command(CodexCommand(action=action))
    session.voices = ["marin", "cedar"]
    await session.command(CodexCommand(action="voice_select", voice="cedar"))
    assert original.setting("codex_voice") == "marin"
    assert session.store.setting("codex_voice") == "cedar"
    session.store = original


def test_validation_never_bypasses_shared_host_or_peer_checks():
    from companion.codex_validation import validation_ready

    checks = {
        key: True
        for key in (
            "verified_peer",
            "same_home",
            "desktop_socket",
            "host_version_matches",
            "threads",
            "voices",
            "account",
        )
    }
    report = {"shared_host": True, "available": False, "checks": checks}
    assert validation_ready(report)
    for key in checks:
        assert not validation_ready({**report, "checks": {**checks, key: False}})
    assert not validation_ready({**report, "shared_host": False})


async def test_validation_backend_is_ephemeral_and_normal_launch_restores_selection(
    tmp_path, monkeypatch
):
    async def unavailable(*_, **__):
        return {"available": False, "shared_host": False, "checks": {}}

    monkeypatch.setattr("companion.codex_api.probe", unavailable)
    for thread in ("test-task", None):
        cfg = Config(root=tmp_path, data=tmp_path, token="x" * 32, codex_validation_thread=thread)
        app = create_app(cfg)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": "Bearer " + cfg.token},
        ) as client:
            result = (await client.get("/backend")).json()
            assert result["selected"] == ("codex" if thread else "local")
            assert result["validation"]["active"] is bool(thread)
            assert not app.state.codex.permitted(result["codex"])
            if thread:
                assert (await client.put("/backend", json={"backend": "local"})).status_code == 409
            assert app.state.store.setting("backend", "local") == "local"
        app.state.store.close()


async def test_default_voice_is_not_assumed_from_v2_catalog(session):
    session.rpc.replies["thread/realtime/listVoices"] = {
        "voices": {"v2": ["marin"], "defaultV2": "marin"}
    }
    await session.catalog()
    assert session.voice_id == ""
    await session.start_voice(uuid.uuid4())
    await session.voice_start
    assert "voice" not in next(
        params for method, params in session.rpc.calls if method == "thread/realtime/start"
    )


async def test_mute_and_end_do_not_interrupt_task(session):
    await session.start_voice(uuid.uuid4())
    await session.voice_start
    session.active_turn = "working"
    await session.mute()
    assert session.voice_state == "muted" and session.active_turn == "working"
    assert session.owns_voice and session.capture is None
    await session.end_voice()
    assert session.active_turn == "working"
    assert not any(m == "turn/interrupt" for m, _ in session.rpc.calls)
    await session.stop_task()
    assert session.rpc.calls[-1] == ("turn/interrupt", {"threadId": "task-1", "turnId": "working"})


async def test_voice_auth_rejection_closes_capture_and_blocks_retries_without_stopping_task(
    session,
):
    session.active_turn = "working"
    await session.start_voice(uuid.uuid4())
    await session.voice_start
    await session.handle_event(
        {
            "method": "thread/realtime/error",
            "params": {
                "threadId": "task-1",
                "message": "realtime conversation requires API key auth",
            },
        }
    )
    assert session.voice_state == "off" and not session.owns_voice and session.capture is None
    assert session.active_turn == "working"
    assert "API-key authentication" in session.voice_unavailable_reason
    states = [e for e in session.test_events if e["type"] == "codex_state"]
    assert states[-1]["voice_unavailable_reason"] == session.voice_unavailable_reason
    before = len(session.rpc.calls)
    with pytest.raises(CodexError, match="API-key authentication"):
        await session.start_voice(uuid.uuid4())
    assert len(session.rpc.calls) == before
    await session.text("Continue by text")
    assert session.rpc.calls[-1][0] == "turn/steer"
    assert not any(method == "turn/interrupt" for method, _ in session.rpc.calls)
    await session.handle_event({"method": "account/updated", "params": {"authMode": "apiKey"}})
    assert session.voice_unavailable_reason is None
    assert session.capture is None and session.voice_state == "off"


async def test_unknown_voice_error_never_exposes_provider_body_or_claims_auth_failure(session):
    await session.handle_event(
        {
            "method": "thread/realtime/error",
            "params": {"threadId": "task-1", "message": "private-provider-payload"},
        }
    )
    assert session.voice_unavailable_reason is None
    assert "private-provider-payload" not in json.dumps(session.test_events)


async def test_release_during_voice_start_cannot_reopen_mic(session):
    ready = asyncio.Event()

    async def slow(_):
        await ready.wait()
        return {}

    session.rpc.replies["thread/realtime/start"] = slow
    await session.start_voice(uuid.uuid4())
    await asyncio.sleep(0)
    await asyncio.wait_for(session.mute(), 0.1)
    ready.set()
    await session.voice_start
    assert session.voice_state == "muted"
    assert not any(e.get("enabled") for e in session.test_events if e["type"] == "listening")


async def test_end_during_voice_start_stays_ended(session):
    ready = asyncio.Event()

    async def slow(_):
        await ready.wait()
        return {}

    session.rpc.replies["thread/realtime/start"] = slow
    await session.start_voice(uuid.uuid4())
    await asyncio.sleep(0)
    await session.end_voice()
    ready.set()
    await session.voice_start
    assert not session.owns_voice and session.voice_state == "off"
    assert [m for m, _ in session.rpc.calls] == ["thread/realtime/start", "thread/realtime/stop"]


async def test_uncertain_submission_requires_refresh_without_resending(session):
    session.rpc.replies["turn/start"] = CodexError("timeout")
    with pytest.raises(CodexError):
        await session.text("hello")
    with pytest.raises(CodexError):
        await session.text("hello")
    assert sum(m == "turn/start" for m, _ in session.rpc.calls) == 1
    session.rpc.replies["thread/timeline/list"] = {"data": []}
    await session.refresh_history()
    assert not session.submission_uncertain


async def test_desktop_voice_owner_blocks_second_microphone(session):
    session.voice_state = "in_desktop"
    with pytest.raises(CodexError):
        await session.start_voice(uuid.uuid4())
    await session.end_voice()
    assert not any(m == "thread/realtime/stop" for m, _ in session.rpc.calls)


async def test_unconfirmed_voice_shutdown_cannot_restart(session):
    session.owns_voice = True
    session.rpc.replies["thread/realtime/stop"] = CodexError("timeout")
    await session.end_voice()
    assert session.voice_state == "unknown"
    with pytest.raises(CodexError):
        await session.start_voice(uuid.uuid4())


async def test_microphone_sessions_detect_gaps_and_discard_late_packets(session):
    capture = str(uuid.uuid4())
    session.owns_voice = True
    session.capture = capture
    await session.input_audio(microphone_packet(str(uuid.uuid4()), 0, 0, b"\x01\x00" * 40))
    assert session.input_queue.empty()
    await session.input_audio(microphone_packet(capture, 0, 0, b"\x01\x00" * 40))
    with pytest.raises(CodexError):
        await session.input_audio(microphone_packet(capture, 2, 80, b"\x01\x00" * 40))
    assert session.capture is None and not session.owns_voice


async def test_canonical_history_scans_pages_before_determining_ownership(session):
    pages = [
        {
            "data": [
                {
                    "type": "realtime",
                    "position": 0,
                    "item": {"id": "s", "realtimeSessionId": "v", "type": "realtimeSessionStarted"},
                }
            ],
            "nextCursor": "page2",
        },
        {
            "activeRealtimeSessionAtPageStart": "v",
            "data": [
                {
                    "type": "realtime",
                    "position": 1,
                    "item": {
                        "id": "voice",
                        "type": "transcriptSegment",
                        "role": "user",
                        "text": "Hello",
                    },
                },
                {
                    "type": "realtime",
                    "position": 2,
                    "item": {"id": "end", "type": "realtimeSessionClosed"},
                },
            ],
        },
    ]

    async def page(_):
        return pages.pop(0)

    session.rpc.replies["thread/timeline/list"] = page
    await session.refresh_history()
    assert session.voice_state == "off" and not session.history_loading
    assert [x["id"] for x in session.timeline.snapshot()] == ["voice"]


async def test_history_refresh_preserves_new_events_during_read(session):
    async def page(_):
        await session.handle_event(
            {
                "method": "item/completed",
                "params": {
                    "threadId": "task-1",
                    "item": {"id": "new", "type": "agentMessage", "text": "fresh"},
                },
            }
        )
        return {"data": []}

    session.rpc.replies["thread/timeline/list"] = page
    await session.refresh_history()
    assert session.timeline.snapshot()[0]["text"] == "fresh"


@pytest.mark.parametrize("active", [True, False])
async def test_newest_history_page_decides_voice_ownership(session, active):
    pages = [
        {
            "activeRealtimeSessionAtPageStart": "current" if active else None,
            "data": [
                {
                    "type": "item",
                    "position": 10,
                    "item": {
                        "id": "latest",
                        "type": "agentMessage",
                        "text": "Recent reply",
                    },
                }
            ],
            "nextCursor": "older",
        },
        {
            "data": [
                {
                    "type": "realtime",
                    "position": 1,
                    "item": {
                        "id": "old-voice",
                        "realtimeSessionId": "old",
                        "type": "realtimeSessionClosed" if active else "realtimeSessionStarted",
                    },
                }
            ],
        },
    ]

    async def page(_):
        return pages.pop(0)

    session.rpc.replies["thread/timeline/list"] = page
    # A completed previous preparation task must not disable ownership refresh.
    session.voice_start = asyncio.create_task(asyncio.sleep(0))
    await session.voice_start
    await session.refresh_history()
    assert session.voice_state == ("in_desktop" if active else "off")
    assert session.history_session == ("current" if active else None)


async def test_voice_started_during_history_reload_wins_over_older_snapshot(session):
    async def page(_):
        await session.handle_event(
            {
                "method": "thread/realtime/started",
                "params": {
                    "threadId": "task-1",
                    "realtimeSessionId": "new-owner",
                },
            }
        )
        return {"activeRealtimeSessionAtPageStart": None, "data": []}

    session.rpc.replies["thread/timeline/list"] = page
    await session.refresh_history()
    assert session.voice_state == "in_desktop" and session.history_session == "new-owner"


def test_newest_first_pages_keep_recent_voice_replies_and_live_items():
    timeline = CodexTimeline(limit=3)
    # The actual host returns newer pages first; each page is chronological.
    for page in ([8, 9], [6, 7], [4, 5]):
        timeline.load(
            [
                {
                    "type": "realtime",
                    "position": position,
                    "item": {
                        "id": str(position),
                        "type": "transcriptSegment",
                        "role": "assistant",
                        "text": f"reply {position}",
                    },
                }
                for position in page
            ]
        )
    assert [row["id"] for row in timeline.snapshot()] == ["7", "8", "9"]
    assert timeline.truncated and len(timeline.completed) == 3
    timeline.upsert({"id": "live", "type": "agentMessage", "text": "New reply"})
    timeline.load(
        [
            {
                "type": "item",
                "position": 0,
                "item": {
                    "id": "old",
                    "type": "agentMessage",
                    "text": "Old reply",
                },
            }
        ]
    )
    assert [row["id"] for row in timeline.snapshot()] == ["8", "9", "live"]
    assert timeline.delta("old", "stale") is None


def test_history_deduplicates_promoted_references_and_is_bounded():
    timeline = CodexTimeline(limit=2)
    entries = [
        {
            "type": "item",
            "position": 2,
            "item": {"id": "a", "type": "agentMessage", "text": "Answer"},
        },
        {
            "type": "realtime",
            "position": 3,
            "item": {"id": "ref", "type": "bemItemPromoted", "item_id": "a"},
        },
        {
            "type": "realtime",
            "position": 1,
            "item": {"id": "u", "type": "transcriptSegment", "role": "user", "text": "Hi"},
        },
    ]
    timeline.load(entries)
    timeline.load(entries)
    assert [x["id"] for x in timeline.snapshot()] == ["u", "a"]
    timeline.upsert({"id": "a", "type": "agentMessage", "text": ""})
    assert timeline.rows["a"]["text"] == "Answer"
    timeline.upsert({"id": "b", "type": "agentMessage", "text": "next"})
    assert len(timeline.rows) == 2 and timeline.truncated


async def test_cross_client_approval_resolution_is_once(session):
    message = {
        "id": 7,
        "method": "item/fileChange/requestApproval",
        "params": {"threadId": "task-1", "reason": "test"},
    }
    await session.handle_event(message)
    await session.handle_event(
        {"method": "serverRequest/resolved", "params": {"threadId": "task-1", "requestId": 7}}
    )
    with pytest.raises(ValueError):
        await session.command(CodexCommand(action="approval", request_id="7", decision="accept"))
    assert session.rpc.responses == []
    await session.handle_event(message)
    await session.command(CodexCommand(action="approval", request_id="7", decision="decline"))
    with pytest.raises(ValueError):
        await session.command(CodexCommand(action="approval", request_id="7", decision="accept"))
    assert session.rpc.responses == [(7, {"decision": "decline"}, None)]


def test_approval_never_expands_permission_scope_or_invents_form_data():
    requested = {"fileSystem": {"write": ["/tmp/example"]}}
    assert approval_result(
        "item/permissions/requestApproval", "accept", {"permissions": requested}
    ) == {"permissions": requested, "scope": "turn"}
    assert (
        approval_result("item/permissions/requestApproval", "decline", {"permissions": requested})[
            "permissions"
        ]
        == {}
    )
    with pytest.raises(ValueError):
        approval_result("mcpServer/elicitation/request", "accept", {"mode": "form"})
    with pytest.raises(ValueError):
        approval_result(
            "item/fileChange/requestApproval", "accept", {"availableDecisions": ["decline"]}
        )


async def test_settings_inherit_permissions_and_reject_fake_plugin_restrictions(session):
    session.models = [
        {"model": "test-model", "supportedReasoningEfforts": [{"reasoningEffort": "low"}]}
    ]
    await session.command(CodexCommand(action="settings", model="test-model", effort="low"))
    assert session.rpc.calls[-1][1] == {
        "threadId": "task-1",
        "model": "test-model",
        "effort": "low",
    }
    with pytest.raises(ValueError):
        await session.command(CodexCommand(action="settings", disabled_plugins=["computer-use"]))


def chunk(samples, rate=24000, item="a", channels=1):
    return {
        "itemId": item,
        "data": base64.b64encode(np.asarray(samples, dtype="<i2").tobytes()).decode(),
        "sampleRate": rate,
        "numChannels": channels,
        "samplesPerChannel": len(samples) // channels,
    }


async def test_audio_continuous_resampling_order_and_actual_ack_window():
    events, packets = [], []

    async def event(kind, **values):
        events.append((kind, values))

    async def output(p):
        packets.append(decode_audio(p))

    audio = CodexAudio(event, output)
    source = (np.sin(np.arange(44100) * 0.017) * 12000).astype("<i2")
    try:
        for part in np.array_split(source, 73):
            await audio.feed(chunk(part, 44100))
        audio.finish()
        await asyncio.sleep(0.02)
        assert audio.sent <= 24000 and packets
        pcm = np.frombuffer(b"".join(p[3] for p in packets), "<i2")
        expected = resample_poly(source.astype(np.float32) / 32768, 80, 147)
        assert len(pcm) == 24000
        assert np.max(np.abs(pcm.astype(float) / 32768 - expected)) < 1 / 32768
        assert all(len(p[3]) <= 3840 for p in packets)
        assert [p[1] for p in packets] == list(range(len(packets)))
        await audio.feed(chunk(np.zeros(4800), item="b"))
        await asyncio.sleep(0.01)
        assert audio.sent == 24000  # Delivery waits for actually played samples.
        audio.acknowledge("wrong-turn", 24000)
        assert audio.acked == 0
        audio.acknowledge(audio.turn, 24000)
        await asyncio.sleep(0.01)
        assert audio.sent == 28800
        old = audio.turn
        await audio.clear()
        await audio.feed(chunk([1, 2], item="b"))
        assert audio.turn is None and old is not None
    finally:
        await audio.clear()


@pytest.mark.parametrize(
    "change",
    [
        {"sampleRate": 0},
        {"numChannels": 3},
        {"data": "bad base64"},
        {"itemId": None},
        {"samplesPerChannel": 1000},
    ],
)
async def test_invalid_audio_is_rejected(change):
    async def sink(*_, **__):
        pass

    audio = CodexAudio(sink, sink)
    with pytest.raises(CodexError):
        await audio.feed(chunk([1, 2]) | change)
    await audio.clear()


async def test_audio_overflow_and_adaptive_jitter_are_bounded():
    async def sink(*_, **__):
        pass

    audio = CodexAudio(sink, sink)
    try:
        await audio.feed(chunk(np.zeros(24000 * 6)))
        with pytest.raises(CodexError):
            await audio.feed(chunk([1, 2]))
        for _ in range(100):
            audio.underrun()
        assert audio.threshold == 7200
    finally:
        await audio.clear()


async def test_negative_cues_cannot_trigger_cheerful_gestures(session):
    session.owns_voice = True
    session.audio.turn = str(uuid.uuid4())
    session.avatar_mode = "model"
    await session.present({"emotion": "concerned", "action": "twirl"})
    assert [e["type"] for e in session.test_events] == ["expression"]
    with pytest.raises(ValueError):
        AvatarCue(emotion="anything", action="run_shell")


def test_acceptance_is_bound_to_exact_build_and_complete_evidence(tmp_path):
    report = tmp_path / "acceptance.json"
    fingerprint = {"schema": "abc"}
    report.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "checks": {g: {"passed": True, "evidence": "fixture"} for g in ACCEPTANCE_GATES},
            }
        )
    )
    assert accepted_report(report, fingerprint)
    assert not accepted_report(report, {"schema": "new"})
    report.write_text("{}")
    assert not accepted_report(report, fingerprint)


async def test_unavailable_codex_cannot_change_saved_backend(tmp_path, monkeypatch):
    async def unavailable(*_, **__):
        return {"available": False, "reasons": ["No shared host"]}

    monkeypatch.setattr("companion.codex_api.probe", unavailable)
    cfg = Config(root=tmp_path, data=tmp_path, token="x" * 32, demo=True)
    app = create_app(cfg)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Authorization": "Bearer " + cfg.token},
    ) as client:
        assert (await client.put("/backend", json={"backend": "codex"})).status_code == 409
        assert (await client.get("/backend")).json()["selected"] == "local"
        assert (
            await client.get("/codex/status", headers={"Origin": "http://untrusted.test"})
        ).status_code == 403
    app.state.store.close()


async def test_rpc_routes_server_requests_and_uncertain_timeout_without_retry(tmp_path):
    rpc = CodexRPC(tmp_path / "unused.sock")

    class Socket:
        def __init__(self):
            self.outbound = []
            self.inbound = asyncio.Queue()

        async def send(self, value):
            self.outbound.append(json.loads(value))

        def __aiter__(self):
            return self

        async def __anext__(self):
            return await self.inbound.get()

        async def close(self):
            pass

    sock = Socket()
    rpc.socket = sock
    rpc.reader = asyncio.create_task(rpc._read())
    try:
        call = asyncio.create_task(rpc.call("test", {}))
        await asyncio.sleep(0)
        await sock.inbound.put(json.dumps({"id": 1, "result": {"ok": True}}))
        assert await call == {"ok": True}
        await sock.inbound.put(
            json.dumps({"id": 5, "method": "item/fileChange/requestApproval", "params": {}})
        )
        assert (await rpc.events.get())["id"] == 5
        with pytest.raises(CodexError):
            await rpc.call("turn/start", {}, timeout=0.01)
        assert len(sock.outbound) == 2 and not rpc.pending
    finally:
        await rpc.close()


async def test_rpc_rejects_regular_file_in_place_of_socket(tmp_path):
    path = tmp_path / "host"
    path.touch()
    with pytest.raises(CodexError):
        await CodexRPC(path).connect()


async def test_new_task_registers_only_presentation_tool_and_inherits_permissions(
    session, tmp_path
):
    session.rpc.replies["thread/start"] = {"thread": {"id": "new-task"}}
    session.rpc.replies["thread/read"] = {"thread": {"id": "new-task", "cwd": str(tmp_path)}}
    session.rpc.replies["thread/resume"] = {"thread": {"id": "new-task", "cwd": str(tmp_path)}}
    await session.command(CodexCommand(action="new", cwd=str(tmp_path), project_id="project"))
    params = next(p for m, p in session.rpc.calls if m == "thread/start")
    assert params["dynamicTools"][0]["type"] == "function"
    assert params["dynamicTools"][0]["name"] == "ari_avatar_present"
    assert (
        not {"permissions", "sandbox", "approvalPolicy", "model", "developerInstructions"}
        & params.keys()
    )
    assert params["projectId"] == "project"


async def test_archived_task_is_read_without_resuming(session, tmp_path):
    session.archived = True
    session.rpc.replies["thread/read"] = {"thread": {"id": "archive", "cwd": str(tmp_path)}}
    await session.select("archive")
    assert session.read_only
    assert not any(m == "thread/resume" for m, _ in session.rpc.calls)
    with pytest.raises(CodexError):
        await session.text("continue")
    with pytest.raises(CodexError):
        await session.start_voice(uuid.uuid4())


async def test_end_voice_cannot_steal_desktop_ownership(session):
    session.voice_state = "in_desktop"
    await session.end_voice()
    assert session.voice_state == "in_desktop"
    with pytest.raises(CodexError):
        await session.start_voice(uuid.uuid4())


async def test_simultaneous_tools_keep_overlay_click_through_until_all_complete(session):
    for id in ["a", "b"]:
        await session.handle_event(
            {
                "method": "item/started",
                "params": {
                    "threadId": "task-1",
                    "item": {"id": id, "type": "mcpToolCall", "tool": "computer"},
                },
            }
        )
    await session.handle_event(
        {
            "method": "item/completed",
            "params": {"threadId": "task-1", "item": {"id": "a", "type": "mcpToolCall"}},
        }
    )
    assert session.test_events[-1] == {
        "version": 1,
        "provider": "codex",
        "type": "codex_automation",
        "active": True,
    }
    await session.handle_event(
        {
            "method": "item/completed",
            "params": {"threadId": "task-1", "item": {"id": "b", "type": "mcpToolCall"}},
        }
    )
    assert session.test_events[-1]["active"] is False


async def test_saved_backend_and_voice_survive_store_restart(tmp_path, monkeypatch):
    async def available(*_, **__):
        return {"available": True}

    monkeypatch.setattr("companion.codex_api.probe", available)
    cfg = Config(root=tmp_path, data=tmp_path, token="x" * 32, demo=True)
    for initial in (True, False):
        app = create_app(cfg)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": "Bearer " + cfg.token},
        ) as client:
            if initial:
                assert (await client.put("/backend", json={"backend": "codex"})).status_code == 200
                app.state.store.set_setting("codex_voice", "marin")
            else:
                assert (await client.get("/backend")).json()["selected"] == "codex"
                assert app.state.store.setting("codex_voice", "") == "marin"
            assert app.state.store.conversations() == []
        app.state.store.close()


async def test_ari_does_not_reject_another_clients_tool_request(session):
    for thread in ["someone-else", "task-1"]:
        await session.handle_event(
            {
                "id": 10,
                "method": "item/tool/call",
                "params": {"threadId": thread, "tool": "desktop-tool"},
            }
        )
    assert session.rpc.responses == []


async def test_slow_history_command_does_not_block_mute_or_playback_ack(session):
    from companion.codex_api import CodexCommands
    from companion.protocol import Command

    pending = asyncio.Event()

    async def slow(_):
        await pending.wait()
        return {"data": []}

    session.rpc.replies["thread/timeline/list"] = slow
    commands = CodexCommands(session)
    try:
        commands.submit(Command(type="codex", codex=CodexCommand(action="history_more")))
        session.cursor = "next"
        await asyncio.sleep(0)
        session.owns_voice = True
        await asyncio.wait_for(session.mute(), 0.1)
        assert session.voice_state == "muted"
        for _ in range(8):
            commands.submit(Command(type="text", text="queued"))
        with pytest.raises(CodexError):
            commands.submit(Command(type="text", text="overflow"))
    finally:
        await commands.close()


async def test_end_before_start_rpc_dispatch_does_not_start_voice_after_stop(session):
    await session.start_voice(uuid.uuid4())
    await session.end_voice()
    await asyncio.sleep(0)
    assert not any(m == "thread/realtime/start" for m, _ in session.rpc.calls)
    assert not session.owns_voice and session.voice_state == "off"


async def test_stop_rejects_future_audio_until_user_speaks_again(session):
    session.owns_voice = True
    await session.stop_task()
    await session.handle_event(
        {
            "method": "thread/realtime/outputAudio/delta",
            "params": {"threadId": "task-1", "audio": chunk([1, 2], item="late")},
        }
    )
    assert session.audio.turn is None
    await session.handle_event(
        {
            "method": "thread/realtime/item/started",
            "params": {
                "threadId": "task-1",
                "item": {
                    "id": "next-user",
                    "type": "transcriptSegment",
                    "role": "user",
                    "text": "",
                },
            },
        }
    )
    assert not session.suppress_audio


@pytest.mark.parametrize("device_field", ["d", "D"])
def test_shared_host_evidence_requires_actual_desktop_socket_peer(device_field):
    from companion.codex_compat import desktop_socket_attached

    path = "/tmp/codex.sock"
    fixture = """p10
ccodex
f3
D0x123
n/tmp/codex.sock
f4
D0x456
n->0x789
p20
cChatGPT
f10
D0x789
n->0x456"""
    fixture = fixture.replace("\nD", "\n" + device_field)
    assert desktop_socket_attached(fixture, path)
    assert desktop_socket_attached(fixture, path, server_pid=10)
    assert not desktop_socket_attached(fixture, path, server_pid=99)
    assert not desktop_socket_attached(fixture.replace("n->0x456", "n->0x999"), path)
    assert not desktop_socket_attached(
        fixture.replace("n/tmp/codex.sock", "n/tmp/other.sock"), path
    )
    assert not desktop_socket_attached(fixture.replace("cChatGPT", "cOtherApp"), path)


def test_unicode_history_display_is_batched_below_native_limit_without_loss():
    from companion.codex_history import display_batches

    rows = [{"id": str(i), "role": "assistant", "text": "한日😀" * 12000} for i in range(35)]
    batches = list(display_batches(rows))
    assert len(batches) > 1
    assert [row for batch in batches for row in batch] == rows
    assert all(
        len(json.dumps({"messages": batch}, ensure_ascii=False).encode()) < 1024 * 1024
        for batch in batches
    )
    assert list(display_batches([])) == [[]]


def test_tool_catalog_projection_excludes_large_executable_schemas():
    from companion.codex_history import tool_catalog_summary

    value = {
        "data": [
            {
                "name": "computer",
                "runtimeStatus": "ready",
                "tools": {"click": {"schema": "x" * 1200000}},
                "toolsError": None,
            }
        ]
    }
    result = tool_catalog_summary(value)
    assert result["data"][0]["toolCount"] == 1
    assert result["data"][0]["runtimeStatus"] == "ready"
    assert len(json.dumps(result)) < 1000
    assert "schema" not in json.dumps(result)


async def test_live_turn_is_reconciled_when_persisted_timeline_lags(session):
    session.rpc.replies["thread/timeline/list"] = {
        "data": [
            {"type": "turnStarted", "turn_id": "old", "position": 1},
            {"type": "turnCompleted", "turn_id": "old", "position": 2},
        ]
    }
    session.rpc.replies["thread/turns/list"] = {"data": [{"id": "live", "status": "inProgress"}]}
    await session.refresh_history()
    assert session.active_turn == "live" and not session.submission_uncertain
    await session.text("list files")
    assert session.rpc.calls[-1][0] == "turn/steer"
    assert session.rpc.calls[-1][1]["expectedTurnId"] == "live"
    assert not any(m == "turn/start" for m, _ in session.rpc.calls)


async def test_newer_turn_event_wins_over_concurrent_state_snapshot(session):
    async def read(_):
        await session.handle_event(
            {"method": "turn/started", "params": {"threadId": "task-1", "turn": {"id": "new"}}}
        )
        return {"data": [{"id": "old", "status": "inProgress"}]}

    session.rpc.replies["thread/turns/list"] = read
    await session.reconcile_task_state()
    assert session.active_turn == "new"


async def test_webrtc_authentication_is_delegated_to_host_and_media_is_not_duplicated(session):
    capture = str(uuid.uuid4())
    await session.start_voice(capture)
    await session.voice_start
    params = session.rpc.calls[0][1]
    assert params["transport"] == {"type": "webrtc", "sdp": "fixture SDP"}
    assert not any(key in params for key in ("apiKey", "accessToken", "authorization"))
    pcm = b"\x05\x00" * 1280
    await session.input_audio(microphone_packet(capture, 0, 0, pcm))
    await session.input_queue.join()
    assert session.rtc.packets == [pcm]
    assert not any(m == "thread/realtime/appendAudio" for m, _ in session.rpc.calls)
    await session.handle_event(
        {
            "method": "thread/realtime/outputAudio/delta",
            "params": {"threadId": "task-1", "audio": chunk([3] * 200)},
        }
    )
    assert session.audio.turn is None
    await session.rtc.audio(b"\x05\x00" * 480)
    assert session.audio.turn is not None
    await session.mute()
    assert session.rtc.muted and session.rtc.packets == []
    assert not session.rpc.closed


async def test_start_acknowledgement_does_not_open_microphone_before_rtc_ready(session):
    transport = FakeRTC(lambda pcm: None, lambda message: None)
    transport.ready.clear()
    session.rtc_factory = lambda audio, failed: transport
    await session.start_voice(uuid.uuid4())
    await asyncio.sleep(0)
    assert not session.owns_voice and session.voice_state == "starting"
    assert not any(e.get("enabled") for e in session.test_events if e["type"] == "listening")
    await session.handle_event(
        {"method": "thread/realtime/sdp", "params": {"threadId": "task-1", "sdp": "answer"}}
    )
    assert transport.answers == ["answer"]
    await session.end_voice()
    await session.voice_start
    assert transport.closed and session.voice_state == "off"
    assert not any(e.get("enabled") for e in session.test_events if e["type"] == "listening")


async def test_stale_rtc_audio_and_failures_cannot_affect_replacement_session(session):
    await session.start_voice(uuid.uuid4())
    await session.voice_start
    old = session.rtc
    await session.end_voice()
    await session.start_voice(uuid.uuid4())
    await session.voice_start
    await old.audio(b"\x01\x00" * 480)
    await old.failed("old failure")
    assert session.owns_voice and session.audio.turn is None
    await session.rtc.audio(b"\x02\x00" * 480)
    first = session.audio.turn
    await session.stop_task()
    await session.rtc.audio(b"\x03\x00" * 480)
    assert session.audio.turn is None
    session.suppress_audio = False
    await session.rtc.audio(b"\x04\x00" * 480)
    assert session.audio.turn and session.audio.turn != first


async def test_rtc_failure_ends_voice_without_closing_task_connection(session):
    await session.start_voice(uuid.uuid4())
    await session.voice_start
    rtc = session.rtc
    session.active_turn = "working"
    await rtc.failed("Voice connection failed")
    assert rtc.closed and session.rtc is None and session.capture is None
    assert session.active_turn == "working" and not session.rpc.closed
    assert not any(m == "turn/interrupt" for m, _ in session.rpc.calls)


async def test_unmute_keeps_existing_peer_audio_callbacks_live(session):
    await session.start_voice(uuid.uuid4())
    await session.voice_start
    rtc = session.rtc
    epoch = session.voice_epoch
    await session.mute()
    await session.start_voice(uuid.uuid4())
    assert session.rtc is rtc and session.voice_epoch == epoch and not rtc.muted
    await rtc.audio(b"\x08\x00" * 480)
    assert session.audio.turn is not None
    assert sum(m == "thread/realtime/start" for m, _ in session.rpc.calls) == 1


async def test_reflected_audio_completion_does_not_break_next_rtc_phrase(session):
    await session.start_voice(uuid.uuid4())
    await session.voice_start
    await session.rtc.audio(b"\x01\x00" * 480)
    turn = session.audio.turn
    await session.handle_event(
        {
            "method": "thread/realtime/itemAdded",
            "params": {
                "threadId": "task-1",
                "item": {"type": "response.audio.done"},
            },
        }
    )
    await session.rtc.audio(b"\x02\x00" * 480)
    assert session.audio.turn == turn and session.audio.queued == 960


async def test_closing_after_native_disconnect_releases_media_and_host_subscription(session):
    await session.start_voice(uuid.uuid4())
    await session.voice_start
    rtc = session.rtc

    async def disconnected(_):
        raise RuntimeError("Native websocket is already closed")

    session.send = disconnected
    await session.close()
    assert rtc.closed and session.rpc.closed and session.capture is None
    assert session.input_worker is None and session.audio.worker is None
    assert ("thread/realtime/stop", {"threadId": "task-1"}) in session.rpc.calls
    assert ("thread/unsubscribe", {"threadId": "task-1"}) in session.rpc.calls
    calls = len(session.rpc.calls)
    await session.close()
    assert len(session.rpc.calls) == calls
