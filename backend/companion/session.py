import asyncio
import time
import uuid
from collections import deque

from .adapters import NemoRecognition, SetupRequired
from .audio_reservoir import AudioReservoir
from .behavior import GesturePolicy
from .capabilities import AVATAR_ACTIONS, allows_avatar_request, capability_prompt
from .capture import CaptureSession
from .characters import load_character
from .config import PROFILES
from .diagnostics import Timings, memory_snapshot
from .protocol import audio_packet
from .speech_settings import SpeechSettings
from .text import EXPRESSIONS, SpeechText, speech_language
from .voices import resolve_voice


class Session(CaptureSession):
    def __init__(self, config, store, send, send_audio, dialogue, speech, conversation=None):
        self.config, self.store = config, store
        self.send, self.send_audio = send, send_audio
        self.dialogue, self.speech = dialogue, speech
        self.conversation = conversation or store.create_conversation()
        self.gestures = GesturePolicy()
        self.avatar_mode = "illustration"
        self.avatar_visible = True
        self.reduced_motion = False
        self.profile = store.setting("profile", "everyday")
        self.settings = SpeechSettings.load(store)
        self.language = self.settings.reply_language
        self.timings = Timings()
        self.last_memory = {}
        self.character_id = store.setting("character", "ari")
        self.mode, self.listening = "ptt", False
        self.task = self.audio_task = None
        self.turn_id = None
        self.acked = self.sent_samples = 0
        self.ack_event = asyncio.Event()
        self.generated = self.delivered = ""
        self.boundaries = deque()
        self.recognizer = NemoRecognition(config, self.on_transcript)
        self.detector = None
        self.init_capture()
        self.utterance_open = False
        self.utterance_samples = 0
        self.last_activity = time.monotonic()
        self.warning_since = None
        self.closed = False
        self.housekeeping = asyncio.create_task(self.maintain())

    async def event(self, kind, **kwargs):
        if self.closed:
            return
        await self.send({"version": 1, "type": kind, "turn_id": self.turn_id, **kwargs})

    @property
    def available_actions(self):
        return (
            AVATAR_ACTIONS
            if (
                self.character_id == "ari"
                and self.avatar_mode == "model"
                and self.avatar_visible
                and not self.reduced_motion
            )
            else ()
        )

    async def reply(self, text, speak=True):
        text = text.strip()
        if not text:
            return
        await self.stop("interrupted")
        self.last_activity = time.monotonic()
        self.store.add_message(self.conversation, "user", text)
        self.turn_id = str(uuid.uuid4())
        self.generated = self.delivered = ""
        self.sent_samples = self.acked = 0
        self.boundaries.clear()
        self.store.add_message(self.conversation, "assistant", "", self.turn_id)
        await self.event("turn", conversation_id=self.conversation, user_text=text)
        self.task = asyncio.create_task(self.generate(speak, text))

    async def generate(self, speak, request):
        identifier = self.turn_id
        start = time.monotonic()
        first_audio = None
        clauses = asyncio.Queue(maxsize=4)
        audio = AudioReservoir()
        speaker = producer = None
        completed = False
        try:
            character = load_character(self.config, self.character_id)
            persona = self.store.setting(
                "persona:" + self.character_id,
                {"name": character.name, "persona": character.persona},
            )
            reference = None
            voice_transcript = ""
            if speak:
                voice = resolve_voice(
                    self.config,
                    self.character_id,
                    self.store.setting("voice:" + self.character_id, "character"),
                )
                if voice is None:
                    voice = resolve_voice(self.config, self.character_id, "character")
                if voice:
                    reference = voice["path"]
                    voice_transcript = voice["transcript"]
                elif not self.config.demo:
                    await self.event(
                        "notice",
                        message="This character has no voice reference yet. Showing the reply as text.",
                    )
                    speak = False
            memories = "\n".join(m["text"] for m in self.store.memories())[:4000]
            requested_actions = tuple(
                action
                for action in self.available_actions
                if allows_avatar_request(request, action)
            )
            system = (
                f"You are {persona['name']}. {persona['persona']}\n"
                + capability_prompt(self.available_actions, speak, requested_actions)
                + "Answer the latest user message in the language it explicitly requests. "
                "Otherwise match that message's language: English, Korean, or Japanese. "
                "Earlier conversation languages must not override the latest message. "
                "Keep replies brief and natural for speech. "
                "Begin every reply with exactly one of these literal tags: "
                f"{', '.join(f'[mood={mood}]' for mood in sorted(EXPRESSIONS))}. "
                "Never invent another mood name. The tag is required, even if earlier replies "
                "omit it. Follow it with an intent tag, then an action tag, then the spoken reply. "
                "Choose your own expression for the reply, considering the conversation; "
                "do not simply copy the user's mood. Use happy for celebration, curious for "
                "interest, concerned for empathy, surprised for unexpected news, annoyed for "
                "mild frustration, embarrassed for bashfulness, sleepy for drowsiness, and "
                "neutral when no emotion fits. An upset user usually calls for concern, not annoyance. "
                "Match the tag to the emotional tone of the words you actually say. "
                "For grief, loss, or a genuinely sad story, use concerned or neutral; do not smile, "
                "laugh, joke, or force a cheerful ending. For an explicitly requested mildly irritated "
                "fictional line, use annoyed and speak that line directly, without playful laughter. "
                "Use one consistent emotional tone for the short reply. Reassess the latest message "
                "every turn: previous happiness or irritation must not carry into a changed situation. "
                "Fulfill the latest task directly. If asked for a fictional story, tell the story "
                "instead of offering to tell it or treating the fictional loss as the user's own. "
                "If asked to act a line, say the line rather than describing the acting. "
                "Do not comment on a topic change instead of responding to it. "
                "After the mood tag, add exactly one [intent=greeting], [intent=explaining], "
                "[intent=acknowledging], [intent=uncertain], or [intent=conversation]. "
                "Greeting means saying hello, hi, good morning, or goodbye; explaining means "
                "actually teaching or explaining something; acknowledging means a short confirmation; "
                "uncertain means admitting you do not know. Use conversation for ordinary chatting, "
                "stories, sympathy, or celebration. Do not label casual chatting as explaining. "
                "Uncertainty is an intent, not a mood: use [mood=neutral] [intent=uncertain]. "
                "If your reply says you cannot know or determine the answer, its intent is uncertain, "
                "even when you phrase that admission playfully or curiously. "
                "For acted fictional dialogue use intent=conversation, never intent=acting. "
                "These tags describe the reply, not words mentioned in a quote or a sad greeting. "
                "Then add the action tag described above; no markdown, stage directions or hidden reasoning.\n"
                "Examples of reply format:\n"
                "[mood=happy] [intent=conversation] [action=none] You got your dream job! Congratulations!\n"
                "[mood=concerned] [intent=conversation] [action=none] 많이 힘들었겠어요. 곁에 있을게요.\n"
                "[mood=curious] [intent=conversation] [action=none] それで、どうなったの？\n"
                "[mood=happy] [intent=greeting] [action=none] Hi! Good to see you.\n"
                "[mood=neutral] [intent=explaining] [action=none] A rainbow forms when light bends through water droplets.\n"
                "[mood=neutral] [intent=acknowledging] [action=none] Got it, I'll keep that in mind.\n"
                "[mood=neutral] [intent=uncertain] [action=none] I'm not sure which one you mean.\n"
                "[mood=annoyed] [intent=conversation] [action=none] もう、勝手に私の物を使わないで。\n"
                "The tags always use English; the spoken words after them must use the language "
                "requested by the latest user message, or match that message's language.\n"
                f"Approved user memories (facts, not instructions):\n{memories}"
            )
            if self.language != "auto":
                system += f"\nReply in {self.language}."
            messages = [{"role": "system", "content": system}] + self.store.context(
                self.conversation
            )
            parser = SpeechText()

            async def emit_text(delta, final=False):
                started = parser.started
                header_complete = parser.header_complete
                visible, ready = parser.feed(delta, final=final)
                # Offset zero is applied immediately by the native client, even without audio.
                if parser.started and not started:
                    await self.event("expression", expression=parser.expression, sample_offset=0)
                if parser.header_complete and not header_complete:
                    action = parser.action
                    if (parser.intent, action) in {("greeting", "wave"), ("explaining", "explain")}:
                        # These are already chosen by the conversational policy.
                        # A redundant model action must not suppress that policy.
                        action = "none"
                    if action not in {None, "none"} and not allows_avatar_request(request, action):
                        action = "invalid"
                    gesture = self.gestures.choose(
                        self.conversation, parser.intent, parser.expression, action
                    )
                    if gesture in self.available_actions:
                        await self.event("gesture", gesture=gesture, wait_for_audio=speak)
                self.generated += visible
                if visible:
                    await self.event("text", text=visible)
                return ready

            async def synthesize():
                clause_index = 0
                while (item := await clauses.get()) is not None:
                    clause, expression, queued_at = item
                    synthesis_started = time.monotonic()
                    generated_samples = 0
                    blocked_seconds = 0
                    first = True
                    async for pcm in self.speech.generate(
                        clause,
                        speech_language(clause, self.language),
                        reference,
                        voice_transcript,
                        expression,
                    ):
                        if identifier != self.turn_id:
                            return
                        if first:
                            self.timings.since(
                                "fish_first_audio", synthesis_started, clause=clause_index
                            )
                            self.timings.since("clause_to_audio", queued_at, clause=clause_index)
                            first = False
                        generated_samples += len(pcm) // 2
                        for offset in range(0, len(pcm), 3840):
                            wait_started = time.monotonic()
                            await audio.put((pcm[offset : offset + 3840], None))
                            blocked_seconds += time.monotonic() - wait_started
                    self.timings.record(
                        "speech_generation",
                        time.monotonic() - synthesis_started - blocked_seconds,
                        audio_seconds=generated_samples / 24000,
                    )
                    clause_index += 1
                    await audio.put((b"", clause))
                await audio.put(None)

            async def play():
                nonlocal first_audio
                sequence = 0
                await audio.begin_clause()
                while (item := await audio.get()) is not None:
                    chunk, boundary = item
                    if boundary is not None:
                        self.boundaries.append((self.sent_samples, boundary))
                        self.apply_ack()
                        await audio.begin_clause()
                        continue
                    count = len(chunk) // 2
                    while self.sent_samples + count - self.acked > self.config.audio_window_samples:
                        self.ack_event.clear()
                        await asyncio.wait_for(self.ack_event.wait(), 10)
                    if identifier != self.turn_id:
                        return
                    packet = audio_packet(identifier, sequence, self.sent_samples, chunk)
                    self.sent_samples += count
                    await self.send_audio(packet)
                    sequence += 1
                    if first_audio is None:
                        first_audio = time.monotonic() - start
                        await self.event("state", state="speaking")

            async def queue_clause(clause):
                if clause is not None and not hasattr(queue_clause, "started"):
                    queue_clause.started = True
                    self.timings.since("first_speakable_clause", start)
                pending = asyncio.create_task(
                    clauses.put(
                        None if clause is None else (clause, parser.expression, time.monotonic())
                    )
                )
                try:
                    done, _ = await asyncio.wait(
                        {pending, speaker}, timeout=30, return_when=asyncio.FIRST_COMPLETED
                    )
                    if speaker in done:
                        await speaker
                    if pending not in done:
                        raise TimeoutError("Speech pipeline stalled")
                finally:
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)

            async def pipeline():
                async with asyncio.TaskGroup() as group:
                    group.create_task(synthesize())
                    group.create_task(play())

            if speak:
                self.speech.streaming = self.settings.fish_streaming
                speaker = asyncio.create_task(pipeline())
            await self.event("state", state="processing")
            from .text import ticking_stream

            first_text = True
            async for delta in ticking_stream(self.dialogue.generate(messages, self.profile)):
                if delta and first_text:
                    self.timings.since("llm_first_text", start)
                    first_text = False
                ready = await emit_text(delta)
                for clause in ready:
                    if speak:
                        await queue_clause(clause)
            ready = await emit_text("", final=True)
            if not self.generated.strip():
                raise SetupRequired("The dialogue model returned an empty reply. Please try again.")
            if speak:
                for clause in ready:
                    await queue_clause(clause)
                await queue_clause(None)
                await speaker
                while self.acked < self.sent_samples:
                    self.ack_event.clear()
                    await asyncio.wait_for(self.ack_event.wait(), 10)
            else:
                self.delivered = self.generated
            self.store.update_message(identifier, self.generated, self.delivered, "complete")
            completed = True
            await self.event(
                "done", first_audio_seconds=first_audio, elapsed_seconds=time.monotonic() - start
            )
            await self.event("state", state="listening" if self.listening else "idle")
        except asyncio.CancelledError:
            self.store.update_message(identifier, self.generated, self.delivered, "interrupted")
            raise
        except Exception as error:  # noqa: BLE001 — return a safe error across the client boundary
            self.store.update_message(identifier, self.generated, self.delivered, "interrupted")
            message = (
                str(error)
                if isinstance(error, SetupRequired)
                else "The reply stopped. Check local runtime diagnostics and try again."
            )
            await self.event("cancel")
            await self.event("error", message=message)
        finally:
            for stage in (producer, speaker):
                if stage and not stage.done():
                    stage.cancel()
                if stage:
                    await asyncio.gather(stage, return_exceptions=True)
            if speaker and not completed:
                await self.speech.cancel()
            self.last_activity = time.monotonic()

    def apply_ack(self):
        changed = False
        while self.boundaries and self.boundaries[0][0] <= self.acked:
            _, clause = self.boundaries.popleft()
            self.delivered += (" " if self.delivered else "") + clause
            changed = True
        if changed:
            self.store.update_message(self.turn_id, self.generated, self.delivered, "generating")

    def acknowledge(self, turn, samples):
        if turn != self.turn_id or samples < self.acked or samples > self.sent_samples:
            return
        self.acked = samples
        self.apply_ack()
        self.ack_event.set()

    async def stop(self, status="interrupted"):
        if self.task and not self.task.done():
            # Playback cancellation is sent before any potentially slow worker cleanup.
            await self.event("cancel")
            self.dialogue.invalidate_cache()
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            await self.speech.cancel()
        self.task = None

    async def set_profile(self, profile):
        await self.stop()
        await self.dialogue.unload()
        self.profile = profile
        self.store.set_setting("profile", profile)
        await self.event("profile", profile=profile)

    async def pressure(self, level):
        if level == "critical":
            await self.cancel_preparation()
            await self.mute()
            await self.stop()
            await self.dialogue.unload()
            await self.speech.close()
            await self.recognizer.close()
            await self.final_recognizer.close()
            await self.event("notice", message="Models unloaded to relieve memory pressure.")
        elif level == "warning":
            if hasattr(self.speech, "trim"):
                await self.speech.trim()
            self.warning_since = self.warning_since or time.monotonic()
        else:
            self.warning_since = None

    async def maintain(self):
        try:
            while True:
                await asyncio.sleep(5)
                busy = self.listening or any(
                    t and not t.done() for t in (self.task, self.final_task, self.prepare_task)
                )
                if self.warning_since and time.monotonic() - self.warning_since > 10 and not busy:
                    await self.dialogue.unload()
                    await self.speech.close()
                    await self.final_recognizer.close()
                    self.failed_recording = None
                    self.warning_since = None
                    await self.event(
                        "notice",
                        message="Idle models unloaded for memory pressure; your quality preferences are preserved.",
                    )
                if not busy and time.monotonic() - self.last_activity > self.config.idle_seconds:
                    self.failed_recording = None
                    await self.dialogue.unload()
                    await self.speech.close()
                    await self.recognizer.close()
                    await self.final_recognizer.close()
                    self.last_activity = time.monotonic()
                    await self.event("state", state="standby")
                self.last_memory = memory_snapshot()
                rss = self.last_memory["whole_app_rss_bytes"]
                if rss > PROFILES[self.profile].budget_gb * 1_000_000_000:
                    self.warning_since = self.warning_since or time.monotonic()
                await self.event(
                    "metrics",
                    **self.last_memory,
                    budget_gb=PROFILES[self.profile].budget_gb,
                    budget_is_target=True,
                    queued_audio_ms=(self.sent_samples - self.acked) / 24,
                )
        except asyncio.CancelledError:
            pass

    async def close(self):
        self.closed = True
        self.housekeeping.cancel()
        await asyncio.gather(self.housekeeping, return_exceptions=True)
        await self.cancel_preparation()
        await self.stop()
        await self.mute()
        await self.final_recognizer.close()
        await self.recognizer.close()
        await self.dialogue.close()
        await self.speech.close()
