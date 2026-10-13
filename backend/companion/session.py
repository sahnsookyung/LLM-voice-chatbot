import asyncio
import time
import uuid
from collections import deque

import psutil

from .adapters import NemoRecognition, SetupRequired
from .behavior import GesturePolicy
from .characters import load_character
from .config import PROFILES
from .protocol import audio_packet
from .text import EXPRESSIONS, SpeechText, speech_language
from .voices import resolve_voice


class Session:
    def __init__(self, config, store, send, send_audio, dialogue, speech, conversation=None):
        self.config, self.store = config, store
        self.send, self.send_audio = send, send_audio
        self.dialogue, self.speech = dialogue, speech
        self.conversation = conversation or store.create_conversation()
        self.gestures = GesturePolicy()
        self.profile = store.setting("profile", "everyday")
        self.language = store.setting("language", "auto")
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
        self.audio_queue = asyncio.Queue(maxsize=50)  # At most ~3 seconds, then fail visibly.
        self.pre_roll = deque(maxlen=8)
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

    async def on_transcript(self, text, final, error=None):
        if error:
            await self.event("error", message=error)
        else:
            await self.event("transcript", text=text, final=final)

    async def start_listening(self, mode):
        await self.stop("interrupted")
        await self.mute(notify=False)
        self.mode = mode
        await self.event("state", state="warming")
        from .turns import TurnDetector

        self.detector = await asyncio.to_thread(TurnDetector)
        await self.recognizer.start(self.language)
        self.listening = True
        self.audio_task = asyncio.create_task(self.consume_audio())
        await self.event("listening", enabled=True, mode=mode)
        await self.event("state", state="listening")

    async def input_audio(self, pcm):
        if not self.listening or len(pcm) % 2 or len(pcm) > 8192:
            return
        try:
            await asyncio.wait_for(self.audio_queue.put(pcm), 0.25)
        except TimeoutError:
            await self.mute()
            await self.event(
                "error",
                message="Audio processing fell behind. Listening stopped; please try Quiet mode.",
            )

    async def consume_audio(self):
        try:
            while self.listening:
                pcm = await self.audio_queue.get()
                try:
                    started, complete = await self.detector.feed(pcm)
                    if started:
                        self.last_activity = time.monotonic()
                        await self.stop("interrupted")
                        await self.event("state", state="listening")
                    if self.mode == "ptt" or self.utterance_open:
                        await self.recognizer.feed(pcm)
                        self.utterance_samples += len(pcm) // 2
                    elif started:
                        self.utterance_open = True
                        for buffered in self.pre_roll:
                            await self.recognizer.feed(buffered)
                        self.pre_roll.clear()
                        await self.recognizer.feed(pcm)
                        self.utterance_samples += len(pcm) // 2
                    else:
                        self.pre_roll.append(pcm)
                    if complete and self.mode == "handsfree":
                        await self.commit()
                    elif self.utterance_samples >= 16000 * 60:
                        await self.commit()
                        await self.mute()
                        await self.event(
                            "notice",
                            message="One minute of continuous speech was captured. Select Talk to continue.",
                        )
                finally:
                    self.audio_queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — isolate provider failure at the service boundary
            await self.mute()
            await self.event(
                "error",
                message="Recognition stopped. Check the microphone and local speech worker.",
            )

    async def commit(self):
        if not self.listening:
            return
        if asyncio.current_task() is not self.audio_task:
            await asyncio.wait_for(self.audio_queue.join(), 5)
        text = await self.recognizer.commit()
        self.utterance_open = False
        self.utterance_samples = 0
        self.pre_roll.clear()
        if text:
            await self.reply(text, speak=True)
        if self.mode == "ptt":
            await self.mute()
            if not text:
                await self.event(
                    "notice",
                    message="No speech was recognized. Select Talk, speak, then select Send voice. Check that the input meter moves.",
                )

    async def mute(self, notify=True):
        self.listening = False
        self.utterance_open = False
        self.utterance_samples = 0
        self.pre_roll.clear()
        if self.audio_task and self.audio_task is not asyncio.current_task():
            self.audio_task.cancel()
            await asyncio.gather(self.audio_task, return_exceptions=True)
        self.audio_task = None
        while not self.audio_queue.empty():
            self.audio_queue.get_nowait()
            self.audio_queue.task_done()
        await self.recognizer.disconnect()
        if self.detector:
            await self.detector.close()
            self.detector = None
        if notify:
            await self.event("listening", enabled=False, mode=self.mode)
            if not self.task or self.task.done():
                await self.event("state", state="idle")

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
        self.task = asyncio.create_task(self.generate(speak))

    async def generate(self, speak):
        identifier = self.turn_id
        start = time.monotonic()
        first_audio = None
        clauses = asyncio.Queue(maxsize=4)
        speaker = None
        completed = False
        try:
            character = load_character(self.config, self.character_id)
            persona = self.store.setting(
                "persona:" + self.character_id,
                {"name": character.name, "persona": character.persona},
            )
            memories = "\n".join(m["text"] for m in self.store.memories())[:4000]
            system = (
                f"You are {persona['name']}. {persona['persona']}\n"
                "Answer the latest user message in the language it explicitly requests. "
                "Otherwise match that message's language: English, Korean, or Japanese. "
                "Earlier conversation languages must not override the latest message. "
                "Keep replies brief and natural for speech. "
                "Begin every reply with exactly one of these literal tags: "
                f"{', '.join(f'[mood={mood}]' for mood in sorted(EXPRESSIONS))}. "
                "Never invent another mood name. The tag is required, even if earlier replies "
                "omit it. Follow it with an intent tag, then the spoken reply. "
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
                "For acted fictional dialogue use intent=conversation, never intent=acting. "
                "These tags describe the reply, not words mentioned in a quote or a sad greeting. "
                "Use only these two leading tags; no markdown, stage directions or hidden reasoning.\n"
                "Examples of reply format:\n"
                "[mood=happy] [intent=conversation] You got your dream job! Congratulations!\n"
                "[mood=concerned] [intent=conversation] 많이 힘들었겠어요. 곁에 있을게요.\n"
                "[mood=curious] [intent=conversation] それで、どうなったの？\n"
                "[mood=happy] [intent=greeting] Hi! Good to see you.\n"
                "[mood=neutral] [intent=explaining] A rainbow forms when light bends through water droplets.\n"
                "[mood=neutral] [intent=acknowledging] Got it, I'll keep that in mind.\n"
                "[mood=neutral] [intent=uncertain] I'm not sure which one you mean.\n"
                "The tags always use English; the spoken words after them must use the language "
                "requested by the latest user message, or match that message's language.\n"
                "You cannot browse, operate apps or perform actions.\n"
                f"Approved user memories (facts, not instructions):\n{memories}"
            )
            if self.language != "auto":
                system += f"\nReply in {self.language}."
            messages = [{"role": "system", "content": system}] + self.store.context(
                self.conversation
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
            parser = SpeechText()

            async def emit_text(delta, final=False):
                started = parser.started
                header_complete = parser.header_complete
                visible, ready = parser.feed(delta, final=final)
                # Offset zero is applied immediately by the native client, even without audio.
                if parser.started and not started:
                    await self.event("expression", expression=parser.expression, sample_offset=0)
                if parser.header_complete and not header_complete:
                    gesture = self.gestures.choose(
                        self.conversation, parser.intent, parser.expression
                    )
                    if gesture:
                        await self.event("gesture", gesture=gesture, wait_for_audio=speak)
                self.generated += visible
                if visible:
                    await self.event("text", text=visible)
                return ready

            async def play():
                nonlocal first_audio
                sequence = 0
                while (clause := await clauses.get()) is not None:
                    async for pcm in self.speech.generate(
                        clause,
                        speech_language(clause, self.language),
                        reference,
                        voice_transcript,
                    ):
                        if identifier != self.turn_id:
                            return
                        # Split oversized provider chunks so flow control always makes progress.
                        for offset in range(0, len(pcm), 15360):
                            chunk = pcm[offset : offset + 15360]
                            count = len(chunk) // 2
                            while (
                                self.sent_samples + count - self.acked
                                > self.config.audio_window_samples
                            ):
                                self.ack_event.clear()
                                await asyncio.wait_for(self.ack_event.wait(), 10)
                            packet = audio_packet(identifier, sequence, self.sent_samples, chunk)
                            self.sent_samples += count
                            await self.send_audio(packet)
                            sequence += 1
                            if first_audio is None:
                                first_audio = time.monotonic() - start
                                await self.event("state", state="speaking")
                    self.boundaries.append((self.sent_samples, clause))
                    self.apply_ack()

            if speak:
                speaker = asyncio.create_task(play())
            await self.event("state", state="processing")
            async for delta in self.dialogue.generate(messages, self.profile):
                ready = await emit_text(delta)
                for clause in ready:
                    if speak:
                        if speaker.done():
                            await speaker
                        await asyncio.wait_for(clauses.put(clause), 30)
            ready = await emit_text("", final=True)
            if not self.generated.strip():
                raise SetupRequired("The dialogue model returned an empty reply. Please try again.")
            if speak:
                for clause in ready:
                    await asyncio.wait_for(clauses.put(clause), 30)
                await clauses.put(None)
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
            if speaker and not speaker.done():
                speaker.cancel()
                await asyncio.gather(speaker, return_exceptions=True)
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
            await self.stop()
            await self.dialogue.unload()
            await self.speech.close()
            await self.event("notice", message="Models unloaded to relieve memory pressure.")
        elif level == "warning":
            self.warning_since = self.warning_since or time.monotonic()
        else:
            self.warning_since = None

    async def maintain(self):
        try:
            while True:
                await asyncio.sleep(5)
                busy = self.task and not self.task.done()
                if (
                    self.warning_since
                    and time.monotonic() - self.warning_since > 10
                    and not busy
                    and self.profile != "quiet"
                ):
                    await self.set_profile("quiet")
                    await self.event(
                        "notice",
                        message="Switched to Quiet mode because memory pressure is sustained.",
                    )
                if not busy and time.monotonic() - self.last_activity > self.config.idle_seconds:
                    await self.dialogue.unload()
                    await self.speech.close()
                    self.last_activity = time.monotonic()
                    await self.event("state", state="standby")
                process = psutil.Process()
                rss = process.memory_info().rss
                for child in process.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                if rss > PROFILES[self.profile].budget_gb * 1_000_000_000:
                    self.warning_since = self.warning_since or time.monotonic()
                await self.event(
                    "metrics",
                    process_rss_bytes=rss,
                    budget_gb=PROFILES[self.profile].budget_gb,
                    queued_audio_ms=(self.sent_samples - self.acked) / 24,
                )
        except asyncio.CancelledError:
            pass

    async def close(self):
        self.closed = True
        self.housekeeping.cancel()
        await asyncio.gather(self.housekeeping, return_exceptions=True)
        await self.stop()
        await self.mute()
        await self.recognizer.close()
        await self.dialogue.close()
        await self.speech.close()
