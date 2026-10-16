"""Capture lifecycle separated from dialogue; no microphone ownership in the backend."""

import asyncio
import time
import uuid

from .final_asr import FinalRecognition
from .protocol import decode_microphone


class SampleBuffer:
    def __init__(self, samples):
        self.limit = samples * 2
        self.data = bytearray()

    def append(self, pcm):
        self.data.extend(pcm)
        if len(self.data) > self.limit:
            del self.data[: len(self.data) - self.limit]

    def take(self):
        result = bytes(self.data)
        self.data.clear()
        return result

    def clear(self):
        self.data.clear()


class CaptureSession:
    def init_capture(self):
        self.capture_id = None
        self.microphone_version = 1
        self.capture_sequence = self.capture_offset = 0
        self.audio_queue = asyncio.Queue(maxsize=1000)
        self.queued_capture_bytes = 0
        self.pre_roll = SampleBuffer(6400)
        self.recording = bytearray()
        self.draft = ""
        self.failed_recording = None
        self.start_task = self.final_task = self.prepare_task = None
        self.recognition_ready = asyncio.Event()
        self.accepting_audio = True
        self.final_recognizer = FinalRecognition(self.config)

    async def on_transcript(self, text, final, error=None):
        if error:
            await self.event("notice", message=error)
        else:
            self.draft = text
            await self.event(
                "transcript", capture_id=self.capture_id, text=text, final=False, status="draft"
            )

    async def prepare(self, force=False):
        if self.config.demo or (not self.settings.warm_up and not force):
            return
        if self.prepare_task and not self.prepare_task.done():
            return

        async def run():
            try:
                for name, operation in (
                    (
                        "draft_recognition",
                        lambda: self.recognizer.prepare(self.settings.context_ms),
                    ),
                    ("final_recognition", self.final_recognizer.prepare),
                    ("speech", self.speech.prepare),
                    ("dialogue", lambda: self.dialogue.prepare(self.profile)),
                ):
                    if name == "final_recognition" and self.settings.recognition_mode == "fast":
                        continue
                    await self.event("preparation", worker=name, ready=False)
                    start = time.monotonic()
                    await operation()
                    self.timings.since("prepare_" + name, start)
                    await self.event("preparation", worker=name, ready=True)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — contain provider failures without private text
                await self.event(
                    "notice",
                    message="A speech worker is not ready. Check setup or retry preparation.",
                )

        self.prepare_task = asyncio.create_task(run())

    async def cancel_preparation(self):
        if self.prepare_task and not self.prepare_task.done():
            self.prepare_task.cancel()
            await asyncio.gather(self.prepare_task, return_exceptions=True)
            await self.event("preparation", worker="", ready=False, cancelled=True)
        self.prepare_task = None

    async def start_listening(self, mode, capture_id=None, microphone_version=1):
        await self.stop("interrupted")
        await self.mute(notify=False)
        await self.cancel_preparation()
        self.capture_id = str(capture_id or uuid.uuid4())
        self.microphone_version = microphone_version
        self.capture_sequence = self.capture_offset = 0
        self.mode, self.listening, self.accepting_audio = mode, True, True
        self.recognition_ready.clear()
        self.draft = ""
        await self.event("listening", enabled=True, mode=mode, capture_id=self.capture_id)
        await self.event("preparation", worker="draft_recognition", ready=False)

        async def start():
            began = time.monotonic()
            try:
                if mode == "handsfree":
                    from .turns import TurnDetector

                    self.detector = await asyncio.to_thread(
                        TurnDetector, self.settings.sensitivity, self.settings.pause_tolerance
                    )
                await self.recognizer.start(
                    self.settings.input_language, self.settings.vocabulary, self.settings.context_ms
                )
                self.recognition_ready.set()
                self.timings.since("recognition_startup", began)
                await self.event(
                    "preparation",
                    worker="draft_recognition",
                    ready=True,
                    capture_id=self.capture_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — contain provider failures without private text
                await self.mute()
                await self.event(
                    "error",
                    message="Recognition could not start. Check the local models and retry.",
                )

        self.start_task = asyncio.create_task(start())
        self.audio_task = asyncio.create_task(self.consume_audio(wait_ready=True))

    async def input_audio(self, pcm):
        if not self.listening or not self.accepting_audio:
            return
        if self.microphone_version == 2:
            try:
                identifier, sequence, offset, pcm = decode_microphone(pcm)
                if identifier != self.capture_id:
                    return
                if sequence != self.capture_sequence or offset != self.capture_offset:
                    raise ValueError(
                        "Microphone audio has a gap or duplicate. Please record again."
                    )
                self.capture_sequence += 1
                self.capture_offset += len(pcm) // 2
            except ValueError:
                await self.mute()
                await self.event(
                    "error",
                    message="Microphone packets were lost or reordered. Please record again.",
                )
                return
        if not pcm or len(pcm) % 2 or len(pcm) > 8192:
            return
        if self.queued_capture_bytes + len(pcm) > 320000 or self.audio_queue.full():
            await self.mute()
            await self.event(
                "error",
                message="Recognition fell more than ten seconds behind. Retry after preparation finishes.",
            )
            return
        self.queued_capture_bytes += len(pcm)
        self.audio_queue.put_nowait(pcm)

    async def consume_audio(self, wait_ready=False):
        try:
            if wait_ready:
                await self.recognition_ready.wait()
            while self.listening:
                pcm = await self.audio_queue.get()
                self.queued_capture_bytes -= len(pcm)
                try:
                    started, complete = (False, False)
                    if self.mode == "handsfree":
                        started, complete = await self.detector.feed(pcm)
                    if started and not self.utterance_open:
                        await self.cancel_finalization()
                        await self.stop("interrupted")
                        self.utterance_open = True
                        prefix = self.pre_roll.take()
                        if prefix:
                            await self.feed_recognition(prefix)
                    if self.mode == "ptt" or self.utterance_open:
                        await self.feed_recognition(pcm)
                    else:
                        self.pre_roll.append(pcm)
                    if complete and self.mode == "handsfree":
                        await self.commit()
                    elif self.utterance_samples >= 16000 * 60:
                        self.accepting_audio = False
                        await self.event(
                            "listening", enabled=False, mode=self.mode, capture_id=self.capture_id
                        )
                        await self.commit()
                        if self.final_task and self.final_task is not asyncio.current_task():
                            await self.final_task
                        await self.mute()
                        await self.event(
                            "notice", message="The 60-second recording limit was reached."
                        )
                finally:
                    self.audio_queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — contain provider failures without private text
            await self.mute()
            await self.event(
                "error",
                message="Recognition stopped. Check the microphone and local speech worker.",
            )

    async def feed_recognition(self, pcm):
        remaining = max(0, 60 * 32000 - len(self.recording))
        pcm = pcm[:remaining]
        if pcm:
            self.recording.extend(pcm)
            self.utterance_samples += len(pcm) // 2
            await self.recognizer.feed(pcm)

    async def request_commit(self, capture_id=None):
        if capture_id and str(capture_id) != self.capture_id:
            return
        if self.final_task and not self.final_task.done():
            return
        self.accepting_audio = False
        self.final_task = asyncio.create_task(self.commit())

    async def commit(self):
        if not self.listening:
            return
        identifier = self.capture_id
        if asyncio.current_task() is not self.audio_task:
            await self.audio_queue.join()
        started = time.monotonic()
        await self.event(
            "transcript", capture_id=identifier, status="finalizing", text=self.draft, final=False
        )
        pcm = bytes(self.recording)
        self.recording.clear()
        self.utterance_open = False
        self.utterance_samples = 0
        self.pre_roll.clear()

        async def finalize():
            await self.finish_final(pcm, identifier, started)

        if self.mode == "handsfree" and asyncio.current_task() is self.audio_task:
            # Flush RNNT before accepting the next utterance. The stronger pass
            # then runs independently so a fresh speech onset can cancel it.
            try:
                async with asyncio.timeout(min(30, max(5, len(pcm) / 64000))):
                    self.draft = await self.recognizer.commit()
                self.final_task = asyncio.create_task(finalize())
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — contain provider failures without private text
                self.final_task = asyncio.create_task(
                    self.finish_final(pcm, identifier, started, failed=True)
                )
        else:
            await self.finish_final(pcm, identifier, started, commit_draft=True)

    async def finish_final(self, pcm, identifier, started, commit_draft=False, failed=False):
        try:
            if failed:
                raise RuntimeError("Draft finalization failed")
            timeout = min(30, max(5, len(pcm) / 32000 / 2))
            async with asyncio.timeout(max(0.01, timeout - (time.monotonic() - started))):
                if commit_draft:
                    self.draft = await self.recognizer.commit()
                if self.settings.recognition_mode == "accurate" and not self.config.demo and pcm:
                    text = await self.final_recognizer.transcribe(pcm, self.settings)
                else:
                    text = self.draft
            if identifier != self.capture_id:
                return
            self.timings.since(
                "recognition_finalization",
                started,
                mode=self.settings.recognition_mode,
                audio_seconds=len(pcm) / 32000,
            )
            await self.event(
                "transcript", capture_id=identifier, status="final", text=text, final=True
            )
            if text:
                await self.reply(text, speak=True)
            else:
                await self.event(
                    "notice",
                    message="No speech was recognized. Check the input meter and record again.",
                )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — contain provider failures without private text
            if identifier == self.capture_id:
                # Bounded, in-memory retry buffer. Never enters conversation storage.
                self.failed_recording = (identifier, pcm, self.draft)
                await self.event(
                    "transcript",
                    capture_id=identifier,
                    status="failed",
                    text=self.draft,
                    final=False,
                )
                await self.event(
                    "notice",
                    message="Final recognition failed. Retry, edit the words, or use the draft.",
                )
        finally:
            pcm = None
            if self.mode == "ptt" and identifier == self.capture_id:
                await self.mute(preserve_failure=True)

    async def transcription_action(self, action, capture_id=None, text=""):
        if not self.failed_recording or str(capture_id) != self.failed_recording[0]:
            return
        identifier, pcm, draft = self.failed_recording
        if action in {"edit", "use_draft"}:
            self.failed_recording = None
            await self.event(
                "transcript",
                capture_id=identifier,
                status="final",
                text=text if action == "edit" else draft,
                final=True,
            )
            await self.reply(text if action == "edit" else draft)
            return
        if self.final_task and not self.final_task.done():
            return

        async def retry():
            try:
                await self.event(
                    "transcript",
                    capture_id=identifier,
                    status="finalizing",
                    text=draft,
                    final=False,
                )
                async with asyncio.timeout(min(30, max(5, len(pcm) / 64000))):
                    result = await self.final_recognizer.transcribe(pcm, self.settings)
                if not self.failed_recording or self.failed_recording[0] != identifier:
                    return
                self.failed_recording = None
                await self.event(
                    "transcript", capture_id=identifier, status="final", text=result, final=True
                )
                await self.reply(result)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — contain provider failures without private text
                await self.event(
                    "transcript", capture_id=identifier, status="failed", text=draft, final=False
                )

        self.final_task = asyncio.create_task(retry())

    async def cancel_finalization(self):
        if self.final_task and self.final_task is not asyncio.current_task():
            self.final_task.cancel()
            await asyncio.gather(self.final_task, return_exceptions=True)
            self.final_task = None
        self.failed_recording = None

    async def mute(self, notify=True, preserve_failure=False):
        self.listening = False
        self.accepting_audio = False
        self.utterance_open = False
        self.utterance_samples = 0
        self.recording.clear()
        self.pre_roll.clear()
        if not preserve_failure:
            await self.cancel_finalization()
        for task in (self.start_task, self.audio_task):
            if task and task is not asyncio.current_task():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.start_task = self.audio_task = None
        while not self.audio_queue.empty():
            self.audio_queue.get_nowait()
            self.audio_queue.task_done()
        self.queued_capture_bytes = 0
        await self.recognizer.disconnect()
        if self.detector:
            await self.detector.close()
            self.detector = None
        if notify:
            await self.event("listening", enabled=False, mode=self.mode, capture_id=self.capture_id)
            if not self.task or self.task.done():
                await self.event("state", state="idle")
