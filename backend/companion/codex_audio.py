"""Validated realtime PCM, continuous resampling, bounded jitter/playback queue."""

import asyncio
import base64
import time
import uuid
from collections import deque

import numpy as np

from .codex_rpc import CodexError
from .protocol import audio_packet
from .resampling import StreamingResampler


class CodexAudio:
    RATE = 24000
    LIMIT = RATE * 6

    def __init__(self, event, send_audio):
        self.event, self.send_audio = event, send_audio
        self.turn = None
        self.queue = deque()
        self.queued = self.sent = self.acked = self.sequence = 0
        self.threshold = 2880  # 120 ms; adapt only after an observed native underrun.
        self.condition = asyncio.Event()
        self.worker = None
        self.resampler = None
        self.source_format = None
        self.item_id = None
        self.stale = deque(maxlen=256)
        self.finished = False
        self.started = False
        self.last_input = 0.0

    async def feed(self, chunk):
        encoded = chunk.get("data", "")
        if not isinstance(encoded, str) or len(encoded) > 1024 * 1024:
            raise CodexError("Invalid Codex audio packet size.")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except ValueError as error:
            raise CodexError("Invalid Codex audio encoding.") from error
        await self.feed_pcm(
            raw,
            chunk.get("sampleRate"),
            chunk.get("numChannels"),
            chunk.get("itemId"),
            chunk.get("samplesPerChannel"),
        )

    async def feed_pcm(self, raw, rate, channels, item, declared_count=None):
        if rate not in {16000, 22050, 24000, 44100, 48000} or channels not in {1, 2}:
            raise CodexError("Unsupported Codex audio format; voice stopped.")
        if not isinstance(item, str) or not item or len(item) > 256:
            raise CodexError(
                "Codex audio lacks an item identity; safe interruption is unavailable."
            )
        if item in self.stale:
            return
        if not raw or len(raw) % (2 * channels) or len(raw) // (2 * channels) > rate * 6:
            raise CodexError("Invalid Codex audio sample count.")
        count = len(raw) // (2 * channels)
        if declared_count not in (None, count):
            raise CodexError("Codex audio sample metadata did not match its payload.")
        if self.item_id != item:
            self._flush_converter()
            if self.item_id:
                self.stale.append(self.item_id)
            self.item_id = item
            self.source_format = (rate, channels)
            self.resampler = StreamingResampler(rate, self.RATE)
        elif self.source_format != (rate, channels):
            raise CodexError("Codex changed audio format inside an item.")
        elif self.resampler is None:
            raise CodexError("Codex sent audio after this item completed.")
        if self.turn is None:
            self.turn = str(uuid.uuid4())
            await self.event("codex_audio_begin", turn_id=self.turn)
            self.worker = asyncio.create_task(self._deliver())
        samples = (
            np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(-1, channels).mean(axis=1)
            / 32768
        )
        self._enqueue(self.resampler.feed(samples))
        self.finished = False
        self.last_input = time.monotonic()
        self.condition.set()

    def _enqueue(self, samples):
        pcm = (np.clip(samples, -1, 32767 / 32768) * 32768).round().astype("<i2").tobytes()
        count = len(pcm) // 2
        if self.queued + self.sent - self.acked + count > self.LIMIT:
            raise CodexError("Codex audio exceeded six seconds of queued playback. Voice stopped.")
        for start in range(0, len(pcm), 3840):
            self.queue.append(pcm[start : start + 3840])
        self.queued += count

    def _flush_converter(self):
        if self.resampler and not self.resampler.closed:
            self._enqueue(self.resampler.feed([], final=True))
        self.resampler = None

    def finish(self):
        self._flush_converter()
        self.finished = True
        self.condition.set()

    def acknowledge(self, turn, samples):
        if turn == self.turn and self.acked <= samples <= self.sent:
            self.acked = samples
            self.condition.set()

    def underrun(self):
        self.threshold = min(7200, self.threshold + 960)  # At most 300 ms.
        self.started = False

    async def _deliver(self):
        try:
            while True:
                self.condition.clear()
                if not self.started and (self.queued >= self.threshold or self.finished):
                    self.started = True
                while self.started and self.queue and self.sent - self.acked < self.RATE:
                    packet = self.queue[0]
                    count = len(packet) // 2
                    if self.sent - self.acked + count > self.RATE:
                        break
                    self.queue.popleft()
                    self.queued -= count
                    await self.send_audio(audio_packet(self.turn, self.sequence, self.sent, packet))
                    self.sent += count
                    self.sequence += 1
                # Small final fragments must not be stranded below startup jitter.
                try:
                    await asyncio.wait_for(self.condition.wait(), timeout=0.3)
                except TimeoutError:
                    if self.queue and time.monotonic() - self.last_input >= 0.3:
                        self.started = True
        except asyncio.CancelledError:
            raise
        except (OSError, RuntimeError, ValueError):
            await self.event(
                "codex_audio_failed", message="Playback delivery failed. End voice and reconnect."
            )

    async def clear(self):
        if self.item_id:
            self.stale.append(self.item_id)
        if self.worker and self.worker is not asyncio.current_task():
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)
        if self.turn:
            await self.event("cancel", turn_id=self.turn)
        self.turn = self.worker = self.resampler = self.item_id = self.source_format = None
        self.queue.clear()
        self.queued = self.sent = self.acked = self.sequence = 0
        self.started = self.finished = False
