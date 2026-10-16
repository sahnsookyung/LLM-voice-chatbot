"""Host-authenticated WebRTC media. No account tokens or device access here.

The shared app-server creates the call and returns its SDP answer. Native Ari
owns microphone authorization/capture and actual playback; this adapter only
converts their PCM to/from the continuous encrypted media track.
"""

import asyncio
import json
import time
from fractions import Fraction

from aiortc import AudioStreamTrack, RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.exceptions import InvalidStateError, OperationError
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame, AudioResampler

from .codex_rpc import CodexError


class MicrophoneTrack(AudioStreamTrack):
    RATE = 16000
    SAMPLES = 320
    LIMIT = RATE * 2  # One second of PCM16, in addition to the bounded native sender.

    def __init__(self):
        super().__init__()
        self.pending = bytearray()
        self.muted = True
        self.pts = 0
        self.deadline = None

    def set_muted(self, muted):
        self.muted = muted
        self.pending.clear()

    def feed(self, pcm):
        if self.muted or self.readyState != "live":
            return
        if not pcm or len(pcm) % 2 or len(self.pending) + len(pcm) > self.LIMIT:
            raise CodexError("Microphone WebRTC buffer overflow or invalid PCM; voice stopped.")
        self.pending.extend(pcm)

    async def recv(self):
        if self.readyState != "live":
            raise MediaStreamError
        now = time.monotonic()
        if self.deadline is None:
            self.deadline = now
        await asyncio.sleep(max(0, self.deadline - now))
        # Do not burst old frames after sleep/wake or an event-loop stall.
        self.deadline += self.SAMPLES / self.RATE
        if self.deadline < time.monotonic() - 0.1:
            self.deadline = time.monotonic() + self.SAMPLES / self.RATE
        if self.readyState != "live":
            raise MediaStreamError
        count = min(len(self.pending), self.SAMPLES * 2)
        pcm = bytes(self.pending[:count]) + bytes(self.SAMPLES * 2 - count)
        del self.pending[:count]
        frame = AudioFrame(format="s16", layout="mono", samples=self.SAMPLES)
        frame.planes[0].update(pcm)
        frame.sample_rate, frame.pts, frame.time_base = self.RATE, self.pts, Fraction(1, self.RATE)
        self.pts += self.SAMPLES
        # GPT-Live needs the input timeline to continue even while the human is muted.
        return frame

    def stop(self):
        self.set_muted(True)
        super().stop()


class CodexRTC:
    def __init__(self, audio, failed):
        self.audio, self.failed = audio, failed
        self.peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        self.microphone = MicrophoneTrack()
        self.peer.addTrack(self.microphone)
        self.channel = self.peer.createDataChannel("oai-events")
        self.ready = asyncio.Event()
        self.closed = False
        self.failure = None
        self.failure_task = None
        self.reader = None
        self.answer_received = False

        @self.channel.on("open")
        def opened():
            self.ready.set()

        @self.channel.on("close")
        def channel_closed():
            if not self.closed:
                self._fail("Codex voice connection closed. Start voice again to reconnect.")

        @self.channel.on("message")
        def message(data):
            # Transcripts/history come from the canonical host timeline only.
            # Never persist raw channel events or create a second task handoff.
            if not isinstance(data, str) or len(data) > 65536:
                self._fail("Invalid Codex voice control packet; voice stopped.")
                return
            try:
                value = json.loads(data)
                if not isinstance(value, dict):
                    raise TypeError
                if value.get("type") in {"error", "session.error"}:
                    self._fail("Codex reported a voice service error. Check the desktop and retry.")
            except (ValueError, TypeError):
                self._fail("Invalid Codex voice control packet; voice stopped.")

        @self.peer.on("connectionstatechange")
        def changed():
            if self.peer.connectionState in {"failed", "closed"} and not self.closed:
                self._fail("Codex voice connection failed. Check the connection and retry.")

        @self.peer.on("track")
        def track_added(track):
            if track.kind != "audio" or self.reader is not None:
                self._fail("Unexpected Codex media track; voice stopped.")
                return
            self.reader = asyncio.create_task(self._receive(track))

    def _fail(self, message):
        if self.closed or self.failure:
            return
        self.failure = message
        self.microphone.set_muted(True)
        self.ready.set()
        self.failure_task = asyncio.create_task(self.failed(message))

    async def offer(self):
        try:
            async with asyncio.timeout(10):
                await self.peer.setLocalDescription(await self.peer.createOffer())
        except (InvalidStateError, OperationError, ValueError, OSError, TimeoutError) as error:
            raise CodexError("Unable to prepare Codex voice media.") from error
        return self.peer.localDescription.sdp

    async def answer(self, sdp):
        if self.closed:
            return
        if self.answer_received or not isinstance(sdp, str) or len(sdp) > 65536:
            raise CodexError("Invalid or repeated Codex WebRTC answer; voice stopped.")
        self.answer_received = True
        try:
            async with asyncio.timeout(10):
                await self.peer.setRemoteDescription(RTCSessionDescription(sdp, "answer"))
        except (InvalidStateError, OperationError, ValueError, RuntimeError, TimeoutError) as error:
            raise CodexError("Codex WebRTC negotiation failed; voice stopped.") from error

    async def wait_ready(self):
        await asyncio.wait_for(self.ready.wait(), timeout=20)
        if self.closed or self.failure:
            raise CodexError(self.failure or "Codex voice was closed during startup.")

    def set_muted(self, muted):
        self.microphone.set_muted(muted)

    def feed(self, pcm):
        self.microphone.feed(pcm)

    async def _receive(self, track):
        converter = AudioResampler(format="s16", layout="mono", rate=24000)
        previous_end = None
        try:
            while not self.closed:
                # Silence is not a failed connection. ICE/DTLS state handles
                # disconnects; some peers send no RTP during quiet intervals.
                frame = await track.recv()
                if (
                    frame.sample_rate not in {16000, 24000, 48000}
                    or len(frame.layout.channels) not in {1, 2}
                    or not 0 < frame.samples <= frame.sample_rate
                ):
                    raise CodexError("Unsupported Codex WebRTC audio format; voice stopped.")
                if frame.pts is None or frame.time_base is None:
                    raise CodexError("Codex WebRTC audio is missing its media clock.")
                start = frame.pts * frame.time_base
                if previous_end is not None and start < previous_end:
                    raise CodexError("Codex WebRTC audio arrived out of order; voice stopped.")
                previous_end = start + Fraction(frame.samples, frame.sample_rate)
                for converted in converter.resample(frame):
                    # Await delivery with a bound: the RTP receiver must never grow
                    # indefinitely behind an unresponsive native websocket.
                    async with asyncio.timeout(1):
                        await self.audio(converted.to_ndarray().tobytes())
        except asyncio.CancelledError:
            raise
        except MediaStreamError:
            self._fail("Codex voice media ended. Start voice again to reconnect.")
        except (CodexError, ValueError, OSError, RuntimeError, TimeoutError):
            self._fail("Codex voice media delivery failed. Voice stopped; task work is unchanged.")

    async def close(self):
        if self.closed:
            return
        self.closed = True
        self.ready.set()
        self.microphone.stop()
        current = asyncio.current_task()
        tasks = [t for t in (self.reader, self.failure_task) if t and t is not current]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.peer.close()
