"""Media/cancellation contracts. These do not attest account voice entitlement."""

import asyncio
from fractions import Fraction

import numpy as np
import pytest
from aiortc import AudioStreamTrack, RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame
from companion.codex_rpc import CodexError
from companion.codex_rtc import CodexRTC, MicrophoneTrack


async def test_microphone_preserves_samples_across_arbitrary_packet_boundaries():
    track = MicrophoneTrack()
    track.set_muted(False)
    samples = np.arange(997, dtype="<i2")
    for start, end in ((0, 17), (17, 513), (513, 997)):
        track.feed(samples[start:end].tobytes())
    frames = [await track.recv() for _ in range(4)]
    actual = np.concatenate([f.to_ndarray().reshape(-1) for f in frames])
    np.testing.assert_array_equal(actual[:997], samples)
    assert not actual[997:].any()
    assert [f.pts for f in frames] == [0, 320, 640, 960]
    assert all(f.time_base == Fraction(1, 16000) for f in frames)
    track.stop()


async def test_mute_discards_pending_audio_but_keeps_silent_input_clock():
    track = MicrophoneTrack()
    track.set_muted(False)
    track.feed(b"\x05\x00" * 1000)
    await track.recv()
    track.set_muted(True)
    track.feed(b"\xff\x7f" * 320)
    frame = await track.recv()
    assert frame.pts == 320 and not frame.to_ndarray().any()
    track.set_muted(False)
    track.feed(b"\x07\x00" * 320)
    assert (await track.recv()).to_ndarray().min() == 7
    track.stop()
    with pytest.raises(MediaStreamError):
        await track.recv()


async def test_microphone_buffer_is_bounded_without_silent_drops():
    track = MicrophoneTrack()
    track.set_muted(False)
    track.feed(bytes(track.LIMIT))
    with pytest.raises(CodexError, match="overflow"):
        track.feed(b"\x00\x00")
    assert len(track.pending) == track.LIMIT
    track.stop()
    assert not track.pending


async def test_close_releases_pending_negotiation_and_all_media_tasks():
    errors = []

    async def failure(message):
        errors.append(message)

    async def audio(pcm):
        pass

    rtc = CodexRTC(audio, failure)
    waiting = asyncio.create_task(rtc.wait_ready())
    await rtc.close()
    with pytest.raises(CodexError, match="closed"):
        await waiting
    assert rtc.peer.connectionState == "closed" and not errors


async def test_receive_resamples_continuously_and_rejects_reordered_media():
    pcm, failures = [], []

    async def audio(data):
        pcm.append(np.frombuffer(data, dtype="<i2").copy())

    async def failed(message):
        failures.append(message)

    class Frames:
        def __init__(self):
            self.n = 0

        async def recv(self):
            self.n += 1
            frame = AudioFrame(format="s16", layout="mono", samples=960)
            frame.sample_rate = 48000
            frame.time_base = Fraction(1, 48000)
            frame.pts = (self.n - 1) * 960 if self.n < 11 else 0
            frame.planes[0].update(np.full(960, 5000, dtype="<i2").tobytes())
            return frame

    rtc = CodexRTC(audio, failed)
    await rtc._receive(Frames())
    await asyncio.sleep(0)
    assert len(pcm) == 10
    # Continuous resampling has one filter delay, not a fresh delay per packet.
    assert 4760 < sum(len(p) for p in pcm) <= 4800
    assert np.concatenate(pcm)[50:].min() > 4900
    assert len(failures) == 1
    await rtc.close()


@pytest.mark.skipif(
    __import__("os").environ.get("ARI_RTC_LOOPBACK") != "1",
    reason="Opt-in local UDP loopback, no OpenAI or microphone",
)
async def test_real_webrtc_bidirectional_audio_loopback():
    """Actual Opus/SRTP/data channel and resampling, with no account/device access."""
    outgoing, incoming, errors = [], [], []

    async def audio(pcm):
        outgoing.append(np.frombuffer(pcm, dtype="<i2").copy())

    async def failed(message):
        errors.append(message)

    rtc = CodexRTC(audio, failed)
    remote = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    readers = []

    class Tone(AudioStreamTrack):
        async def recv(self):
            frame = await super().recv()
            t = (np.arange(frame.samples) + frame.pts) / frame.sample_rate
            frame.planes[0].update((np.sin(t * 2 * np.pi * 440) * 8000).astype("<i2").tobytes())
            return frame

    remote.addTrack(Tone())

    @remote.on("track")
    def track_received(track):
        async def read():
            while True:
                frame = await track.recv()
                incoming.append(frame.to_ndarray().copy())

        readers.append(asyncio.create_task(read()))

    try:
        await remote.setRemoteDescription(RTCSessionDescription(await rtc.offer(), "offer"))
        await remote.setLocalDescription(await remote.createAnswer())
        await rtc.answer(remote.localDescription.sdp)
        await rtc.wait_ready()
        rtc.set_muted(False)
        for i in range(30):
            t = (np.arange(320) + i * 320) / 16000
            rtc.feed((np.sin(t * 2 * np.pi * 330) * 8000).astype("<i2").tobytes())
            await asyncio.sleep(0.02)
        rtc.set_muted(True)
        await asyncio.sleep(0.4)
        assert len(outgoing) > 15 and len(incoming) > 15
        assert max(abs(p.astype(float)).max() for p in outgoing) > 2000
        assert max(abs(p.astype(float)).max() for p in incoming) > 2000
        assert not errors
    finally:
        await rtc.close()
        for reader in readers:
            reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
        await remote.close()
    assert rtc.peer.connectionState == remote.connectionState == "closed"
