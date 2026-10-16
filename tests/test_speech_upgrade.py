import asyncio
import uuid

import httpx
import numpy as np
import pytest
from companion.adapters import DemoDialogue, DemoSpeech
from companion.app import create_app
from companion.capture import SampleBuffer
from companion.config import Config
from companion.protocol import Command, decode_microphone, microphone_packet
from companion.session import Session
from companion.speech_settings import SpeechSettings
from companion.store import Store
from companion.text import SpeechText, speech_language
from pydantic import ValidationError


def test_preferences_migrate_once_and_survive_restart(tmp_path):
    path = tmp_path / "history.sqlite"
    store = Store(path)
    store.set_setting("language", "korean")
    settings = SpeechSettings.load(store)
    assert settings.input_language == settings.reply_language == "korean"
    settings.input_language = "auto"
    settings.vocabulary = ["Ari", "Codex", "MLX"]
    store.set_setting("speech_settings", settings.model_dump())
    store.close()
    store = Store(path)
    assert SpeechSettings.load(store) == settings
    store.close()


@pytest.mark.parametrize(
    "change",
    [
        {"context_ms": 200},
        {"fish_precision": 4},
        {"pause_tolerance": 0},
        {"vocabulary": ["x" * 81]},
        {"input_language": "german"},
    ],
)
def test_invalid_settings_do_not_reach_workers(change):
    with pytest.raises(ValidationError):
        Command(type="speech_settings", speech_settings=change)


@pytest.mark.parametrize("size", [2, 128, 1024, 2048, 8192])
def test_preroll_is_400ms_independent_of_device_packet_size(size):
    source = np.arange(15000, dtype="<i2").tobytes()
    buffer = SampleBuffer(6400)
    for start in range(0, len(source), size):
        buffer.append(source[start : start + size])
    assert buffer.take() == source[-12800:]
    assert buffer.take() == b""


def test_microphone_framing_roundtrip():
    capture = str(uuid.uuid4())
    assert decode_microphone(microphone_packet(capture, 17, 42, bytes(200))) == (
        capture,
        17,
        42,
        bytes(200),
    )
    for packet in (b"", b"LVM2", microphone_packet(capture, 1, 0, b"x")):
        with pytest.raises(ValueError):
            decode_microphone(packet)


def test_mixed_speech_has_no_forced_monolingual_instruction():
    assert speech_language("Codex 설정을 바꿔 주세요", "korean") == "auto"
    assert speech_language("今日は API を使います", "japanese") == "auto"
    assert speech_language("안녕하세요", "auto") == "korean"


def test_clause_deadline_and_safe_boundaries(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("companion.text.time.monotonic", lambda: clock[0])
    parser = SpeechText()
    _, clauses = parser.feed("Dr. Kim checked 3.14 today but the result needs another check")
    assert clauses == []
    clock[0] += 0.61
    _, clauses = parser.feed("")
    assert clauses == ["Dr. Kim checked 3.14 today"]
    _, clauses = parser.feed(".", final=True)
    assert clauses == ["but the result needs another check."]


class Recognition:
    def __init__(self):
        self.chunks = []
        self.ready = asyncio.Event()
        self.ready.set()

    async def start(self, *args):
        await self.ready.wait()

    async def feed(self, pcm):
        self.chunks.append(pcm)

    async def commit(self):
        return "draft words"

    async def disconnect(self):
        pass

    async def close(self):
        pass


@pytest.fixture
async def capture_session(tmp_path):
    events = []

    async def send(event):
        if isinstance(event, dict):
            events.append(event)

    store = Store(tmp_path / "history.sqlite")
    session = Session(
        Config(data=tmp_path, token="x" * 64, demo=True),
        store,
        send,
        send,
        DemoDialogue(),
        DemoSpeech(),
    )
    session.recognizer = Recognition()
    yield session, events
    await session.close()
    store.close()


async def test_release_during_preparation_keeps_all_audio(capture_session):
    s, events = capture_session
    s.recognizer.ready.clear()
    identifier = str(uuid.uuid4())
    await s.start_listening("ptt", identifier, 2)
    pcm = np.arange(5000, dtype="<i2").tobytes()
    for seq, start in enumerate(range(0, len(pcm), 2048)):
        await s.input_audio(
            microphone_packet(identifier, seq, start // 2, pcm[start : start + 2048])
        )
    await s.request_commit(identifier)
    assert not s.final_task.done()
    assert not any(e["type"] == "turn" for e in events)
    s.recognizer.ready.set()
    await asyncio.wait_for(s.final_task, 1)
    assert b"".join(s.recognizer.chunks) == pcm
    assert any(e.get("status") == "final" for e in events)
    assert s.detector is None  # PTT never initializes VAD/Smart Turn.


async def test_stale_frames_and_duplicate_frames_cannot_reach_recognition(capture_session):
    s, events = capture_session
    identifier = str(uuid.uuid4())
    await s.start_listening("ptt", identifier, 2)
    await s.input_audio(microphone_packet(str(uuid.uuid4()), 0, 0, bytes(128)))
    packet = microphone_packet(identifier, 0, 0, bytes(128))
    await s.input_audio(packet)
    await s.audio_queue.join()
    await s.input_audio(packet)
    assert b"".join(s.recognizer.chunks) == bytes(128)
    assert not s.listening
    assert any(e["type"] == "error" for e in events)


async def test_startup_overflow_is_visible_and_cancellable(capture_session):
    s, events = capture_session
    s.recognizer.ready.clear()
    await s.start_listening("ptt")
    for _ in range(40):
        await s.input_audio(bytes(8192))
    assert not s.listening
    assert s.queued_capture_bytes == 0
    assert any(e["type"] == "error" and "ten seconds" in e["message"] for e in events)


async def test_accurate_failure_never_submits_draft_without_choice(capture_session):
    s, events = capture_session
    s.config.demo = False

    class Final:
        async def transcribe(self, *args):
            raise RuntimeError("worker failed")

        async def close(self):
            pass

    s.final_recognizer = Final()
    await s.start_listening("ptt")
    await s.input_audio(bytes(2048))
    await s.request_commit()
    await asyncio.wait_for(s.final_task, 1)
    assert not s.store.messages(s.conversation)
    assert any(e.get("status") == "failed" for e in events)
    assert s.failed_recording is not None
    await s.transcription_action("use_draft", s.capture_id)
    assert s.store.messages(s.conversation)[0]["generated"] == "draft words"
    assert s.failed_recording is None


async def test_new_capture_cancels_final_asr_and_late_result(capture_session):
    s, _events = capture_session
    s.config.demo = False
    entered, cancelled = asyncio.Event(), asyncio.Event()

    class Final:
        async def transcribe(self, *args):
            entered.set()
            try:
                await asyncio.sleep(60)
            finally:
                cancelled.set()

        async def close(self):
            pass

    s.final_recognizer = Final()
    await s.start_listening("ptt")
    old = s.capture_id
    await s.input_audio(bytes(200))
    await s.request_commit()
    await asyncio.wait_for(entered.wait(), 1)
    await s.start_listening("ptt")
    assert cancelled.is_set() and s.capture_id != old
    assert not s.store.messages(s.conversation)


async def test_synthesis_runs_ahead_but_stays_bounded(capture_session):
    s, events = capture_session
    produced = []

    class Dialogue(DemoDialogue):
        async def generate(self, *args):
            yield "[mood=neutral] " + "One sentence. " * 30

    class Speech(DemoSpeech):
        async def generate(self, text, *args):
            produced.append(text)
            for _ in range(25):
                yield bytes(3840)  # 2 seconds/clause

    s.dialogue, s.speech = Dialogue(), Speech()
    await s.reply("Hi")
    await asyncio.sleep(0.1)
    assert 2 <= len(produced) <= 4
    assert s.sent_samples <= 24000
    await s.stop()
    assert not s.delivered
    assert any(e["type"] == "cancel" for e in events)


async def test_slow_synthesis_cannot_start_a_short_clause_and_starve_mid_word(capture_session):
    s, _events = capture_session
    emitted = []
    finish = asyncio.Event()

    class Speech(DemoSpeech):
        async def generate(self, *args):
            yield bytes(3840)
            await finish.wait()
            yield bytes(3840)

    class Dialogue(DemoDialogue):
        async def generate(self, *args):
            yield "One complete clause."

    async def audio(packet):
        from companion.protocol import decode_audio

        turn, _, offset, pcm = decode_audio(packet)
        emitted.append(pcm)
        s.acknowledge(turn, offset + len(pcm) // 2)

    s.dialogue, s.speech, s.send_audio = Dialogue(), Speech(), audio
    await s.reply("Hello")
    await asyncio.sleep(0.05)
    assert not emitted
    finish.set()
    await asyncio.wait_for(s.task, 1)
    assert b"".join(emitted) == bytes(7680)


async def test_long_clause_reservoir_starts_before_completion_without_exceeding_budget():
    from companion.audio_reservoir import AudioReservoir

    queue = AudioReservoir()

    async def produce():
        for _ in range(200):
            await queue.put((bytes(3840), None))
        await queue.put((b"", "complete"))
        await queue.put(None)

    task = asyncio.create_task(produce())
    await asyncio.wait_for(queue.begin_clause(), 1)
    assert not task.done()
    assert queue.bytes <= queue.limit
    samples = 0
    while (item := await queue.get()) is not None:
        samples += len(item[0])
        assert queue.bytes <= queue.limit
    await task
    assert samples == 200 * 3840


async def test_failed_worker_cancels_full_text_queue_without_replaying(capture_session):
    s, events = capture_session
    calls = []

    class Speech(DemoSpeech):
        async def generate(self, text, *args):
            calls.append(text)
            yield bytes(3840)
            await asyncio.sleep(0.03)
            raise RuntimeError("worker exited")

    class Dialogue(DemoDialogue):
        async def generate(self, *args):
            yield "[mood=neutral] " + "Another sentence. " * 5

    s.speech, s.dialogue = Speech(), Dialogue()
    await s.reply("Hello")
    await asyncio.wait_for(s.task, 1)
    assert len(calls) == 1
    assert not s.delivered
    assert any(e["type"] == "cancel" for e in events)
    assert any(e["type"] == "error" for e in events)


async def test_redundant_wave_tag_does_not_suppress_greeting(capture_session):
    s, events = capture_session
    s.avatar_mode = "model"

    class Dialogue(DemoDialogue):
        async def generate(self, *args):
            yield "[mood=happy] [intent=greeting] [action=wave] Good morning!"

    s.dialogue = Dialogue()
    await s.reply("Hi, Ari", speak=False)
    await s.task
    assert any(e["type"] == "gesture" and e["gesture"] == "wave" for e in events)


async def test_diagnostics_require_auth_and_exclude_private_capture_contents(capture_session):
    s, _events = capture_session
    app = create_app(s.config)
    app.state.sessions.add(s)
    s.settings.vocabulary = ["private-vocabulary"]
    s.draft = "private-unsubmitted-words"
    s.recording = bytearray(b"private-audio")
    s.timings.record("recognition_finalization", 0.4)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.get("/speech-diagnostics")).status_code == 401
            response = await client.get(
                "/speech-diagnostics", headers={"Authorization": "Bearer " + s.config.token}
            )
            assert response.status_code == 200
            assert "private-" not in response.text
            exported = response.json()["sessions"][0]
            assert exported["vocabulary_entries"] == 1
            assert exported["records"][0]["seconds"] == 0.4
            assert not s.store.messages(s.conversation)
    finally:
        app.state.sessions.clear()
        app.state.store.close()
