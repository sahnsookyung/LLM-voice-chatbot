import asyncio
from pathlib import Path

import pytest
from companion.adapters import DemoDialogue, DemoSpeech
from companion.behavior import GesturePolicy
from companion.config import Config
from companion.protocol import decode_audio
from companion.session import Session
from companion.store import Store
from companion.text import EXPRESSIONS


@pytest.mark.asyncio
@pytest.mark.parametrize("speak", [False, True])
@pytest.mark.parametrize(
    "mood,intent,gesture",
    [
        ("happy", "greeting", "wave"),
        ("neutral", "explaining", "explain"),
        ("neutral", "acknowledging", "nod"),
        ("neutral", "uncertain", "shrug"),
        ("concerned", "greeting", None),
        ("annoyed", "uncertain", None),
    ],
)
async def test_reply_intent_drives_one_turn_scoped_action_before_speech(
    tmp_path, speak, mood, intent, gesture
):
    events, spoken = [], []

    class Dialogue(DemoDialogue):
        async def generate(self, messages, profile):
            for char in f"[mood={mood}] [intent={intent}] First sentence. Another sentence.":
                yield char

    class Speech(DemoSpeech):
        async def generate(self, text, *args):
            spoken.append(text)
            if gesture:
                assert any(e["type"] == "gesture" for e in events)
            yield bytes(480)

    class Always:
        def random(self):
            return 0

    async def send(event):
        events.append(event)

    async def audio(packet):
        turn, _seq, offset, pcm = decode_audio(packet)
        session.acknowledge(turn, offset + len(pcm) // 2)

    store = Store(tmp_path / "history.sqlite")
    session = Session(
        Config(data=tmp_path, token="x" * 64, demo=True), store, send, audio, Dialogue(), Speech()
    )
    session.gestures = GesturePolicy(Always())
    try:
        await session.reply("A synthetic request", speak=speak)
        await asyncio.wait_for(session.task, 2)
        assert not any(e["type"] == "error" for e in events)
        gestures = [e for e in events if e["type"] == "gesture"]
        assert [e["gesture"] for e in gestures] == ([gesture] if gesture else [])
        if gesture:
            assert gestures[0]["turn_id"] == session.turn_id
            assert gestures[0]["wait_for_audio"] == speak
            assert events.index(gestures[0]) < next(
                i for i, e in enumerate(events) if e["type"] == "text"
            )
        assert session.generated.strip() == "First sentence. Another sentence."
        assert spoken == (["First sentence.", "Another sentence."] if speak else [])
    finally:
        await session.close()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("speak", [False, True])
@pytest.mark.parametrize(
    "prefix,expression",
    [(f"[mood={mood}]", mood) for mood in sorted(EXPRESSIONS)]
    + [("", "neutral"), ("[mood=unknown]", "neutral")],
)
async def test_reply_expression_reaches_client_without_leaking_tags(
    tmp_path, speak, prefix, expression
):
    events, spoken = [], []

    class Dialogue(DemoDialogue):
        calls = 0

        async def generate(self, messages, profile):
            self.calls += 1
            yield ""
            yield "\n"
            for char in prefix:
                yield char
            if prefix:
                # The expression is available before any reply text or synthesized audio.
                assert events[-1]["type"] == "expression"
            yield "Hello there. "
            yield "How are you?"

    class Speech(DemoSpeech):
        async def generate(self, text, *args):
            spoken.append(text)
            assert any(e["type"] == "expression" for e in events)
            yield bytes(480)

    async def send(event):
        events.append(event)

    async def audio(packet):
        turn, _seq, offset, pcm = decode_audio(packet)
        session.acknowledge(turn, offset + len(pcm) // 2)

    store = Store(tmp_path / "history.sqlite")
    dialogue = Dialogue()
    session = Session(
        Config(data=tmp_path, token="x" * 64, demo=True), store, send, audio, dialogue, Speech()
    )
    try:
        await session.reply("Hi", speak=speak)
        await asyncio.wait_for(session.task, 2)
        assert not any(e["type"] == "error" for e in events)
        expressions = [e for e in events if e["type"] == "expression"]
        assert expressions == [
            {
                "version": 1,
                "type": "expression",
                "turn_id": session.turn_id,
                "expression": expression,
                "sample_offset": 0,
            }
        ]
        first_text = next(e for e in events if e["type"] == "text")
        assert events.index(expressions[0]) < events.index(first_text)
        displayed = "".join(e["text"] for e in events if e["type"] == "text")
        assert displayed.strip() == "Hello there. How are you?"
        assert spoken == (["Hello there.", "How are you?"] if speak else [])
        row = store.messages(session.conversation)[-1]
        assert row["status"] == "complete"
        assert row["generated"].strip() == row["delivered"].strip() == displayed.strip()
        assert dialogue.calls == 1
    finally:
        await session.close()
        store.close()


@pytest.mark.asyncio
async def test_playback_acknowledgments_control_memory(tmp_path):
    store = Store(tmp_path / "history.sqlite")
    events = []
    packets = []

    async def send(event):
        events.append(event)

    async def send_audio(packet):
        packets.append(decode_audio(packet))

    s = Session(
        Config(root=Path(__file__).resolve().parents[1], data=tmp_path, token="x" * 64, demo=True),
        store,
        send,
        send_audio,
        DemoDialogue(),
        DemoSpeech(),
    )
    await s.reply("Hi", True)
    async with asyncio.timeout(3):
        while s.sent_samples < 24000:
            await asyncio.sleep(0.01)
    assert s.delivered == ""
    s.acknowledge("wrong", 24000)
    assert s.acked == 0
    s.acknowledge(s.turn_id, 999999)
    assert s.acked == 0
    s.acknowledge(s.turn_id, 24000)
    async with asyncio.timeout(3):
        while s.sent_samples < 48000:
            await asyncio.sleep(0.01)
    assert "development preview" in s.delivered
    await s.stop()
    rows = store.messages(s.conversation)
    assert rows[-1]["status"] == "interrupted"
    assert "configured separately" not in rows[-1]["delivered"]
    assert any(e["type"] == "cancel" for e in events)
    assert all(p[0] == s.turn_id for p in packets)
    await s.close()
    store.close()


@pytest.mark.asyncio
async def test_text_only_reply_and_close_are_idempotent(tmp_path):
    store = Store(tmp_path / "history.sqlite")
    events = []

    async def send(e):
        events.append(e)

    s = Session(
        Config(root=Path(__file__).resolve().parents[1], data=tmp_path, token="x" * 64, demo=True),
        store,
        send,
        send,
        DemoDialogue(),
        DemoSpeech(),
    )
    await s.reply("Hi", False)
    await s.task
    row = store.messages(s.conversation)[-1]
    assert row["generated"] == row["delivered"]
    assert row["status"] == "complete"
    n = len(events)
    await s.close()
    await s.close()
    assert len(events) == n
    store.close()


@pytest.mark.asyncio
async def test_handsfree_skips_idle_asr_work_and_preserves_preroll(tmp_path):
    class Detector:
        calls = 0

        async def feed(self, pcm):
            self.calls += 1
            return self.calls == 3, False

        async def close(self):
            pass

    class Recognition:
        def __init__(self):
            self.chunks = []

        async def feed(self, pcm):
            self.chunks.append(pcm)

        async def disconnect(self):
            pass

        async def close(self):
            pass

    async def send(event):
        pass

    store = Store(tmp_path / "history.sqlite")
    s = Session(
        Config(root=Path(__file__).resolve().parents[1], data=tmp_path, token="x" * 64, demo=True),
        store,
        send,
        send,
        DemoDialogue(),
        DemoSpeech(),
    )
    s.detector = Detector()
    s.recognizer = Recognition()
    s.listening = True
    s.mode = "handsfree"
    s.audio_task = asyncio.create_task(s.consume_audio())
    await s.input_audio(bytes(1024))
    await s.audio_queue.join()
    await s.input_audio(bytes(1024))
    await s.audio_queue.join()
    assert not s.recognizer.chunks
    await s.input_audio(bytes(1024))
    await s.audio_queue.join()
    assert len(s.recognizer.chunks) == 3
    await s.close()
    store.close()


@pytest.mark.asyncio
async def test_input_drain_waits_for_end_of_turn_commit(tmp_path):
    committing, release = asyncio.Event(), asyncio.Event()

    class Detector:
        async def feed(self, pcm):
            return True, True

        async def close(self):
            pass

    class Recognition:
        async def feed(self, pcm):
            pass

        async def commit(self):
            committing.set()
            await release.wait()
            return ""

        async def disconnect(self):
            pass

        async def close(self):
            pass

    async def send(event):
        pass

    store = Store(tmp_path / "history.sqlite")
    session = Session(
        Config(root=Path(__file__).resolve().parents[1], data=tmp_path, token="x" * 64, demo=True),
        store,
        send,
        send,
        DemoDialogue(),
        DemoSpeech(),
    )
    session.detector, session.recognizer = Detector(), Recognition()
    session.listening, session.mode = True, "handsfree"
    session.audio_task = asyncio.create_task(session.consume_audio())
    await session.input_audio(bytes(1024))
    await asyncio.wait_for(committing.wait(), 1)
    drained = asyncio.create_task(session.audio_queue.join())
    await asyncio.sleep(0)
    assert not drained.done()
    release.set()
    await asyncio.wait_for(drained, 1)
    await session.close()
    store.close()


@pytest.mark.asyncio
async def test_empty_model_reply_fails_visibly(tmp_path):
    class EmptyDialogue(DemoDialogue):
        async def generate(self, messages, profile):
            yield ""

    events = []

    async def send(event):
        events.append(event)

    store = Store(tmp_path / "history.sqlite")
    session = Session(
        Config(root=Path(__file__).resolve().parents[1], data=tmp_path, token="x" * 64, demo=True),
        store,
        send,
        send,
        EmptyDialogue(),
        DemoSpeech(),
    )
    await session.reply("Hi", speak=False)
    await session.task
    assert any(e["type"] == "error" and "empty reply" in e["message"] for e in events)
    assert not any(e["type"] == "done" for e in events)
    assert store.messages(session.conversation)[-1]["status"] == "interrupted"
    await session.close()
    store.close()


@pytest.mark.asyncio
async def test_provider_large_audio_chunk_progresses_with_immediate_acks(tmp_path):
    class BurstSpeech(DemoSpeech):
        async def generate(self, *args):
            yield bytes(60000)

    store = Store(tmp_path / "history.sqlite")

    async def send(event):
        pass

    async def audio(packet):
        turn, _seq, offset, pcm = decode_audio(packet)
        assert len(pcm) <= 15360
        s.acknowledge(turn, offset + len(pcm) // 2)

    s = Session(
        Config(root=Path(__file__).resolve().parents[1], data=tmp_path, token="x" * 64, demo=True),
        store,
        send,
        audio,
        DemoDialogue(),
        BurstSpeech(),
    )
    await s.reply("Hi", True)
    await asyncio.wait_for(s.task, 2)
    assert s.sent_samples == s.acked
    assert store.messages(s.conversation)[-1]["status"] == "complete"
    await s.close()
    store.close()
