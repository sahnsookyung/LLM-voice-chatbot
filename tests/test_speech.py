import asyncio
import base64
import json
import threading
from types import SimpleNamespace

import pytest
from companion.adapters import DemoDialogue, DemoSpeech, LocalSpeech
from companion.config import Config
from companion.protocol import decode_audio
from companion.session import Session
from companion.speech import DELIVERY, fish_model_ready, fish_text
from companion.store import Store
from companion.text import SpeechText
from companion.tts_worker import SpeechCancelled, install_fish_hooks


@pytest.mark.parametrize("expression", DELIVERY)
def test_mood_controls_stay_separate_from_spoken_words(expression):
    result = fish_text("I hear you.", expression)
    assert result.endswith("I hear you.")
    assert "[chuckle]" not in result
    assert result.startswith(DELIVERY[expression])


@pytest.mark.parametrize("text", ["Haha!", "Ha ha!", "Hehe.", "하하!", "ふふっ"])
def test_happy_laugh_is_a_vocal_event(text):
    assert fish_text(text, "happy") == "[chuckle]"


@pytest.mark.parametrize("expression", ["neutral", "concerned", "annoyed", "sleepy"])
def test_serious_expression_never_inserts_laughter(expression):
    assert "[chuckle]" not in fish_text("Haha!", expression)


@pytest.mark.parametrize(
    "text", ['She said "Haha!"', "I can't laugh today.", "That made me laugh."]
)
def test_discussing_or_quoting_laughter_is_still_speech(text):
    assert fish_text(text, "happy") == "[happy] " + text


def test_untrusted_delivery_tags_and_unknown_moods_are_not_instructions():
    assert fish_text("[screaming] Hello <|speaker:1|>", "evil") == "screaming Hello"
    assert fish_text("[] <|im_end|>", "happy") == ""


@pytest.mark.parametrize("chunk_size", [1, 3, 12, 500])
def test_spaced_streamed_metadata_never_reaches_display_or_voice(chunk_size):
    parser = SpeechText()
    text = (
        "[ Mood = Happy ] [ intent = greeting ] [ Action = WAVE ] Hello! [ Intent = greeting ] Hi!"
    )
    visible, spoken = "", []
    for start in range(0, len(text), chunk_size):
        delta, clauses = parser.feed(text[start : start + chunk_size])
        visible += delta
        spoken.extend(clauses)
    delta, clauses = parser.feed("", final=True)
    visible += delta
    spoken.extend(clauses)
    assert (parser.expression, parser.intent, parser.action) == ("happy", "greeting", "wave")
    assert visible.strip() == "Hello!  Hi!"
    assert spoken == ["Hello!", "Hi!"]


def test_fish_setup_requires_codec_and_tokenizer(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").touch()
    assert not fish_model_ready(tmp_path)
    for filename in ("codec.safetensors", "tokenizer.json", "tokenizer_config.json"):
        (tmp_path / filename).touch()
    assert fish_model_ready(tmp_path)


def test_reference_cache_invalidates_on_voice_or_transcript_change_and_cancels_mid_phrase():
    encoded = []

    def encode(audio, text):
        encoded.append((audio, text))
        return len(encoded)

    model = SimpleNamespace(_prepare_reference_prompt=encode, _sample_semantic=lambda: 42)
    cancelled = threading.Event()
    install_fish_hooks(model, cancelled)
    a, b = object(), object()
    assert model._prepare_reference_prompt(a, "a") == 1
    assert model._prepare_reference_prompt(a, "a") == 1
    assert model._prepare_reference_prompt(a, "corrected") == 2
    assert model._prepare_reference_prompt(b, "b") == 3
    assert model._prepare_reference_prompt(a, "a") == 4  # only one voice retained
    assert model._sample_semantic() == 42
    cancelled.set()
    with pytest.raises(SpeechCancelled):
        model._sample_semantic()
    with pytest.raises(SpeechCancelled):
        model._prepare_reference_prompt(a, "a")
    cancelled.clear()
    assert model._sample_semantic() == 42


@pytest.mark.asyncio
async def test_adapter_sends_emotion_and_retains_pcm_protocol(tmp_path):
    sent = []
    reader = asyncio.StreamReader()
    for event in (
        {"type": "audio", "pcm": base64.b64encode(b"\x00\x01" * 32).decode()},
        {"type": "done"},
    ):
        reader.feed_data((json.dumps(event) + "\n").encode())

    async def drain():
        pass

    speech = LocalSpeech(Config(data=tmp_path))
    speech.process = SimpleNamespace(
        returncode=None,
        stdout=reader,
        stdin=SimpleNamespace(write=sent.append, drain=drain),
    )
    received = [
        chunk async for chunk in speech.generate("Hello", "english", "ref.wav", "Hi", "happy")
    ]
    assert received == [b"\x00\x01" * 32]
    assert json.loads(sent[0])["expression"] == "happy"
    assert not speech.synthesizing


@pytest.mark.asyncio
@pytest.mark.parametrize("expression", [*DELIVERY, "unknown"])
async def test_session_voice_delivery_matches_face_and_history_stays_plain(tmp_path, expression):
    spoken, events = [], []

    class Dialogue(DemoDialogue):
        async def generate(self, *args):
            for letter in (
                f"[mood={expression}] [intent=conversation] [action=none] I hear you. Tell me more."
            ):
                yield letter

    class Speech(DemoSpeech):
        async def generate(self, text, language, reference, transcript, mood="neutral"):
            spoken.append((text, mood))
            yield bytes(480)

    async def send(event):
        events.append(event)

    async def audio(packet):
        turn, _, offset, pcm = decode_audio(packet)
        session.acknowledge(turn, offset + len(pcm) // 2)

    store = Store(tmp_path / "history.sqlite")
    session = Session(Config(data=tmp_path, demo=True), store, send, audio, Dialogue(), Speech())
    try:
        await session.reply("Hello", speak=True)
        await session.task
        mood = expression if expression in DELIVERY else "neutral"
        assert spoken == [("I hear you.", mood), ("Tell me more.", mood)]
        assert [e["expression"] for e in events if e["type"] == "expression"][-1] == mood
        assert not [e for e in events if e["type"] == "error"]
        assert all("[" not in row["generated"] for row in store.messages(session.conversation))
    finally:
        await session.close()
        store.close()
