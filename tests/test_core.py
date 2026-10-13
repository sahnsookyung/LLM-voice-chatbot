import asyncio
import json
import struct
import uuid
import zipfile
from pathlib import Path

import httpx
import pytest
from companion.app import create_app
from companion.characters import import_bundle, validate_character
from companion.config import Config
from companion.protocol import audio_packet, decode_audio
from companion.store import Store
from companion.text import SpeechText, speech_language

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.asyncio
async def test_session_is_reserved_before_websocket_accept(tmp_path):
    config = Config(root=ROOT, data=tmp_path, token="x" * 64, demo=True)
    app = create_app(config)
    endpoint = next(route.endpoint for route in app.routes if route.path == "/session")
    accepting = asyncio.Event()

    class Socket:
        def __init__(self):
            self.headers = {"authorization": "Bearer " + config.token}
            self.closed = None

        async def accept(self):
            accepting.set()
            await asyncio.Event().wait()

        async def close(self, code, reason=None):
            self.closed = code

    first = asyncio.create_task(endpoint(Socket()))
    await accepting.wait()
    second = Socket()
    await endpoint(second)
    assert second.closed == 4409
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test", headers=Socket().headers
    ) as client:
        assert not (await client.get("/health")).json()["session_active"]
    app.state.store.close()


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "history.sqlite")
    yield value
    value.close()


def test_delivered_context_and_crash_recovery(tmp_path):
    path = tmp_path / "history.sqlite"
    s = Store(path)
    c = s.create_conversation()
    s.add_message(c, "user", "Question")
    m = s.add_message(c, "assistant", "")
    s.update_message(m, "Heard sentence. Unheard secret.", "Heard sentence.", "generating")
    s.close()
    s = Store(path)
    assert s.messages(c)[1]["status"] == "interrupted"
    assert "Unheard secret" not in str(s.context(c))
    assert "Heard sentence." in str(s.context(c))
    s.delete_conversation(c)
    assert s.messages(c) == []
    s.close()


def test_memory_changes_clear_summaries(store):
    c = store.create_conversation()
    store.db.execute("UPDATE conversations SET summary=? WHERE id=?", ("old fact", c))
    m = store.add_memory("Approved fact")
    assert store.conversations()[0]["summary"] == ""
    store.delete_memory(m)
    assert store.memories() == []


def test_continued_speech_does_not_invent_assistant_context(store):
    c = store.create_conversation()
    store.add_message(c, "user", "First sentence.")
    m = store.add_message(c, "assistant", "")
    store.update_message(m, "Unheard words.", "", "interrupted")
    store.add_message(c, "user", "And a continuation.")
    context = store.context(c)
    assert context[0] == {"role": "user", "content": "First sentence."}
    assert context[1] == {"role": "assistant", "content": ""}
    assert context[2]["content"].endswith("And a continuation.")
    m = store.add_message(c, "assistant", "")
    store.update_message(m, "Heard words. Unheard words.", "Heard words.", "interrupted")
    store.add_message(c, "user", "A correction.")
    context = store.context(c)
    assert context[-2] == {"role": "assistant", "content": "Heard words."}
    assert "interrupted" in context[-1]["content"]
    assert "Unheard" not in str(context)


@pytest.mark.parametrize(
    "parts,expected,mood",
    [
        (["[mo", "od=happy] Hel", "lo there. More text."], ["Hello there.", "More text."], "happy"),
        (["안녕하세요. 반가워요!"], ["안녕하세요.", "반가워요!"], "neutral"),
        (["こんにちは。元気ですか？"], ["こんにちは。", "元気ですか？"], "neutral"),
        (["That costs 3.14 dollars."], ["That costs 3.14 dollars."], "neutral"),
        (["[mood=malicious] Hello."], ["Hello."], "neutral"),
        (["", "\n", " [mo", "od=surprised] Wow!"], ["Wow!"], "surprised"),
        (["  ", "Plain text."], ["Plain text."], "neutral"),
    ],
)
def test_incremental_speech(parts, expected, mood):
    parser = SpeechText()
    clauses = []
    display = ""
    for part in parts:
        visible, ready = parser.feed(part)
        display += visible
        clauses += ready
    visible, ready = parser.feed("", final=True)
    clauses += ready
    assert clauses == expected
    assert parser.expression == mood
    assert "[mood=" not in display


def test_audio_envelope_roundtrip():
    turn = str(uuid.uuid4())
    pcm = struct.pack("<hhh", -32768, 0, 32767)
    assert decode_audio(audio_packet(turn, 2, 100, pcm)) == (turn, 2, 100, pcm)
    with pytest.raises(ValueError):
        decode_audio(b"invalid")


def test_language_detection():
    assert speech_language("반가워요", "auto") == "korean"
    assert speech_language("元気ですか", "auto") == "japanese"
    assert speech_language("hello", "japanese") == "japanese"


def bundle(tmp_path, entries):
    path = tmp_path / "input.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name, value in entries.items():
            z.writestr(name, value)
    return path


def test_import_rejects_traversal_and_executable(tmp_path):
    for entries in ({"../outside.txt": "bad"}, {"script.js": "alert(1)"}):
        with pytest.raises(ValueError):
            import_bundle(bundle(tmp_path, entries), tmp_path / "characters")
    assert not (tmp_path / "outside.txt").exists()


def test_original_character_and_safe_import(tmp_path):
    original = validate_character(ROOT / "characters/ari")
    assert original.procedural
    data = original.model_dump()
    data["id"] = "custom"
    data["voice"] = None
    path = bundle(tmp_path, {"character.json": json.dumps(data)})
    assert import_bundle(path, tmp_path / "characters").id == "custom"
    with pytest.raises(ValueError):
        import_bundle(path, tmp_path / "characters")


@pytest.mark.asyncio
async def test_api_auth_memory_and_export(tmp_path):
    config = Config(root=ROOT, data=tmp_path, token="x" * 64, demo=True)
    app = create_app(config)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.get("/health")).status_code == 401
        client.headers["Authorization"] = "Bearer " + config.token
        assert (await client.get("/health")).status_code == 200
        assert (
            await client.get("/health", headers={"Origin": "https://example.com"})
        ).status_code == 403
        assert (await client.post("/memories", json={"text": "fact"})).status_code == 400
        result = await client.post("/memories", json={"text": "fact", "approved": True})
        assert result.status_code == 200
        export = (await client.get("/export")).json()
        assert export["memories"][0]["text"] == "fact"
        assert (await client.delete("/memories/" + result.json()["id"])).status_code == 200
        assert (await client.get("/memories")).json() == []
        result = await client.post("/characters/import", files={"file": ("bad.zip", b"not zip")})
        assert result.status_code == 400
    app.state.store.close()


def test_sharded_model_readiness_and_memory_edit(store, tmp_path):
    from companion.config import model_ready

    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "one.safetensors", "b": "two.safetensors"}})
    )
    (tmp_path / "one.safetensors").touch()
    assert not model_ready(tmp_path)
    (tmp_path / "two.safetensors").touch()
    assert model_ready(tmp_path)
    identifier = store.add_memory("Old fact")
    assert store.update_memory(identifier, "New fact")
    assert store.memories()[0]["text"] == "New fact"
    assert not store.update_memory("missing", "no")


@pytest.mark.asyncio
async def test_memory_edit_requires_approval(tmp_path):
    app = create_app(Config(root=ROOT, data=tmp_path, token="m" * 64, demo=True))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer " + ("m" * 64)},
    ) as client:
        identifier = (
            await client.post("/memories", json={"text": "Original", "approved": True})
        ).json()["id"]
        assert (
            await client.put("/memories/" + identifier, json={"text": "Updated"})
        ).status_code == 400
        assert (
            await client.put("/memories/" + identifier, json={"text": "Updated", "approved": True})
        ).status_code == 200
        assert (await client.get("/memories")).json()[0]["text"] == "Updated"
        assert (
            await client.post("/memories", json={"text": "   ", "approved": True})
        ).status_code == 400
    app.state.store.close()


@pytest.mark.asyncio
async def test_create_character_is_data_only_and_selectable(tmp_path):
    app = create_app(Config(root=ROOT, data=tmp_path, token="c" * 64, demo=True))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer " + ("c" * 64)},
    ) as client:
        response = await client.post(
            "/characters",
            json={
                "name": "Test adult character",
                "persona": "A thoughtful adult character.",
                "palette": "plum",
                "use_ari_voice": False,
            },
        )
        assert response.status_code == 200
        character = response.json()
        assert character["voice"] is None
        assert character["palette"] == "plum"
        assert (await client.get("/characters/" + character["id"])).status_code == 200
        assert (
            await client.post(
                "/characters", json={"name": "X", "persona": "X", "palette": "javascript:alert(1)"}
            )
        ).status_code == 422
    app.state.store.close()
