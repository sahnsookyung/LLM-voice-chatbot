from pathlib import Path

import httpx
import pytest
from companion.adapters import DemoDialogue, DemoSpeech
from companion.app import create_app
from companion.config import Config
from companion.protocol import decode_audio
from companion.session import Session
from companion.store import Store
from companion.voices import resolve_voice

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.asyncio
async def test_voice_selection_preview_and_persistence(tmp_path):
    config = Config(root=ROOT, data=tmp_path, token="x" * 64, demo=True)
    app = create_app(config)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        url = "/characters/ari/voices"
        assert (await client.get(url)).status_code == 401
        assert (await client.get(url + "/bright/audio")).status_code == 401
        client.headers["Authorization"] = "Bearer " + config.token
        catalog = (await client.get(url)).json()
        assert catalog["selected"] == "character"
        assert {v["id"] for v in catalog["options"]} == {"character", "bright", "soft"}
        assert all("path" not in v for v in catalog["options"])
        assert (
            await client.put("/characters/ari/voice", json={"voice_id": "missing"})
        ).status_code == 404
        assert (
            await client.put("/characters/ari/voice", json={"voice_id": "../voice"})
        ).status_code == 422
        assert (
            await client.put("/characters/ari/voice", json={"voice_id": "bright"})
        ).status_code == 200
        assert (await client.get(url)).json()["selected"] == "bright"
        clip = await client.get(url + "/bright/audio")
        assert clip.status_code == 200 and clip.content[:4] == b"RIFF"
        assert clip.content == (ROOT / "voices/bright/preview-english.wav").read_bytes()
        assert (await client.get(url + "/missing/audio")).status_code == 404
    app.state.store.close()
    app = create_app(config)
    assert app.state.store.setting("voice:ari") == "bright"
    voice = resolve_voice(config, "ari", app.state.store.setting("voice:ari"))
    assert voice["path"] == ROOT / "voices/bright/voice.wav"
    assert "Ari" in voice["transcript"]
    app.state.store.close()


@pytest.mark.asyncio
async def test_character_without_original_voice_can_select_and_reset(tmp_path):
    config = Config(root=ROOT, data=tmp_path, token="x" * 64, demo=True)
    app = create_app(config)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer " + config.token},
    ) as client:
        response = await client.post(
            "/characters",
            json={"name": "Test", "persona": "An adult character.", "use_ari_voice": False},
        )
        assert response.status_code == 200
        identifier = response.json()["id"]
        url = f"/characters/{identifier}"
        for choice in ("soft", "character"):
            assert (await client.put(url + "/voice", json={"voice_id": choice})).status_code == 200
            assert (await client.get(url + "/voices")).json()["selected"] == choice
        assert (await client.get(url + "/voices/character/audio")).status_code == 404
    app.state.store.close()


@pytest.mark.asyncio
async def test_reply_uses_the_selected_voice_reference(tmp_path):
    selected = []

    class Speech(DemoSpeech):
        async def generate(self, text, language, reference, transcript):
            selected.append((reference, transcript))
            yield bytes(2400)

    async def send(event):
        pass

    async def audio(packet):
        turn, _, offset, pcm = decode_audio(packet)
        session.acknowledge(turn, offset + len(pcm) // 2)

    config = Config(root=ROOT, data=tmp_path, token="x" * 64, demo=True)
    store = Store(tmp_path / "history.sqlite")
    store.set_setting("voice:ari", "soft")
    session = Session(config, store, send, audio, DemoDialogue(), Speech())
    try:
        await session.reply("Hello", speak=True)
        await session.task
        expected = resolve_voice(config, "ari", "soft")
        assert selected and all(
            value == (expected["path"], expected["transcript"]) for value in selected
        )
        assert store.messages(session.conversation)[-1]["status"] == "complete"
    finally:
        await session.close()
        store.close()
