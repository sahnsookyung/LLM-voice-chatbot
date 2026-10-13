import asyncio
import hmac
import shutil
import tempfile
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from .adapters import DemoDialogue, DemoSpeech, LocalDialogue, LocalSpeech, SetupRequired
from .characters import character_path, import_bundle, load_character, safe_asset
from .config import Config
from .protocol import CharacterInput, Command, MemoryInput, PersonaInput, VoiceInput
from .session import Session
from .store import Store
from .voices import resolve_voice, voice_catalog


def create_app(config: Config):
    if len(config.token) < 32:
        raise ValueError("LVC_TOKEN must contain at least 32 random characters")
    store = Store(config.data / "history.sqlite")
    active = set()
    session_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        yield
        await asyncio.gather(*(s.close() for s in list(active)), return_exceptions=True)
        store.close()

    app = FastAPI(
        title="Local Voice Companion",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.store, app.state.sessions = store, active

    def authorized(headers):
        return hmac.compare_digest(headers.get("authorization", ""), f"Bearer {config.token}")

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if not authorized(request.headers):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        # Native client does not need browser origins; reject browser-initiated requests.
        if request.headers.get("origin"):
            return JSONResponse({"detail": "Browser origins are not allowed"}, status_code=403)
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "version": 1,
            "demo": config.demo,
            "session_active": session_lock.locked(),
        }

    @app.get("/conversations")
    async def conversations():
        return store.conversations()

    @app.get("/conversations/{identifier}")
    async def conversation(identifier: str):
        return store.messages(identifier)

    @app.delete("/conversations/{identifier}")
    async def delete_conversation(identifier: str):
        if any(s.conversation == identifier for s in active):
            raise HTTPException(409, "Start a new conversation before deleting the active one")
        store.delete_conversation(identifier)
        return {"deleted": True}

    @app.get("/export")
    async def export():
        return {
            "version": 1,
            "conversations": [
                {**row, "messages": store.messages(row["id"])}
                for row in store.conversations(limit=None)
            ],
            "memories": store.memories(),
        }

    @app.get("/memories")
    async def memories():
        return store.memories()

    @app.post("/memories")
    async def add_memory(memory: MemoryInput):
        if not memory.approved or not memory.text.strip():
            raise HTTPException(400, "Explicit approval is required to save a memory")
        if len(store.memories()) >= 50:
            raise HTTPException(400, "Review your saved memories before adding more")
        return {"id": store.add_memory(memory.text.strip())}

    @app.delete("/memories/{identifier}")
    async def delete_memory(identifier: str):
        # Stop active generations so deleted facts cannot remain in a running request.
        for session in list(active):
            await session.stop()
            await session.dialogue.unload()
        store.delete_memory(identifier)
        return {"deleted": True}

    @app.put("/memories/{identifier}")
    async def edit_memory(identifier: str, memory: MemoryInput):
        if not memory.approved or not memory.text.strip():
            raise HTTPException(400, "Explicit approval and nonempty text are required")
        for session in list(active):
            await session.stop()
            await session.dialogue.unload()
        if not store.update_memory(identifier, memory.text.strip()):
            raise HTTPException(404, "Memory not found")
        return {"saved": True}

    @app.get("/characters")
    async def characters():
        def describe(identifier):
            value = load_character(config, identifier).model_dump()
            value.update(store.setting("persona:" + identifier, {}))
            return value

        result = [describe("ari")]
        root = config.data / "characters"
        for directory in sorted(root.glob("*")) if root.exists() else []:
            if directory.is_dir():
                try:
                    result.append(describe(directory.name))
                except (ValueError, OSError):
                    continue
        return result

    @app.post("/characters")
    async def create_character(value: CharacterInput):
        original = load_character(config, "ari")
        identifier = "character-" + uuid.uuid4().hex[:12]
        root = config.data / "characters" / identifier
        root.mkdir(parents=True)
        character = original.model_copy(
            update={
                "id": identifier,
                "name": value.name,
                "persona": value.persona,
                "palette": value.palette,
                "voice": original.voice if value.use_ari_voice else None,
            }
        )
        try:
            if character.voice:
                shutil.copyfile(
                    character_path(config, "ari") / character.voice.audio,
                    root / character.voice.audio,
                )
            (root / "character.json").write_text(character.model_dump_json(indent=2))
        except OSError:
            shutil.rmtree(root)
            raise
        return character.model_dump()

    @app.get("/characters/{identifier}")
    async def character(identifier: str):
        if not identifier.replace("-", "").replace("_", "").isalnum():
            raise HTTPException(400, "Invalid character")
        try:
            result = load_character(config, identifier).model_dump()
            result.update(store.setting("persona:" + identifier, {}))
            return result
        except (ValueError, OSError):
            raise HTTPException(404, "Character not found")

    @app.put("/characters/{identifier}/persona")
    async def persona(identifier: str, value: PersonaInput):
        await character(identifier)
        store.set_setting("persona:" + identifier, value.model_dump())
        return {"saved": True}

    @app.get("/characters/{identifier}/voices")
    async def voices(identifier: str):
        await character(identifier)
        options = voice_catalog(config, identifier)
        selected = store.setting("voice:" + identifier, "character")
        if not any(v["id"] == selected for v in options):
            selected = "character"
        return {
            "selected": selected,
            "options": [{k: v[k] for k in ("id", "name", "description")} for v in options],
        }

    @app.put("/characters/{identifier}/voice")
    async def select_voice(identifier: str, value: VoiceInput):
        await character(identifier)
        if (
            value.voice_id != "character"
            and resolve_voice(config, identifier, value.voice_id) is None
        ):
            raise HTTPException(404, "Voice not found")
        for session in list(active):
            if session.character_id == identifier:
                await session.stop()
        store.set_setting("voice:" + identifier, value.voice_id)
        return {"saved": True}

    @app.get("/characters/{identifier}/voices/{voice_id}/audio")
    async def preview_voice(identifier: str, voice_id: str):
        await character(identifier)
        voice = resolve_voice(config, identifier, voice_id)
        if voice is None:
            raise HTTPException(404, "Voice not found")
        preview = voice["path"].with_name("preview-fish.wav")
        return FileResponse(preview if preview.is_file() else voice["path"], media_type="audio/wav")

    @app.get("/characters/{identifier}/assets/{name:path}")
    async def asset(identifier: str, name: str):
        await character(identifier)
        try:
            path = safe_asset(character_path(config, identifier), name)
            if path.suffix.lower() not in {".png", ".wav"}:
                raise ValueError("Unsupported asset")
            return FileResponse(path)
        except ValueError:
            raise HTTPException(404, "Asset not found")

    @app.post("/characters/import")
    async def import_character(file: UploadFile):
        with tempfile.TemporaryDirectory(dir=config.data) as temp:
            path = Path(temp) / "character.zip"
            size = 0
            with path.open("wb") as output:
                while data := await file.read(1024 * 1024):
                    size += len(data)
                    if size > 64 * 1024 * 1024:
                        raise HTTPException(413, "Character archive exceeds 64 MiB")
                    output.write(data)
            try:
                imported = await asyncio.to_thread(import_bundle, path, config.data / "characters")
            except (ValueError, OSError, zipfile.BadZipFile) as error:
                raise HTTPException(400, str(error))
            return imported.model_dump()

    @app.websocket("/session")
    async def websocket(ws: WebSocket):
        if not authorized(ws.headers) or ws.headers.get("origin"):
            await ws.close(code=4401)
            return
        if session_lock.locked():
            await ws.close(code=4409, reason="One companion session is already active")
            return
        async with session_lock:
            await serve_session(ws)

    async def serve_session(ws):
        await ws.accept()
        lock = asyncio.Lock()

        async def send(event):
            async with lock:
                await ws.send_json(event)

        async def send_audio(data):
            async with lock:
                await ws.send_bytes(data)

        conversation_id = ws.query_params.get("conversation")
        if conversation_id and not store.has_conversation(conversation_id):
            await ws.close(code=4404)
            return
        session = Session(
            config,
            store,
            send,
            send_audio,
            DemoDialogue() if config.demo else LocalDialogue(config),
            DemoSpeech() if config.demo else LocalSpeech(config),
            conversation_id,
        )
        active.add(session)
        try:
            await send(
                {
                    "type": "ready",
                    "version": 1,
                    "conversation_id": session.conversation,
                    "profile": session.profile,
                    "language": session.language,
                    "character_id": session.character_id,
                    "demo": config.demo,
                }
            )
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    await session.input_audio(message["bytes"])
                    continue
                raw = message.get("text", "")
                if len(raw) > 20000:
                    await ws.close(code=1009)
                    break
                try:
                    command = Command.model_validate_json(raw)
                    if command.type == "text":
                        await session.reply(command.text, command.speak)
                    elif command.type == "start":
                        await session.start_listening(command.mode)
                    elif command.type == "commit":
                        await session.commit()
                    elif command.type == "mute":
                        await session.mute()
                    elif command.type == "stop":
                        await session.stop()
                        await session.event(
                            "state", state="listening" if session.listening else "idle"
                        )
                    elif command.type == "ack":
                        session.acknowledge(command.turn_id, command.samples)
                    elif command.type == "profile":
                        await session.set_profile(command.profile)
                    elif command.type == "language":
                        await session.mute()
                        session.language = command.language
                        store.set_setting("language", command.language)
                    elif command.type == "character":
                        load_character(config, command.character_id)
                        await session.stop()
                        await session.speech.close()
                        session.character_id = command.character_id
                        store.set_setting("character", command.character_id)
                        await session.event("character", character_id=command.character_id)
                    elif command.type == "pressure":
                        await session.pressure(command.pressure)
                    elif command.type == "avatar":
                        session.avatar_mode = command.avatar_mode
                        session.avatar_visible = command.avatar_visible
                        session.reduced_motion = command.reduced_motion
                except ValidationError:
                    await session.event("error", message="Invalid application command")
                except (SetupRequired, ValueError, OSError) as error:
                    await session.event("error", message=str(error))
                except Exception:  # noqa: BLE001 — isolate provider failure at the service boundary
                    await session.event(
                        "error", message="The operation failed. Check local runtime diagnostics."
                    )
        except WebSocketDisconnect:
            pass
        finally:
            try:
                await session.close()
            finally:
                active.discard(session)

    return app
