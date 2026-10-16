"""Authenticated Ari API for Codex; local history and worker lifecycle stay separate."""

import asyncio
import time

from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from .codex_compat import probe, socket_path
from .codex_rpc import CodexError, CodexRPC
from .codex_types import BackendChoice
from .codex_validation import TransientCodexSettings, validation_ready, validation_thread
from .protocol import Command


class CodexCommands:
    """Slow history/catalog RPCs cannot block microphone, ACK, mute or Stop delivery."""

    def __init__(self, session):
        self.session = session
        self.queue = asyncio.Queue(maxsize=8)
        self.busy = False
        self.worker = asyncio.create_task(self.run())

    def submit(self, command):
        try:
            thread = self.session.thread["id"] if self.session.thread else None
            self.queue.put_nowait((thread, command))
        except asyncio.QueueFull as error:
            raise CodexError(
                "Too many pending Codex controls. Wait for history to finish loading."
            ) from error

    async def run(self):
        while True:
            thread, command = await self.queue.get()
            self.busy = True
            try:
                current = self.session.thread["id"] if self.session.thread else None
                scoped = command.type == "text" or (
                    command.type == "codex"
                    and command.codex.action in {"settings", "voice_select", "history_more"}
                )
                if scoped and thread != current:
                    raise CodexError(
                        "Task changed while text was queued. Submit it explicitly to the selected task."
                    )
                if command.type == "text":
                    await self.session.text(command.text)
                else:
                    await self.session.command(command.codex)
            except CodexError as error:
                await self.session.event("notice", message=str(error))
            except (ValueError, TypeError, KeyError):
                await self.session.event(
                    "notice",
                    message="Codex could not complete that control. Refresh history and check task status before retrying.",
                )
            finally:
                self.busy = False
                self.queue.task_done()

    def discard_pending(self):
        while not self.queue.empty():
            self.queue.get_nowait()
            self.queue.task_done()

    async def close(self):
        self.worker.cancel()
        await asyncio.gather(self.worker, return_exceptions=True)


class CodexService:
    def __init__(self, config, store):
        self.config, self.store = config, store
        self.active = set()
        self.cached_probe = None
        self.probed_at = 0
        self.probe_lock = asyncio.Lock()
        self.validation_thread = (
            validation_thread(config.codex_validation_thread)
            if config.codex_validation_thread
            else None
        )

    def permitted(self, readiness):
        return (
            validation_ready(readiness)
            if self.validation_thread
            else readiness.get("available") is True
        )

    async def status(self, force=False):
        async with self.probe_lock:
            if force or self.cached_probe is None or time.monotonic() - self.probed_at > 30:
                self.cached_probe = await probe(self.config.runtime / "codex/acceptance.json")
                self.probed_at = time.monotonic()
            return self.cached_probe

    async def close(self):
        await asyncio.gather(
            *(session.close() for session in list(self.active)), return_exceptions=True
        )


def install_codex(app, config, store, authorized, session_lock):
    service = CodexService(config, store)

    @app.get("/backend")
    async def backend():
        return {
            "selected": "codex" if service.validation_thread else store.setting("backend", "local"),
            "codex": await service.status(),
            "validation": {
                "active": bool(service.validation_thread),
                "thread_id": service.validation_thread,
            },
        }

    @app.get("/codex/status")
    async def status():
        return await service.status(force=True)

    @app.get("/codex/diagnostics")
    async def diagnostics():
        return {
            "version": 1,
            "compatibility": await service.status(),
            "sessions": [s.timings.export() for s in service.active],
        }

    @app.put("/backend")
    async def choose(value: BackendChoice):
        if service.validation_thread:
            raise HTTPException(
                409,
                "This validation run is scoped to one task. Reopen Ari normally to restore your saved backend.",
            )
        if value.backend == "codex" and not (await service.status(force=True))["available"]:
            raise HTTPException(
                409,
                "Codex shared-host acceptance is incomplete. Local mode remains selected; see Codex readiness.",
            )
        store.set_setting("backend", value.backend)
        return {"selected": value.backend}

    @app.websocket("/codex/session")
    async def socket(ws: WebSocket):
        if not authorized(ws.headers) or ws.headers.get("origin"):
            await ws.close(code=4401)
            return
        if session_lock.locked():
            await ws.close(code=4409, reason="Another Ari session is still closing")
            return
        async with session_lock:
            await ws.accept()
            if not service.permitted(await service.status(force=True)):
                await ws.send_json(
                    {
                        "type": "error",
                        "message": "Codex shared-host readiness checks have not passed. Open Settings → Codex readiness.",
                    }
                )
                await ws.close(code=4412)
                return
            from .codex_session import CodexSession

            send_lock = asyncio.Lock()

            async def send(event):
                async with send_lock:
                    await ws.send_json(event)

            async def send_audio(packet):
                async with send_lock:
                    await ws.send_bytes(packet)

            rpc = CodexRPC(socket_path())
            session = None
            commands = None
            expiry = None
            try:
                await rpc.connect()
                settings = (
                    TransientCodexSettings(store, service.validation_thread)
                    if service.validation_thread
                    else store
                )
                session = CodexSession(
                    rpc, settings, send, send_audio, validation_thread=service.validation_thread
                )
                service.active.add(session)
                await session.open()
                if service.validation_thread:

                    async def expire():
                        await asyncio.sleep(3600)
                        await ws.close(
                            code=4412,
                            reason="Validation session expired. Reopen explicitly to continue testing.",
                        )

                    expiry = asyncio.create_task(expire())
                commands = CodexCommands(session)
                while True:
                    message = await ws.receive()
                    if message["type"] == "websocket.disconnect":
                        break
                    try:
                        if message.get("bytes") is not None:
                            await session.input_audio(message["bytes"])
                            continue
                        raw = message.get("text", "")
                        if len(raw) > 20000:
                            raise ValueError("Application command is too large.")
                        command = Command.model_validate_json(raw)
                        if command.type == "codex" and command.codex:
                            if command.codex.action in {
                                "voice_end",
                                "stop_task",
                                "approval",
                                "answer",
                            }:
                                if command.codex.action == "stop_task":
                                    commands.discard_pending()
                                await session.command(command.codex)
                            else:
                                commands.submit(command)
                        elif command.type == "text":
                            commands.submit(command)
                        elif command.type == "start":
                            if command.capture_id is None or command.microphone_version != 2:
                                raise ValueError(
                                    "Codex voice requires identified microphone frames."
                                )
                            if (
                                commands.busy
                                or not commands.queue.empty()
                                or session.history_loading
                            ):
                                raise ValueError(
                                    "Wait for Codex history to finish loading before starting voice."
                                )
                            await session.start_voice(command.capture_id)
                        elif command.type == "mute":
                            await session.mute()
                        elif command.type == "stop":
                            commands.discard_pending()
                            await session.stop_task()
                        elif command.type == "ack":
                            session.audio.acknowledge(command.turn_id, command.samples)
                        elif command.type == "avatar":
                            session.avatar_mode = command.avatar_mode
                            session.avatar_visible = command.avatar_visible
                            session.reduced_motion = command.reduced_motion
                        elif command.type == "telemetry":
                            session.timings.record(command.metric, command.seconds, clock="native")
                            if command.metric == "playback_gap":
                                session.audio.underrun()
                        elif command.type == "pressure":
                            if command.pressure == "critical":
                                await session.end_voice()
                        elif command.type == "character":
                            store.set_setting("character", command.character_id)
                            await session.event("character", character_id=command.character_id)
                        else:
                            raise ValueError("This local speech command is not used in Codex mode.")
                    except (CodexError, ValueError, ValidationError) as error:
                        await session.event("error", message=str(error))
            except WebSocketDisconnect:
                pass
            except (OSError, RuntimeError, ValueError, TypeError, KeyError):
                try:
                    await send(
                        {
                            "type": "error",
                            "message": "Codex connection failed. Check readiness; task submission may be uncertain.",
                        }
                    )
                except (OSError, RuntimeError):
                    return  # Browser/socket is gone; finally still releases this Ari connection.
            finally:
                if expiry:
                    expiry.cancel()
                    await asyncio.gather(expiry, return_exceptions=True)
                if commands:
                    await commands.close()
                if session:
                    try:
                        await session.close()
                    finally:
                        service.active.discard(session)
                else:
                    await rpc.close()

    return service
