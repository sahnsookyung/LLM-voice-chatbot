"""Bounded bidirectional app-server transport. Never starts a second Codex host."""

import asyncio
import json
import socket
from pathlib import Path

from websockets.asyncio.client import unix_connect
from websockets.exceptions import WebSocketException

from .codex_security import CodexSecurityError, validate_socket_path, verify_peer


class CodexError(RuntimeError):
    pass


# Canonical timeline items may contain screenshots/tool artifacts even though
# Ari projects only their text. Keep a finite ceiling for those host responses.
MAX_RPC_MESSAGE = 32 * 1024 * 1024


async def connect_verified(path, *, max_size=MAX_RPC_MESSAGE):
    """Open a verified transport without sending any client identity or RPC."""
    try:
        original = validate_socket_path(path)
    except FileNotFoundError as error:
        raise CodexError("No shared Codex host. Use the Ari + Codex launcher.") from error
    except CodexSecurityError as error:
        raise CodexError(str(error)) from error
    endpoint = original[0]
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.setblocking(False)
    try:
        await asyncio.wait_for(
            asyncio.get_running_loop().sock_connect(connection, str(endpoint)), 5
        )
        pid = verify_peer(connection)
        if validate_socket_path(path) != original:
            raise CodexSecurityError("Codex socket changed during connection. Recheck readiness.")
        ws = await unix_connect(
            sock=connection,
            uri="ws://localhost/rpc",
            open_timeout=5,
            max_size=max_size,
            max_queue=2,
            ping_interval=20,
            ping_timeout=20,
        )
        return ws, pid, endpoint
    except BaseException as error:
        connection.close()
        if isinstance(error, CodexSecurityError):
            raise CodexError(str(error)) from error
        raise


class CodexRPC:
    def __init__(self, path: Path):
        self.path = path
        self.socket = None
        self.reader = None
        self.pending = {}
        self.events = asyncio.Queue(maxsize=256)
        self.serial = 0
        self.closed = False
        self.identity = {}
        self.peer_pid = None
        self.endpoint = path

    async def connect(self):
        self.socket, self.peer_pid, self.endpoint = await connect_verified(self.path)
        self.reader = asyncio.create_task(self._read())
        try:
            self.identity = await self.call(
                "initialize",
                {
                    "clientInfo": {
                        "name": "ari_companion",
                        "title": "Ari Companion",
                        "version": "0.2.0",
                    },
                    "capabilities": {"experimentalApi": True},
                },
            )
            await self.notify("initialized", {})
        except BaseException:
            await self.close()
            raise
        return self.identity

    async def _send(self, message):
        if self.closed or self.socket is None:
            raise CodexError("Codex disconnected. Refresh history before continuing.")
        try:
            await self.socket.send(json.dumps(message, ensure_ascii=False))
        except (OSError, WebSocketException) as error:
            raise CodexError(
                "Codex transport failed; acceptance may be uncertain. Reconnect before retrying."
            ) from error

    async def call(self, method, params, timeout=15):
        if len(self.pending) >= 64:
            raise CodexError("Codex requests exceeded the bounded queue.")
        self.serial += 1
        identifier = self.serial
        future = asyncio.get_running_loop().create_future()
        self.pending[identifier] = future
        try:
            await self._send({"id": identifier, "method": method, "params": params})
            return await asyncio.wait_for(future, timeout)
        except TimeoutError as error:
            # Writes may already have been accepted. Never retry them automatically.
            raise CodexError(
                "Codex did not acknowledge the request. Refresh before retrying."
            ) from error
        finally:
            self.pending.pop(identifier, None)

    async def notify(self, method, params):
        await self._send({"method": method, "params": params})

    async def respond(self, identifier, result=None, error=None):
        await self._send({"id": identifier, **({"error": error} if error else {"result": result})})

    async def _read(self):
        failure = "Codex connection lost; submission status may be unknown."
        try:
            async for raw in self.socket:
                message = json.loads(raw)
                if not isinstance(message, dict):
                    raise CodexError("Invalid Codex event.")
                if "method" not in message and "id" in message:
                    future = self.pending.get(message["id"])
                    if future is not None and not future.done():
                        if "error" in message:
                            # Provider error bodies may contain prompts or credentials.
                            code = message["error"].get("code", "unknown")
                            future.set_exception(
                                CodexError(f"Codex rejected the request (code {code}).")
                            )
                        else:
                            future.set_result(message.get("result", {}))
                else:
                    self.events.put_nowait(message)
        except asyncio.CancelledError:
            raise
        except asyncio.QueueFull:
            failure = "Codex event buffer filled. Reconnect to reload canonical history."
            self.closed = True
        except (
            OSError,
            WebSocketException,
            ValueError,
            TypeError,
            KeyError,
            CodexError,
        ):
            sent_close = getattr(getattr(self.socket, "protocol", None), "close_sent", None)
            if getattr(sent_close, "code", None) == 1009:
                failure = "A Codex history or tool response exceeded the message limit. Reconnect after updating Ari."
            self.closed = True  # All terminal failures become one sanitized disconnect event.
        finally:
            self.closed = True
            for future in list(self.pending.values()):
                if not future.done():
                    future.set_exception(CodexError(failure))
            # Overflow must terminate rather than dropping audio and continuing.
            while self.events.full():
                self.events.get_nowait()
            self.events.put_nowait({"method": "ari/disconnected", "params": {"reason": failure}})
            if self.socket:
                await self.socket.close()

    async def close(self):
        self.closed = True
        if self.socket:
            await self.socket.close()
        if self.reader and self.reader is not asyncio.current_task():
            self.reader.cancel()
            await asyncio.gather(self.reader, return_exceptions=True)
