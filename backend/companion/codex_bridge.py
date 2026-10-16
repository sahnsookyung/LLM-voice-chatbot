"""Transparent desktop stdio relay to the desktop-configured shared Codex host.

The desktop supplies its own initialize, configuration and environment. Nothing
is impersonated or logged. Other CLI invocations exec the bundled binary normally.
"""

import asyncio
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import psutil
from websockets.exceptions import WebSocketException

from .codex_rpc import CodexError, connect_verified
from .codex_security import BUNDLED_CLI, validate_socket_directory

ROOT = Path(__file__).resolve().parents[2]
BRIDGE_SCRIPT = ROOT / "scripts/codex_desktop_bridge.py"
MAX_MESSAGE = 16 * 1024 * 1024
PLUGIN_KEYS = {
    "plugins.codex-app-tools@openai-bundled.mcp_servers.codex_app.enabled",
    "plugins.code-review@openai-bundled.mcp_servers.code-review.enabled",
}


def bridge_directory():
    return ROOT / ".runtime/codex/desktop-host"


def bridge_socket(directory=None):
    return (directory or bridge_directory()) / "app-server-control/app-server-control.sock"


def is_desktop_server(arguments):
    """Only the pinned desktop's main server; auth/schema/CLI probes pass through."""
    keys, plain = set(), []
    index = 0
    while index < len(arguments):
        arg = arguments[index]
        if arg in {"-c", "--config"}:
            if index + 1 >= len(arguments):
                return False
            keys.add(arguments[index + 1].split("=", 1)[0])
            index += 2
        elif arg.startswith("--config="):
            keys.add(arg[len("--config=") :].split("=", 1)[0])
            index += 1
        else:
            plain.append(arg)
            index += 1
    return plain == ["app-server", "--analytics-default-enabled"] and PLUGIN_KEYS <= keys


def host_environment(source):
    env = dict(source)
    # This points child CLI invocations at the actual binary, not back at the relay.
    env["CODEX_CLI_PATH"] = str(BUNDLED_CLI)
    env.pop("CODEX_APP_SERVER_USE_LOCAL_DAEMON", None)
    env.pop("CODEX_APP_SERVER_WS_URL", None)
    return env


def launch_digest(arguments, env):
    # Values remain in process memory. Store only a digest to detect changed
    # runtime settings/auth environment on reconnect, never the values themselves.
    stable_env = {k: v for k, v in env.items() if k not in {"_", "SHLVL", "PWD", "OLDPWD"}}
    payload = json.dumps([arguments, sorted(stable_env.items())], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def read_state(directory):
    path = directory / "host.json"
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd) as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise CodexError("Shared host record has unsafe permissions.")
        state = json.loads(stream.read(16384))
        if not isinstance(state, dict):
            raise CodexError("Shared host record is invalid.")
        return state


def write_state(directory, state):
    fd, name = tempfile.mkstemp(prefix="host-", dir=directory)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(state, stream)
        os.replace(name, directory / "host.json")
    finally:
        if os.path.exists(name):
            os.unlink(name)


def process_matches(pid, created, executable):
    try:
        process = psutil.Process(pid)
        return (
            process.uids().real == os.getuid()
            and process.create_time() == created
            and Path(process.exe()).resolve() == Path(executable).resolve()
        )
    except (psutil.Error, OSError, TypeError, ValueError):
        return False


async def ensure_host(arguments, source_env, directory):
    path = bridge_socket(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.mkdir(mode=0o700, exist_ok=True)
    validate_socket_directory(directory / "host.json")
    validate_socket_directory(path)
    lock = os.open(directory / "host.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    process = None
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CodexError("Shared host startup is already in progress.") from error
        env = host_environment(source_env)
        digest = launch_digest(arguments, env)
        if os.path.lexists(path):
            state = read_state(directory)
            if state.get("configuration_sha256") != digest:
                raise CodexError(
                    "Existing shared host has different desktop runtime settings. "
                    "It was left running; reconcile its settings before reconnecting."
                )
            if not process_matches(state.get("host_pid"), state.get("host_created"), BUNDLED_CLI):
                raise CodexError(
                    "Shared host identity changed. Existing endpoint was left untouched."
                )
        else:
            if (directory / "host.json").exists():
                old = read_state(directory)
                if process_matches(old.get("host_pid"), old.get("host_created"), BUNDLED_CLI):
                    raise CodexError(
                        "The existing host is still starting. No second host was created."
                    )
            # A detached host must outlive this relay's asyncio loop. Asyncio's
            # subprocess transport can kill a child when the transport closes.
            # Spawn synchronously so cancellation cannot orphan an unrecorded
            # child; only process creation blocks, never host readiness or work.
            process = subprocess.Popen(  # noqa: ASYNC220
                [str(BUNDLED_CLI), *arguments, "--listen", "unix://" + str(path)],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            state = {
                "host_pid": process.pid,
                "host_created": psutil.Process(process.pid).create_time(),
                "configuration_sha256": digest,
                "initialized": False,
            }
            write_state(directory, state)
            for _ in range(100):
                if path.exists() or process.poll() is not None:
                    break
                await asyncio.sleep(0.05)
        ws, pid, endpoint = await connect_verified(path, max_size=MAX_MESSAGE)
        if pid != state["host_pid"]:
            await ws.close()
            raise CodexError("Socket and configured host identities differ.")
        return ws, state, endpoint
    except BaseException:
        # Only our newly spawned, not-yet-exposed startup may be stopped. Never an
        # existing host or one that has received desktop/task protocol messages.
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                await asyncio.to_thread(process.wait, timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait)
        raise
    finally:
        os.close(lock)


async def relay(reader, writer, ws, initialized=None):
    """One ordered, backpressured stream in each direction; no retries or rewrites."""
    initialize_id = None

    async def upstream():
        nonlocal initialize_id
        while line := await reader.readline():
            if len(line) > MAX_MESSAGE or not line.endswith(b"\n"):
                raise CodexError("Desktop protocol frame is too large or incomplete.")
            raw = line[:-1].decode("utf-8")
            message = json.loads(raw)
            if not isinstance(message, dict):
                raise CodexError("Invalid protocol object.")
            if message.get("method") == "initialize":
                initialize_id = message.get("id")
            await ws.send(raw)

    async def downstream():
        async for raw in ws:
            if not isinstance(raw, str) or len(raw.encode()) > MAX_MESSAGE:
                raise CodexError("Shared host returned an invalid frame.")
            message = json.loads(raw)
            if not isinstance(message, dict):
                raise CodexError("Invalid protocol object.")
            writer.write(raw.encode() + b"\n")
            await writer.drain()
            if (
                initialize_id is not None
                and message.get("id") == initialize_id
                and "result" in message
                and initialized
            ):
                initialized()

    tasks = [asyncio.create_task(upstream()), asyncio.create_task(downstream())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await ws.close()


async def run_desktop_bridge(arguments):
    directory = bridge_directory()
    ws, state, _ = await ensure_host(arguments, os.environ, directory)
    input_transport = output_transport = None
    try:
        parent = psutil.Process(os.getppid())
        state.update(
            {
                "bridge_pid": os.getpid(),
                "bridge_created": psutil.Process().create_time(),
                "desktop_pid": parent.pid,
                "desktop_created": parent.create_time(),
                "initialized": False,
            }
        )
        write_state(directory, state)
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader(limit=MAX_MESSAGE + 1)
        input_transport, _ = await loop.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader), sys.stdin.buffer
        )
        output_transport, protocol = await loop.connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout.buffer
        )
        writer = asyncio.StreamWriter(output_transport, protocol, None, loop)

        def initialized():
            state["initialized"] = True
            write_state(directory, state)

        await relay(reader, writer, ws, initialized)
    finally:
        if input_transport is not None:
            input_transport.close()
        if output_transport is not None:
            output_transport.close()
        await ws.close()
        # Host/task lifetime is independent of this desktop connection.


def main():
    arguments = sys.argv[1:]
    if not is_desktop_server(arguments):
        os.execve(str(BUNDLED_CLI), [str(BUNDLED_CLI), *arguments], host_environment(os.environ))
    try:
        asyncio.run(run_desktop_bridge(arguments))
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError, ValueError, TypeError, psutil.Error, WebSocketException):
        # Never echo argv, environment, provider messages or credentials.
        print(
            "Ari desktop relay failed. Check shared-host readiness; no request was retried.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
