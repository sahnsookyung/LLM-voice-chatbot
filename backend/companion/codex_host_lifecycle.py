"""Explicit launcher-only refresh; never called when an Ari window closes."""

import asyncio
import fcntl
import os

import psutil

from .codex_bridge import bridge_socket, process_matches, read_state
from .codex_compat import command_output, socket_client_pids
from .codex_rpc import CodexError, CodexRPC
from .codex_security import BUNDLED_CLI, validate_socket_directory, validate_socket_path


async def require_idle(rpc):
    """Read canonical state without resuming tasks or executing anything."""
    loaded = await rpc.call("thread/loaded/list", {})
    if not isinstance(loaded.get("data"), list) or loaded.get("nextCursor"):
        raise CodexError("Cannot verify every loaded task. Host was left running.")
    for identifier in loaded["data"]:
        result = await rpc.call("thread/read", {"threadId": identifier, "includeTurns": False})
        if result.get("thread", {}).get("status", {}).get("type") not in {"idle", "notLoaded"}:
            raise CodexError("A task is active or its status is uncertain. Host was left running.")
        for method in ("thread/queue/list", "thread/backgroundTerminals/list"):
            value = await rpc.call(method, {"threadId": identifier})
            if value.get("data") != [] or value.get("nextCursor"):
                raise CodexError(
                    "Queued work or background terminals remain. Host was left running."
                )


async def restart_idle_host(directory):
    """Retire only our verified quiescent host, at the user's explicit launch request.

    Desktop creates new tool pipe/verification identities on every launch. Retiring
    an unused host avoids retaining those stale identities. Untracked hosts and
    hosts with another connected client or pending work are never stopped.
    """
    path = bridge_socket(directory)
    if not directory.exists():
        return False
    validate_socket_directory(directory / "host.json")
    if not os.path.lexists(path):
        return False
    lock = os.open(directory / "host.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    rpc = CodexRPC(path)
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CodexError("Host startup is in progress. Nothing was stopped.") from error
        state = read_state(directory)
        pid, created = state.get("host_pid"), state.get("host_created")
        if not process_matches(pid, created, BUNDLED_CLI):
            raise CodexError("Cannot verify ownership of the existing host. Nothing was stopped.")
        await rpc.connect()
        if rpc.peer_pid != pid:
            raise CodexError("Host/socket identities differ. Nothing was stopped.")

        async def require_only_this_client():
            output = await command_output("/usr/sbin/lsof", "-n", "-P", "-U", "-FpcfdDn", timeout=5)
            if socket_client_pids(output, rpc.endpoint, pid) != {os.getpid()}:
                raise CodexError(
                    "Another client is attached. Quit both clients before refreshing the host."
                )

        async with asyncio.timeout(30):
            await require_only_this_client()
            await require_idle(rpc)
            await require_only_this_client()
            if not process_matches(pid, created, BUNDLED_CLI):
                raise CodexError("Host identity changed. Nothing was stopped.")
            endpoint, identity = validate_socket_path(path)
            # No force-kill or process-group kill: only the verified, idle host.
            psutil.Process(pid).terminate()
            await rpc.close()
            for _ in range(100):
                if not process_matches(pid, created, BUNDLED_CLI):
                    break
                await asyncio.sleep(0.05)
            else:
                raise CodexError("Idle host has not exited. No replacement was started.")
        for candidate, expected in ((path, identity[0]), (endpoint, identity[1:])):
            try:
                info = candidate.lstat()
                if (info.st_dev, info.st_ino) == expected:
                    candidate.unlink()
            except FileNotFoundError:
                pass
        return True
    finally:
        await rpc.close()
        os.close(lock)
