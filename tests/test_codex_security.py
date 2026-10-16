"""Real local socket checks and launcher safety; no Codex service or account used."""

import asyncio
import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from companion import codex_security
from companion.codex_rpc import CodexError, CodexRPC
from companion.codex_security import CodexSecurityError, validate_socket_path, verify_peer
from websockets.asyncio.server import unix_serve


@pytest.fixture
def socket_dir():
    # macOS Unix socket paths cannot fit pytest's long per-test directory names.
    with tempfile.TemporaryDirectory(prefix="ari-socket-", dir="/private/tmp") as root:
        directory = Path(root) / "control"
        directory.mkdir(mode=0o700)
        yield directory


@pytest.fixture
def endpoint(socket_dir):
    directory = socket_dir
    path = directory / "host.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen()
    yield path, server
    server.close()


def test_private_socket_path_rejects_exposed_directory_and_symlinks(endpoint):
    path, _ = endpoint
    assert validate_socket_path(path)
    path.parent.chmod(0o755)
    with pytest.raises(CodexSecurityError, match="private directory"):
        validate_socket_path(path)
    path.parent.chmod(0o700)
    alias = path.parent / "alias.sock"
    alias.symlink_to(path)
    with pytest.raises(CodexSecurityError):
        validate_socket_path(alias)
    redirect = path.parent.parent / "redirect"
    redirect.symlink_to(path.parent, target_is_directory=True)
    with pytest.raises(CodexSecurityError):
        validate_socket_path(redirect / path.name)


def test_inspected_managed_link_requires_private_target_and_tracks_inode(endpoint, monkeypatch):
    path, _ = endpoint
    target = path.with_name("a" * 64)
    path.rename(target)
    directory = path.parent.parent / "app-server-control"
    directory.mkdir(mode=0o700)
    link = directory / "app-server-control.sock"
    link.symlink_to(target)
    monkeypatch.setattr(codex_security, "daemon_socket_root", lambda: target.parent)
    resolved, identity = validate_socket_path(link)
    assert resolved == target
    link.unlink()
    link.symlink_to(target)
    # A replaced inode or unchanged inode is rechecked against the target identity.
    assert validate_socket_path(link)[0] == target
    assert identity[1:] == (target.stat().st_dev, target.stat().st_ino)
    target.parent.chmod(0o755)
    with pytest.raises(CodexSecurityError):
        validate_socket_path(link)
    target.parent.chmod(0o700)


def test_real_macos_peer_credentials_and_executable(endpoint):
    if sys.platform != "darwin":
        pytest.skip("macOS peer credentials")
    path, listener = endpoint
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(path))
        peer, _ = listener.accept()
        try:
            assert verify_peer(client, Path(sys.executable)) == os.getpid()
            with pytest.raises(CodexSecurityError):
                verify_peer(client)  # A Python stub cannot impersonate the bundled Codex binary.
        finally:
            peer.close()


def test_extended_acl_cannot_bypass_private_directory_mode(endpoint):
    path, _ = endpoint
    subprocess.run(["/bin/chmod", "+a", "everyone allow read,search", str(path.parent)], check=True)
    try:
        with pytest.raises(CodexSecurityError, match="extended access grant"):
            validate_socket_path(path)
    finally:
        subprocess.run(["/bin/chmod", "-N", str(path.parent)], check=True)


def test_peer_with_wrong_uid_is_rejected(monkeypatch):
    monkeypatch.setattr(
        codex_security, "peer_identity", lambda _: (os.getuid() + 1, 1, codex_security.BUNDLED_CLI)
    )
    with pytest.raises(CodexSecurityError):
        verify_peer(None)


async def test_rejected_peer_receives_no_handshake_or_identity(endpoint):
    path, listener = endpoint
    listener.setblocking(False)
    loop = asyncio.get_running_loop()

    async def accept():
        peer, _ = await loop.sock_accept(listener)
        try:
            return await loop.sock_recv(peer, 1024)
        finally:
            peer.close()

    received = asyncio.create_task(accept())
    with pytest.raises(CodexError, match="desktop-compatible"):
        await CodexRPC(path).connect()
    assert await asyncio.wait_for(received, 1) == b""


async def test_verified_unix_transport_initializes_and_closes(socket_dir, monkeypatch):
    directory = socket_dir
    path = directory / "host.sock"
    received = []

    async def serve(ws):
        async for raw in ws:
            message = json.loads(raw)
            received.append(message)
            if "id" in message:
                await ws.send(
                    json.dumps({"id": message["id"], "result": {"codexHome": str(directory)}})
                )

    monkeypatch.setattr(
        "companion.codex_rpc.verify_peer",
        lambda connection: verify_peer(connection, Path(sys.executable)),
    )
    async with unix_serve(serve, str(path)):
        rpc = CodexRPC(path)
        try:
            assert (await rpc.connect())["codexHome"] == str(directory)
            assert rpc.peer_pid == os.getpid()
            await rpc.call("thread/list", {})
        finally:
            await rpc.close()
        assert rpc.reader.done()
    assert received[0]["params"]["clientInfo"]["name"] == "ari_companion"


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/launch_codex.py"
    spec = importlib.util.spec_from_file_location("ari_launcher_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.commands = []
    return module


@pytest.mark.parametrize(
    "case,expected", [("name", True), ("path", True), ("absent", False), ("denied", None)]
)
async def test_launcher_fails_closed_when_process_inspection_is_denied(
    launcher, monkeypatch, case, expected
):
    process = SimpleNamespace(
        info={
            "name": "ChatGPT" if case == "name" else "renamed",
            "exe": str(launcher.DESKTOP / "Contents/MacOS/ChatGPT") if case == "path" else "/other",
        }
    )

    def processes(*_):
        if case == "denied":
            raise launcher.psutil.AccessDenied()
        return [process]

    monkeypatch.setattr(launcher.psutil, "process_iter", processes)
    if expected is None:
        with pytest.raises(RuntimeError, match="Refusing"):
            await launcher.is_running("ChatGPT")
    else:
        assert await launcher.is_running("ChatGPT") is expected
    assert launcher.commands == []


async def test_unknown_desktop_stops_before_process_or_daemon_actions(launcher, monkeypatch):
    async def fingerprint():
        return {"desktop_sha256": "unknown"}

    async def forbidden(*_):
        pytest.fail("Incompatible launch must stop before inspecting or starting clients")

    monkeypatch.setattr(sys, "argv", ["launch_codex.py", "--validation-thread", "test-task"])
    monkeypatch.setattr(launcher, "desktop_fingerprint", fingerprint)
    monkeypatch.setattr(launcher, "is_running", forbidden)
    monkeypatch.setattr(launcher, "prepare_bridge", forbidden)
    with pytest.raises(SystemExit, match="has not been checked"):
        await launcher.main()
    assert launcher.commands == []


async def test_probe_distinguishes_restricted_inspection_from_missing_host(monkeypatch, tmp_path):
    from companion import codex_compat

    async def fingerprint():
        return {"desktop_sha256": "unknown"}

    class RestrictedRPC:
        def __init__(self, _):
            pass

        async def connect(self):
            raise PermissionError("sandbox")

        async def close(self):
            pass

    monkeypatch.setattr(codex_compat, "desktop_fingerprint", fingerprint)
    monkeypatch.setattr(codex_compat, "CodexRPC", RestrictedRPC)
    result = await codex_compat.probe(tmp_path / "no-acceptance.json")
    assert not result["available"] and not result["shared_host"]
    assert result["launcher"]["available"] is False
    assert any("denied local socket/process inspection" in reason for reason in result["reasons"])
    assert not any("Finish desktop work" in reason for reason in result["reasons"])


@pytest.mark.parametrize(
    "desktop_hash",
    [
        "75e397e2280dde93b70a85aa926f4282c728eacc84dfeb15ba7fb6ddd84628af",
        "2f231a64e6dd30111bcd09e03c149cd06c8679b25914842772423251c4608ded",
    ],
)
async def test_launcher_preserves_desktop_configuration_path_and_waits_for_attachment(
    launcher, monkeypatch, desktop_hash
):
    async def fingerprint():
        return {
            "desktop_sha256": desktop_hash,
            "schema_sha256": launcher.INSPECTED_SCHEMA,
        }

    async def not_running(_):
        return False

    async def start(*args, **kwargs):
        launcher.commands.append((args, kwargs))

    async def ready(_):
        assert len(launcher.commands) == 1  # desktop is started first; never a separate daemon
        return {"shared_host": True}

    async def output(*args):
        launcher.commands.append((args, {}))
        return ""

    async def no_delay(_):
        pass

    monkeypatch.setattr(sys, "argv", ["launch_codex.py", "--validation-thread", "test-task"])
    monkeypatch.setattr(launcher, "desktop_fingerprint", fingerprint)
    monkeypatch.setattr(launcher, "is_running", not_running)
    monkeypatch.setattr(launcher, "prepare_bridge", lambda: Path("/private/test-relay"))
    monkeypatch.setattr(launcher.asyncio, "create_subprocess_exec", start)
    monkeypatch.setattr(launcher.asyncio, "sleep", no_delay)
    monkeypatch.setattr(launcher, "probe", ready)
    monkeypatch.setattr(launcher, "command_output", output)
    monkeypatch.setenv("DESKTOP_SETTING", "preserve")
    monkeypatch.setenv("CODEX_APP_SERVER_WS_URL", "not-this-route")
    monkeypatch.setenv("CODEX_APP_SERVER_USE_LOCAL_DAEMON", "1")
    await launcher.main()
    assert len(launcher.commands) == 2
    desktop, settings = launcher.commands[0]
    assert desktop == (str(launcher.DESKTOP / "Contents/MacOS/ChatGPT"),)
    assert settings["env"]["DESKTOP_SETTING"] == "preserve"
    assert settings["env"]["CODEX_CLI_PATH"] == "/private/test-relay"
    assert "CODEX_APP_SERVER_WS_URL" not in settings["env"]
    assert "CODEX_APP_SERVER_USE_LOCAL_DAEMON" not in settings["env"]
    assert launcher.commands[1][0][-3:] == ("--args", "--codex-validation-thread", "test-task")


async def test_canonical_history_response_larger_than_two_megabytes_stays_connected(
    socket_dir, monkeypatch
):
    path = socket_dir / "history.sock"
    payload = "x" * (3 * 1024 * 1024)

    async def serve(ws):
        async for raw in ws:
            message = json.loads(raw)
            if "id" in message:
                result = (
                    {"codexHome": str(socket_dir)}
                    if message["method"] == "initialize"
                    else {"artifact": payload}
                )
                await ws.send(json.dumps({"id": message["id"], "result": result}))

    monkeypatch.setattr(
        "companion.codex_rpc.verify_peer", lambda c: verify_peer(c, Path(sys.executable))
    )
    async with unix_serve(serve, str(path)):
        rpc = CodexRPC(path)
        try:
            await rpc.connect()
            assert (await rpc.call("thread/timeline/list", {}))["artifact"] == payload
            assert not rpc.closed
        finally:
            await rpc.close()


@pytest.mark.parametrize(
    "same_hash, kernel_result, expected",
    [
        (True, 0, True),
        (False, 0, False),
        (True, -1, False),
    ],
)
def test_relocated_host_requires_exact_kernel_code_identity(
    tmp_path, monkeypatch, same_hash, kernel_result, expected
):
    import ctypes
    from types import SimpleNamespace

    from companion import codex_security as security

    binary = tmp_path / "host"
    binary.write_bytes(b"fixture")
    monkeypatch.setattr(security, "BUNDLED_CLI", binary)
    monkeypatch.setattr(security, "_installed_code_hash", lambda identity: bytes([1]) * 20)

    class CSOps:
        def __call__(self, pid, operation, buffer, size):
            assert pid == 123 and operation == 5 and size == 20
            ctypes.memmove(buffer, bytes([1 if same_hash else 2]) * 20, 20)
            return kernel_result

    monkeypatch.setattr(security.ctypes, "CDLL", lambda *a, **k: SimpleNamespace(csops=CSOps()))
    assert security.relocated_host_matches(123) is expected


def test_relocated_host_does_not_trust_unverified_installed_code(monkeypatch):
    import subprocess

    from companion import codex_security as security

    security._installed_code_hash.cache_clear()
    calls = []

    def reject(args, **kwargs):
        calls.append(args)
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(security.subprocess, "run", reject)
    assert security._installed_code_hash(("fixture",)) is None
    assert len(calls) == 1 and "--verify" in calls[0] and "--strict" in calls[0]
    security._installed_code_hash.cache_clear()
