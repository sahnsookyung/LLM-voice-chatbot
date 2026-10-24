"""Read-only capability evidence, separate from successful live acceptance tests."""

import asyncio
import hashlib
import json
import os
import tempfile
from pathlib import Path

import psutil
from websockets.exceptions import WebSocketException

from .codex_bridge import (
    BRIDGE_SCRIPT,
    bridge_directory,
    bridge_socket,
    process_matches,
    read_state,
)
from .codex_rpc import CodexError, CodexRPC
from .codex_security import BUNDLED_CLI

DESKTOP = Path("/Applications/ChatGPT.app")
INSPECTED_DESKTOPS = {
    "75e397e2280dde93b70a85aa926f4282c728eacc84dfeb15ba7fb6ddd84628af": "59c1c2f4c78bf368f38d7ae8372d6d9da94b92c0c56ddf4e32278f4ed5cd803e",
    "2f231a64e6dd30111bcd09e03c149cd06c8679b25914842772423251c4608ded": "59c1c2f4c78bf368f38d7ae8372d6d9da94b92c0c56ddf4e32278f4ed5cd803e",
    "bf9c41fa9770fbfacfba93042ff71c24cb65c82935819377275faac08103383f": "8c745d3a424119540f488fc94a05bd0cdd4963cefa2d0d65e276835f1591b0a3",
}


def launcher_blocker(fingerprint):
    schema = INSPECTED_DESKTOPS.get(fingerprint.get("desktop_sha256"))
    if schema is None:
        return (
            "This desktop build has not been checked with Ari's configuration-preserving relay. "
            "Use normal desktop launch until compatibility is verified."
        )
    if fingerprint.get("schema_sha256") != schema:
        return (
            "This desktop and host schema combination has not passed compatibility inspection. "
            "Use normal desktop launch until the installed pair is verified."
        )
    return None


REQUIRED_SCHEMAS = (
    "v1/InitializeParams.json",
    "v2/ThreadRealtimeStartParams.json",
    "v2/ThreadRealtimeOutputAudioDeltaNotification.json",
    "v2/ThreadTimelineListResponse.json",
    "v2/ThreadSettingsUpdateParams.json",
    "ServerRequest.json",
)
ACCEPTANCE_GATES = (
    "transport_access",
    "desktop_live_history",
    "voice_history_roundtrip",
    "computer_use",
    "voice_ownership",
    "approval_routing",
)


def client_fingerprint():
    root = Path(__file__).resolve().parents[2]
    files = list((root / "backend/companion").glob("*.py"))
    files += list((root / "native/Sources/Companion").glob("*.swift"))
    files += [root / "pyproject.toml", root / "uv.lock"]
    files += [
        root / "scripts/launch_codex.py",
        root / "scripts/probe_codex_voice.py",
        BRIDGE_SCRIPT,
    ]
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def codex_home():
    return Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")).resolve()


def host_id():
    # Identity follows the user's Codex home, not a daemon PID that changes on restart.
    return "codex-local-" + hashlib.sha256(str(codex_home()).encode()).hexdigest()[:20]


def socket_path():
    return bridge_socket()


async def command_output(*args, timeout=15):
    process = await asyncio.create_subprocess_exec(
        *map(str, args), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
    )
    try:
        out, _ = await asyncio.wait_for(process.communicate(), timeout)
    except BaseException:
        process.kill()
        await process.wait()
        raise
    if process.returncode:
        raise RuntimeError("Compatibility command failed.")
    return out.decode()


async def desktop_fingerprint():
    version = (await command_output(BUNDLED_CLI, "--version")).strip()
    with tempfile.TemporaryDirectory(prefix="ari-codex-schema-") as directory:
        await command_output(
            BUNDLED_CLI, "app-server", "generate-json-schema", "--experimental", "--out", directory
        )
        digest = hashlib.sha256()
        for name in REQUIRED_SCHEMAS:
            digest.update(name.encode())
            digest.update((Path(directory) / name).read_bytes())
    # Include desktop application code: its shared-daemon launch behavior can change
    # independently of the host protocol. Stream to avoid retaining the bundle in RAM.
    app_digest = hashlib.sha256()
    with (DESKTOP / "Contents/Resources/app.asar").open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            app_digest.update(block)
    return {
        "host_version": version,
        "schema_sha256": digest.hexdigest(),
        "desktop_sha256": app_digest.hexdigest(),
        "client_sha256": client_fingerprint(),
    }


def socket_descriptors(output):
    descriptors = []
    process = command = None
    descriptor = None
    for line in output.splitlines():
        if not line:
            continue
        field, value = line[0], line[1:]
        if field == "p":
            process, command, descriptor = value, None, None
        elif field == "c":
            command = value
        elif field == "f":
            descriptor = {"pid": process, "command": command}
            descriptors.append(descriptor)
        elif descriptor is not None and field in {"d", "D", "n"}:
            descriptor[field] = value
    return descriptors


def socket_client_pids(output, path, server_pid):
    """Find clients by the verified server's actual Unix PCB addresses."""
    descriptors = socket_descriptors(output)
    owns = any(d["pid"] == str(server_pid) and d.get("n") == str(path) for d in descriptors)
    if not owns:
        raise RuntimeError("Cannot verify listener descriptors.")
    addresses = {
        d.get("d", d.get("D", "")).lower() for d in descriptors if d["pid"] == str(server_pid)
    }
    return {
        int(d["pid"])
        for d in descriptors
        if d["pid"] != str(server_pid)
        and d.get("n", "").startswith("->0x")
        and d["n"][2:].lower() in addresses
    }


def desktop_socket_attached(output, path, server_pid=None, *, client_pids=None):
    """macOS clients often show a peer address, rather than the bound socket path."""
    descriptors = socket_descriptors(output)
    owners = {
        d["pid"]
        for d in descriptors
        if d.get("n") == str(path) and (server_pid is None or d["pid"] == str(server_pid))
    }
    # Darwin lsof exposes Unix PCB addresses as device-character field `d`;
    # uppercase `D` is a device number and is absent for these sockets.
    addresses = {d.get("d", d.get("D", "")).lower() for d in descriptors if d["pid"] in owners}
    clients = {str(pid) for pid in client_pids} if client_pids is not None else None
    for descriptor in descriptors:
        matches = (
            descriptor["pid"] in clients
            if clients is not None
            else descriptor["command"] in {"ChatGPT", "Codex"}
        )
        if not matches:
            continue
        name = descriptor.get("n", "")
        if name == str(path) or (name.startswith("->0x") and name[2:].lower() in addresses):
            return True
    return False


async def desktop_socket_evidence(path, server_pid):
    """Require an actual desktop socket descriptor, not an environment flag."""
    try:
        output = await command_output("/usr/sbin/lsof", "-n", "-P", "-U", "-FpcfdDn", timeout=5)
    except (OSError, RuntimeError, TimeoutError):
        return False
    if desktop_socket_attached(output, path, server_pid):
        return True
    # The relay is a verified desktop child forwarding desktop's own stdio RPC.
    # Evidence requires the live parent, bridge executable/script, initialized
    # connection and an actual socket to the exact verified host PID.
    try:
        state = read_state(bridge_directory())
        if state.get("host_pid") != server_pid or not state.get("initialized"):
            return False
        if not process_matches(server_pid, state["host_created"], BUNDLED_CLI):
            return False
        if not process_matches(
            state["desktop_pid"], state["desktop_created"], DESKTOP / "Contents/MacOS/ChatGPT"
        ):
            return False
        python = Path(__file__).resolve().parents[2] / ".venv/bin/python"
        if not process_matches(state["bridge_pid"], state["bridge_created"], python):
            return False
        bridge = psutil.Process(state["bridge_pid"])
        if bridge.ppid() != state["desktop_pid"] or str(BRIDGE_SCRIPT) not in bridge.cmdline():
            return False
        return desktop_socket_attached(output, path, server_pid, client_pids={state["bridge_pid"]})
    except (OSError, ValueError, TypeError, KeyError, psutil.Error):
        return False


def accepted_report(path, fingerprint):
    try:
        value = json.loads(path.read_text())
        return value.get("fingerprint") == fingerprint and all(
            value.get("checks", {}).get(key, {}).get("passed") is True
            and bool(value["checks"][key].get("evidence"))
            for key in ACCEPTANCE_GATES
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


async def probe(report_path):
    result = {
        "available": False,
        "shared_host": False,
        "host_id": host_id(),
        "checks": {},
        "reasons": [],
    }
    try:
        fingerprint = await desktop_fingerprint()
        result["fingerprint"] = fingerprint
        blocked = launcher_blocker(fingerprint)
        result["launcher"] = {"available": blocked is None, "reason": blocked}
    except (OSError, RuntimeError, TimeoutError):
        result["reasons"].append("The desktop-compatible Codex host or required schema is missing.")
        return result
    rpc = CodexRPC(socket_path())
    try:
        identity = await rpc.connect()
        same_home = Path(identity.get("codexHome", "")).resolve() == codex_home()
        result["checks"]["same_home"] = same_home
        result["checks"]["verified_peer"] = rpc.peer_pid is not None
        result["checks"]["desktop_socket"] = await desktop_socket_evidence(
            rpc.endpoint, rpc.peer_pid
        )
        actual_version = identity.get("userAgent", "").partition("/")[2].split(" ", 1)[0]
        result["checks"]["host_version_matches"] = actual_version == fingerprint[
            "host_version"
        ].removeprefix("codex-cli ")
        result["shared_host"] = all(result["checks"].values())
        result["checks"]["live_acceptance"] = accepted_report(report_path, fingerprint)
        # These reads never resume a task, generate speech, or open the microphone.
        for name, method, params in [
            ("threads", "thread/list", {"limit": 1}),
            ("voices", "thread/realtime/listVoices", {}),
            ("account", "account/read", {"refreshToken": False}),
        ]:
            try:
                value = await rpc.call(method, params, timeout=5)
                result["checks"][name] = bool(value.get("account")) if name == "account" else True
            except (OSError, RuntimeError, TimeoutError, ValueError, WebSocketException):
                result["checks"][name] = False
        result["available"] = result["shared_host"] and all(result["checks"].values())
        if not result["shared_host"]:
            result["reasons"].append(
                "Desktop attachment to this exact shared host is not verified."
            )
        if not result["checks"]["live_acceptance"]:
            result["reasons"].append(
                "Voice/history, Computer Use, ownership, and approval acceptance checks have not passed for this build."
            )
        if not all(result["checks"].get(key) for key in ("threads", "voices", "account")):
            result["reasons"].append(
                "The shared host did not provide all required account, history, and voice interfaces."
            )
    except CodexError as error:
        result["reasons"].append(str(error))
    except PermissionError:
        result["reasons"].append(
            "This environment denied local socket/process inspection. "
            "Run the read-only check from a normal Terminal; this is not evidence of a missing host."
        )
    except (OSError, RuntimeError, TimeoutError, ValueError, WebSocketException):
        result["reasons"].append(
            "No verified shared Codex host. Finish desktop work, quit the desktop app, then use scripts/launch_codex.py."
        )
    finally:
        await rpc.close()
    if not result["shared_host"] and blocked:
        result["reasons"].insert(0, blocked)
    return result
