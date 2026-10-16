"""Check real desktop bootstrap in an empty, isolated profile; never sign in.

Does not stop existing clients, copy credentials, or submit a model/task request.
Only the temporary desktop and host created by this probe are stopped afterward.
"""

import asyncio
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from companion import codex_compat as compat
from companion.codex_bridge import bridge_socket, process_matches, read_state
from companion.codex_host_lifecycle import restart_idle_host
from companion.codex_rpc import CodexRPC
from companion.codex_security import BUNDLED_CLI


async def stop_owned(pid, created, executable, preserve=()):
    if not process_matches(pid, created, executable):
        return
    parent = psutil.Process(pid)
    children = [
        (p, p.create_time()) for p in parent.children(recursive=True) if p.pid not in preserve
    ]
    parent.terminate()
    for _ in range(100):
        if not process_matches(pid, created, executable):
            break
        await asyncio.sleep(0.05)
    if process_matches(pid, created, executable):
        parent.kill()
    for child, started in children:
        try:
            if child.create_time() == started and child.is_running():
                child.terminate()
        except psutil.NoSuchProcess:
            pass


async def run():
    fingerprint = await compat.desktop_fingerprint()
    desktop_exe = compat.DESKTOP / "Contents/MacOS/ChatGPT"
    existing = [
        (p.pid, p.create_time())
        for p in psutil.process_iter(["exe"])
        if p.info["exe"] == str(desktop_exe)
    ]
    with tempfile.TemporaryDirectory(prefix="ari-desktop-relay-", dir="/private/tmp") as folder:
        root = Path(folder)
        home, profile, directory = root / "home", root / "profile", root / "relay"
        for path in (home, profile, directory):
            path.mkdir(mode=0o700)
        entrypoint = root / "probe_bridge.py"
        entrypoint.write_text(
            "import sys, os, json, hashlib\nfrom pathlib import Path\n"
            f"sys.path.insert(0, {str(ROOT / 'backend')!r})\n"
            "from companion import codex_bridge\n"
            f"codex_bridge.bridge_directory = lambda: Path({str(directory)!r})\n"
            "if codex_bridge.is_desktop_server(sys.argv[1:]):\n"
            f"    p = Path({str(root / 'environment-hashes.json')!r})\n"
            "    previous = json.loads(p.read_text()) if p.exists() else []\n"
            "    current = {k: hashlib.sha256(v.encode()).hexdigest() for k,v in os.environ.items()}\n"
            "    previous.append({'env':current, 'args': hashlib.sha256(json.dumps(sys.argv[1:]).encode()).hexdigest()})\n"
            "    p.write_text(json.dumps(previous))\n"
            "codex_bridge.main()\n"
        )
        shim = root / "codex-relay"
        shim.write_text(
            "#!/bin/sh\nexec "
            + shlex.quote(str(ROOT / ".venv/bin/python"))
            + " "
            + shlex.quote(str(entrypoint))
            + ' "$@"\n'
        )
        shim.chmod(0o700)
        env = dict(os.environ)
        env.update(
            CODEX_HOME=str(home),
            CODEX_ELECTRON_USER_DATA_PATH=str(profile),
            CODEX_CLI_PATH=str(shim),
            CODEX_APP_SERVER_FORCE_CLI="1",
        )
        env.pop("CODEX_APP_SERVER_USE_LOCAL_DAEMON", None)
        env.pop("CODEX_APP_SERVER_WS_URL", None)
        process = await asyncio.create_subprocess_exec(
            str(desktop_exe),
            "--user-data-dir=" + str(profile),
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        created = psutil.Process(process.pid).create_time()
        state = {}
        rpc = CodexRPC(bridge_socket(directory))
        try:
            for _ in range(80):
                try:
                    state = read_state(directory)
                    if state.get("initialized"):
                        break
                except FileNotFoundError:
                    pass
                if process.returncode is not None:
                    break
                await asyncio.sleep(0.5)
            if not state.get("initialized"):
                return {
                    "passed": False,
                    "reason": "Isolated desktop did not initialize the relay.",
                    "desktop_exited": process.returncode is not None,
                    "relay_record_created": bool(state),
                }
            identity = await rpc.connect()
            # Use the production verifier with the isolated probe's exact paths.
            compat.bridge_directory = lambda: directory
            compat.BRIDGE_SCRIPT = entrypoint
            desktop_args = psutil.Process(state["bridge_pid"]).cmdline()[2:]
            host_args = psutil.Process(state["host_pid"]).cmdline()[1:]
            checks = {
                "real_desktop_initialize": state["initialized"],
                "desktop_process_is_probe": state["desktop_pid"] == process.pid,
                "desktop_socket_attachment": await compat.desktop_socket_evidence(
                    rpc.endpoint, rpc.peer_pid
                ),
                "original_runtime_arguments_preserved": host_args[:-2] == desktop_args,
                "only_unix_listener_added": host_args[-2:]
                == ["--listen", "unix://" + str(bridge_socket(directory))],
                "ari_same_host": rpc.peer_pid == state["host_pid"],
                "isolated_home": Path(identity["codexHome"]).resolve() == home,
                "existing_desktops_untouched": all(
                    process_matches(pid, started, desktop_exe) for pid, started in existing
                ),
            }
            await stop_owned(process.pid, created, desktop_exe, preserve={state["host_pid"]})
            await process.wait()
            checks["desktop_close_preserves_host"] = (await rpc.call("thread/loaded/list", {}))[
                "data"
            ] == []
            first_host = state["host_pid"]
            first_bridge = state["bridge_pid"]
            process = await asyncio.create_subprocess_exec(
                str(desktop_exe),
                "--user-data-dir=" + str(profile),
                env=env,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
            created = psutil.Process(process.pid).create_time()
            for _ in range(20):
                state = read_state(directory)
                if state.get("bridge_pid") != first_bridge and state.get("initialized"):
                    break
                if process.returncode is not None:
                    break
                await asyncio.sleep(0.5)
            checks["restart_rejects_stale_tool_identities"] = (
                state["host_pid"] == first_host
                and state.get("desktop_pid") != process.pid
                and not await compat.desktop_socket_evidence(rpc.endpoint, rpc.peer_pid)
            )
            launches = json.loads((root / "environment-hashes.json").read_text())
            restart_diagnostics = {
                "main_server_launches": len(launches),
                "runtime_arguments_changed": len(launches) > 1
                and launches[0]["args"] != launches[-1]["args"],
                "changed_environment_keys": sorted(
                    key
                    for key in launches[0]["env"].keys() | launches[-1]["env"].keys()
                    if launches[0]["env"].get(key) != launches[-1]["env"].get(key)
                ),
                "restarted_desktop_exited": process.returncode is not None,
            }
            await stop_owned(process.pid, created, desktop_exe, preserve={state["host_pid"]})
            await process.wait()
            await rpc.close()
            checks["explicit_idle_refresh"] = await restart_idle_host(directory)
            process = await asyncio.create_subprocess_exec(
                str(desktop_exe),
                "--user-data-dir=" + str(profile),
                env=env,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
            created = psutil.Process(process.pid).create_time()
            for _ in range(60):
                state = read_state(directory)
                if state.get("desktop_pid") == process.pid and state.get("initialized"):
                    break
                if process.returncode is not None:
                    break
                await asyncio.sleep(0.5)
            rpc = CodexRPC(bridge_socket(directory))
            await rpc.connect()
            checks["desktop_restart_after_explicit_refresh"] = (
                state["host_pid"] != first_host
                and state.get("desktop_pid") == process.pid
                and await compat.desktop_socket_evidence(rpc.endpoint, rpc.peer_pid)
            )
            return {
                "passed": all(checks.values()),
                "checks": checks,
                "restart_diagnostics": restart_diagnostics,
                "fingerprint": fingerprint,
                "account": "empty home; no sign-in or credentials copied",
                "task_execution": "not attempted",
                "voice_and_shared_history": "unverified",
            }
        finally:
            await rpc.close()
            await stop_owned(process.pid, created, desktop_exe, preserve={state.get("host_pid")})
            await process.wait()
            # The host is intentionally detached from the desktop. Clean up only
            # this probe's empty-home host, after ending the test's connections.
            if state.get("host_pid"):
                await stop_owned(state["host_pid"], state["host_created"], BUNDLED_CLI)


if __name__ == "__main__":
    result = asyncio.run(run())
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
