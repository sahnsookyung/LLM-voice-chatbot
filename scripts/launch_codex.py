#!/usr/bin/env python3
"""Opt-in shared-host launcher. Never quits clients; idle-host refresh is explicit."""

import argparse
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
from companion.codex_bridge import BRIDGE_SCRIPT, bridge_directory
from companion.codex_compat import (
    DESKTOP,
    command_output,
    desktop_fingerprint,
    launcher_blocker,
    probe,
)
from companion.codex_host_lifecycle import restart_idle_host
from companion.codex_rpc import CodexError
from companion.codex_security import validate_socket_directory
from companion.codex_validation import validation_thread


async def running_client(name):
    expected = (
        str(DESKTOP / "Contents/MacOS/ChatGPT")
        if name in {"ChatGPT", "Codex"}
        else str(ROOT / "build/Local Companion.app/Contents/MacOS/Companion")
    )
    try:
        # macOS pgrep can disagree with the actual application name. Inspect both
        # the kernel process name and executable rather than argv matching.
        for process in psutil.process_iter(["pid", "name", "exe"]):
            if process.info["name"] == name or process.info["exe"] == expected:
                return process.info
            if process.info["name"] is None:
                raise RuntimeError(
                    "A process identity is unavailable. Refusing to launch a daemon."
                )
    except psutil.Error as error:
        raise RuntimeError(
            "Cannot inspect running clients. Refusing to launch a daemon."
        ) from error
    return None


def client_quit_message(name, process):
    desktop = name in {"ChatGPT", "Codex"}
    label = "Codex desktop" if desktop else "Local Companion (Ari)"
    identity = f"macOS process {process['name']!r}, PID {process['pid']}"
    detail = (
        f"This Codex build is installed as {DESKTOP}; its process can be named ChatGPT. "
        if desktop
        else ""
    )
    return (
        f"{label} is still running ({identity}). {detail}"
        f"Finish your work and quit {label} with Cmd+Q. "
        "Run this launcher from a separate Terminal window before reopening either app. "
        "Nothing was restarted."
    )


async def require_clients_closed(names, *, wait=False, timeout=120):
    """Optionally wait for the user's quit action; never terminate a client."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    previous = None
    waited = False
    while True:
        for name in names:
            if process := await running_client(name):
                break
        else:
            return waited
        if not wait:
            raise SystemExit(client_quit_message(name, process))
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise SystemExit(
                "Timed out waiting for apps to close. " + client_quit_message(name, process)
            )
        identity = (name, process["pid"])
        if identity != previous:
            label = "Codex desktop" if name in {"ChatGPT", "Codex"} else "Local Companion (Ari)"
            print(
                f"Waiting for {label} (PID {process['pid']}) to close. "
                f"Switch to {label} and press Cmd+Q after finishing your work. "
                "Leave this Terminal open; the launcher will reopen both apps. "
                f"Timeout: {timeout} seconds. Ctrl+C cancels. No app will be force-quit.",
                flush=True,
            )
            previous = identity
        waited = True
        await asyncio.sleep(min(0.5, remaining))


def prepare_bridge():
    directory = bridge_directory()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / "codex-desktop-relay"
    validate_socket_directory(path)
    python = ROOT / ".venv/bin/python"
    content = (
        "#!/bin/sh\nexec "
        + shlex.quote(str(python))
        + " "
        + shlex.quote(str(BRIDGE_SCRIPT))
        + ' "$@"\n'
    )
    fd, temporary = tempfile.mkstemp(prefix="relay-", dir=directory)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(content)
        os.chmod(temporary, 0o700)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--restart-idle-host",
        action="store_true",
        help="Explicitly retire only the relay's verified idle host before desktop gets fresh tool identities",
    )
    parser.add_argument(
        "--wait-for-clients",
        action="store_true",
        help="Wait up to 120 seconds for you to quit Codex/Ari, then launch; never quits them for you",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Read-only readiness check; never launch or stop clients",
    )
    parser.add_argument(
        "--validation-thread",
        type=validation_thread,
        help="Open Ari in temporary validation mode for this existing test task",
    )
    args = parser.parse_args()
    if args.check:
        print(json.dumps(await probe(ROOT / ".runtime/codex/acceptance.json"), indent=2))
        return
    fingerprint = await desktop_fingerprint()
    if blocked := launcher_blocker(fingerprint):
        raise SystemExit(blocked + " No process was started or stopped.")
    names = (
        ("ChatGPT", "Codex", "Companion") if args.validation_thread else ("ChatGPT", "Codex")
    )
    if await require_clients_closed(names, wait=args.wait_for_clients):
        # An update or manual reopening during the wait must not bypass preflight.
        if blocked := launcher_blocker(await desktop_fingerprint()):
            raise SystemExit(blocked + " No process was started or stopped.")
        await require_clients_closed(names)
    relay_path = prepare_bridge()
    if args.restart_idle_host:
        await restart_idle_host(bridge_directory())
    env = os.environ.copy()
    env["CODEX_CLI_PATH"] = str(relay_path)
    env["CODEX_APP_SERVER_FORCE_CLI"] = "1"
    env.pop("CODEX_APP_SERVER_USE_LOCAL_DAEMON", None)
    env.pop("CODEX_APP_SERVER_WS_URL", None)
    await asyncio.create_subprocess_exec(
        str(DESKTOP / "Contents/MacOS/ChatGPT"),
        env=env,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    # Wait for evidence of attachment; an environment flag is not a successful launch.
    readiness = {}
    for _ in range(15):
        await asyncio.sleep(2)
        readiness = await probe(ROOT / ".runtime/codex/acceptance.json")
        if readiness["shared_host"]:
            break
    if not readiness.get("shared_host"):
        raise SystemExit(
            "Desktop relay attachment was not verified. Ari was not opened. Run --check for details. "
            "If reconnecting after a desktop restart, quit both clients and rerun with --restart-idle-host "
            "to refresh the previous idle host's tool identities. Existing processes were left running."
        )
    arguments = (
        ["--args", "--codex-validation-thread", args.validation_thread]
        if args.validation_thread
        else []
    )
    await command_output("/usr/bin/open", ROOT / "build/Local Companion.app", *arguments)
    print(
        "Opened Ari and Codex. Run scripts/check_codex.py to verify actual shared attachment. Voice remains gated until live acceptance checks pass."
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        raise SystemExit("Launcher cancelled. Existing processes were left running.") from None
    except CodexError as error:
        raise SystemExit(str(error)) from None
    except (OSError, RuntimeError, TimeoutError, ValueError):
        raise SystemExit(
            "Launch stopped: host ownership/version or readiness could not be verified. "
            "No further process will be started. Run scripts/check_codex.py for details."
        ) from None
