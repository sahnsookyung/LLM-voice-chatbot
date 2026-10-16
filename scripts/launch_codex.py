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

INSPECTED_SCHEMA = "59c1c2f4c78bf368f38d7ae8372d6d9da94b92c0c56ddf4e32278f4ed5cd803e"


async def is_running(name):
    expected = (
        str(DESKTOP / "Contents/MacOS/ChatGPT")
        if name in {"ChatGPT", "Codex"}
        else str(ROOT / "build/Local Companion.app/Contents/MacOS/Companion")
    )
    try:
        # macOS pgrep can disagree with the actual application name. Inspect both
        # the kernel process name and executable rather than argv matching.
        for process in psutil.process_iter(["name", "exe"]):
            if process.info["name"] == name or process.info["exe"] == expected:
                return True
            if process.info["name"] is None:
                raise RuntimeError(
                    "A process identity is unavailable. Refusing to launch a daemon."
                )
    except psutil.Error as error:
        raise RuntimeError(
            "Cannot inspect running clients. Refusing to launch a daemon."
        ) from error
    return False


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
    if fingerprint["schema_sha256"] != INSPECTED_SCHEMA:
        raise SystemExit(
            "This desktop build has not passed launcher compatibility inspection. Run scripts/check_codex.py; use normal desktop launch meanwhile."
        )
    for name in (
        ("ChatGPT", "Codex", "Companion") if args.validation_thread else ("ChatGPT", "Codex")
    ):
        if await is_running(name):
            raise SystemExit(
                f"Finish your work and quit {name} before using this launcher. Nothing was restarted."
            )
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
    except CodexError as error:
        raise SystemExit(str(error)) from None
    except (OSError, RuntimeError, TimeoutError, ValueError):
        raise SystemExit(
            "Launch stopped: host ownership/version or readiness could not be verified. "
            "No further process will be started. Run scripts/check_codex.py for details."
        ) from None
