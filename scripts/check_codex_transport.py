#!/usr/bin/env python3
"""Test bundled Unix transport in an empty temporary home, never user history/account.

This is not proof of shared desktop attachment, voice entitlement or Computer Use.
Only the explicitly started isolated process is terminated on completion.
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from companion.codex_compat import BUNDLED_CLI, desktop_fingerprint
from companion.codex_rpc import CodexRPC


async def run():
    fingerprint = await desktop_fingerprint()
    with tempfile.TemporaryDirectory(prefix="ari-host-", dir="/private/tmp") as folder:
        home = Path(folder)
        directory = home / "app-server-control"
        directory.mkdir(mode=0o700)
        endpoint = directory / "app-server-control.sock"
        # CODEX_HOME is the provider's actual isolation setting, not a shell alias.
        env = {**os.environ, "CODEX_HOME": str(home)}
        process = await asyncio.create_subprocess_exec(
            str(BUNDLED_CLI),
            "app-server",
            "--listen",
            "unix://" + str(endpoint),
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        rpc, resolved, identity = CodexRPC(endpoint), None, None
        version_process = None
        try:
            for _ in range(100):
                if endpoint.exists() or process.returncode is not None:
                    break
                await asyncio.sleep(0.05)
            initialized = await rpc.connect()
            resolved = rpc.endpoint
            info = resolved.lstat()
            identity = (info.st_dev, info.st_ino)
            loaded = await rpc.call("thread/loaded/list", {})
            version_process = await asyncio.create_subprocess_exec(
                str(BUNDLED_CLI),
                "app-server",
                "daemon",
                "version",
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            output, _ = await asyncio.wait_for(version_process.communicate(), 5)
            version = json.loads(output)
            checks = {
                "verified_bundled_peer": rpc.peer_pid == process.pid,
                "isolated_home": initialized.get("codexHome") == str(home),
                "no_loaded_tasks": loaded.get("data") == [],
                "matching_version": version.get("appServerVersion")
                == fingerprint["host_version"].removeprefix("codex-cli "),
            }
            if not all(checks.values()):
                raise RuntimeError("Isolated transport verification failed.")
            return {
                "kind": "isolated transport check; not live release acceptance",
                "fingerprint": fingerprint,
                "checks": checks,
                "shared_desktop": "unverified",
                "voice": "unverified",
                "computer_use": "unverified",
            }
        finally:
            await rpc.close()
            for child in (version_process, process):
                if child and child.returncode is None:
                    child.terminate()
                    try:
                        await asyncio.wait_for(child.wait(), 5)
                    except TimeoutError:
                        child.kill()
                        await child.wait()
            # Remove only this test's shortened socket if graceful exit left it behind.
            if resolved and resolved.exists():
                info = resolved.lstat()
                if (info.st_dev, info.st_ino) == identity:
                    resolved.unlink()


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(await run(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
