"""Exercise the real relay/host in an empty home, without account or model requests."""

import asyncio
import json
import os
import signal
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from companion.codex_bridge import bridge_socket, process_matches, read_state
from companion.codex_rpc import CodexRPC
from companion.codex_security import BUNDLED_CLI


async def request(process, identifier, method, params):
    process.stdin.write(
        json.dumps({"id": identifier, "method": method, "params": params}).encode() + b"\n"
    )
    await process.stdin.drain()
    async with asyncio.timeout(15):
        while line := await process.stdout.readline():
            value = json.loads(line)
            if value.get("id") == identifier:
                if "error" in value:
                    raise RuntimeError("Probe RPC rejected")
                return value["result"]
    raise RuntimeError("Relay exited before response")


async def finish(process):
    process.stdin.close()
    await asyncio.wait_for(process.wait(), 5)
    if process.returncode:
        raise RuntimeError("Relay did not exit cleanly")


async def run():
    with tempfile.TemporaryDirectory(prefix="ari-relay-", dir="/private/tmp") as folder:
        root = Path(folder)
        home, directory = root / "home", root / "relay"
        home.mkdir(mode=0o700)
        entrypoint = root / "probe_bridge.py"
        entrypoint.write_text(
            "import sys\nfrom pathlib import Path\n"
            f"sys.path.insert(0, {str(ROOT / 'backend')!r})\n"
            "from companion import codex_bridge\n"
            f"codex_bridge.bridge_directory = lambda: Path({str(directory)!r})\n"
            "codex_bridge.main()\n"
        )
        arguments = [
            "-c",
            "features.code_mode_host=false",
            "app-server",
            "--analytics-default-enabled",
            "-c",
            "plugins.codex-app-tools@openai-bundled.mcp_servers.codex_app.enabled=false",
            "-c",
            "plugins.code-review@openai-bundled.mcp_servers.code-review.enabled=false",
            "-c",
            'model_reasoning_effort="low"',
        ]
        processes, state = [], {}
        rpc = CodexRPC(bridge_socket(directory))

        async def start():
            process = await asyncio.create_subprocess_exec(
                str(ROOT / ".venv/bin/python"),
                str(entrypoint),
                *arguments,
                env={**os.environ, "CODEX_HOME": str(home)},
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            processes.append(process)
            identity = await request(
                process,
                1,
                "initialize",
                {
                    "clientInfo": {"name": "ari_relay_test", "version": "0.1"},
                    "capabilities": {"experimentalApi": True},
                },
            )
            process.stdin.write(b'{"method":"initialized"}\n')
            await process.stdin.drain()
            return process, identity

        try:
            process, identity = await start()
            state = read_state(directory)
            config = await request(process, 2, "config/read", {"includeLayers": False})
            other = await rpc.connect()
            checks = {
                "stdio_relay_initialized": identity["codexHome"] == str(home),
                "runtime_override_preserved": config["config"]["model_reasoning_effort"] == "low",
                "second_client_same_peer": rpc.peer_pid == state["host_pid"],
                "second_client_same_home": other["codexHome"] == str(home),
            }
            await finish(process)
            checks["relay_process_exit_preserves_host"] = (
                await rpc.call("thread/loaded/list", {})
            )["data"] == []
            second, _ = await start()
            checks["reconnect_reuses_same_host"] = (
                read_state(directory)["host_pid"] == state["host_pid"]
            )
            await finish(second)
            checks["second_exit_preserves_host"] = (await rpc.call("thread/loaded/list", {}))[
                "data"
            ] == []
            return {
                "passed": all(checks.values()),
                "checks": checks,
                "user_agent": identity["userAgent"],
                "task_execution": "not attempted",
            }
        finally:
            for process in processes:
                if process.returncode is None:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), 5)
            await rpc.close()
            if not state and (directory / "host.json").exists():
                state = read_state(directory)
            if state and process_matches(state["host_pid"], state["host_created"], BUNDLED_CLI):
                os.killpg(state["host_pid"], signal.SIGTERM)
                for _ in range(100):
                    if not process_matches(state["host_pid"], state["host_created"], BUNDLED_CLI):
                        break
                    await asyncio.sleep(0.05)
            # This endpoint belongs only to this temporary home/test process.
            if rpc.endpoint.exists():
                rpc.endpoint.unlink()


if __name__ == "__main__":
    result = asyncio.run(run())
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
