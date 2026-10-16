import asyncio
import json
import os

import pytest
from companion import codex_bridge as bridge
from companion.codex_rpc import CodexError


def arguments():
    return [
        "-c",
        "features.code_mode_host=true",
        "app-server",
        "--analytics-default-enabled",
        *[value for key in sorted(bridge.PLUGIN_KEYS) for value in ["-c", key + "=true"]],
    ]


def test_only_desktop_main_server_is_relayed():
    args = arguments()
    assert bridge.is_desktop_server(args)
    assert not bridge.is_desktop_server(["--version"])
    assert not bridge.is_desktop_server(["app-server"])
    assert not bridge.is_desktop_server(args + ["daemon", "version"])
    assert not bridge.is_desktop_server(args + ["--listen", "stdio://"])
    assert not bridge.is_desktop_server(args + ["-c"])


def test_runtime_environment_preserved_without_recursive_cli_or_websocket_override():
    source = {
        "RUNTIME_CONFIG": "sentinel",
        "AUTH_STORAGE_SETTING": "private-sentinel",
        "CODEX_CLI_PATH": "/relay",
        "CODEX_APP_SERVER_WS_URL": "wrong-endpoint",
    }
    env = bridge.host_environment(source)
    assert env["RUNTIME_CONFIG"] == source["RUNTIME_CONFIG"]
    assert env["AUTH_STORAGE_SETTING"] == source["AUTH_STORAGE_SETTING"]
    assert env["CODEX_CLI_PATH"] == str(bridge.BUNDLED_CLI)
    assert "CODEX_APP_SERVER_WS_URL" not in env
    digest = bridge.launch_digest(arguments(), env)
    assert len(digest) == 64 and "private-sentinel" not in digest
    assert digest != bridge.launch_digest(arguments(), env | {"RUNTIME_CONFIG": "changed"})
    assert digest == bridge.launch_digest(arguments(), env | {"SHLVL": "2"})


async def test_existing_mismatched_or_unverifiable_host_is_never_replaced(tmp_path, monkeypatch):
    directory = tmp_path / "bridge"
    directory.mkdir(mode=0o700)
    path = bridge.bridge_socket(directory)
    path.parent.mkdir(mode=0o700)
    path.touch()
    monkeypatch.setattr(bridge, "validate_socket_directory", lambda _: None)

    def forbidden(*_, **__):
        pytest.fail("An existing endpoint must never cause another server to start")

    monkeypatch.setattr(bridge.subprocess, "Popen", forbidden)
    bridge.write_state(directory, {"configuration_sha256": "other"})
    with pytest.raises(CodexError, match="different desktop runtime"):
        await bridge.ensure_host(arguments(), {}, directory)
    bridge.write_state(
        directory,
        {
            "configuration_sha256": bridge.launch_digest(arguments(), bridge.host_environment({})),
            "host_pid": os.getpid(),
            "host_created": 0,
        },
    )
    with pytest.raises(CodexError, match="identity changed"):
        await bridge.ensure_host(arguments(), {}, directory)
    assert path.exists()


class Socket:
    def __init__(self):
        self.sent = asyncio.Queue()
        self.incoming = asyncio.Queue()
        self.closed = False

    async def send(self, raw):
        await self.sent.put(raw)

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.incoming.get()

    async def close(self):
        self.closed = True


class Writer:
    def __init__(self):
        self.messages = asyncio.Queue()

    def write(self, raw):
        self.messages.put_nowait(raw)

    async def drain(self):
        pass


async def test_relay_preserves_identity_request_ids_approvals_and_order_without_resubmission():
    reader, writer, ws = asyncio.StreamReader(), Writer(), Socket()
    initialized = []
    task = asyncio.create_task(bridge.relay(reader, writer, ws, lambda: initialized.append(True)))
    try:
        messages = [
            {
                "id": "init",
                "method": "initialize",
                "params": {"clientInfo": {"name": "test_client"}},
            },
            {"method": "initialized"},
            {"id": 3, "method": "turn/start", "params": {"threadId": "task"}},
        ]
        for message in messages:
            raw = json.dumps(message)
            reader.feed_data(raw.encode() + b"\n")
            assert await asyncio.wait_for(ws.sent.get(), 1) == raw
        for message in [
            {"id": "init", "result": {}},
            {"id": "approval", "method": "item/commandExecution/requestApproval", "params": {}},
        ]:
            raw = json.dumps(message)
            await ws.incoming.put(raw)
            assert await asyncio.wait_for(writer.messages.get(), 1) == raw.encode() + b"\n"
        assert initialized == [True]
        approval = '{"id":"approval","result":{"decision":"decline"}}'
        reader.feed_data(approval.encode() + b"\n")
        assert await asyncio.wait_for(ws.sent.get(), 1) == approval
        reader.feed_eof()
        await asyncio.wait_for(task, 1)
        assert ws.closed and ws.sent.empty()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_incomplete_frame_closes_connection_without_sending_partial_command():
    reader, writer, ws = asyncio.StreamReader(), Writer(), Socket()
    reader.feed_data(b'{"method":"turn/start"')
    reader.feed_eof()
    with pytest.raises(CodexError, match="incomplete"):
        await bridge.relay(reader, writer, ws)
    assert ws.closed and ws.sent.empty()


@pytest.mark.parametrize("failure", [None, "parent", "script", "identity", "initialized", "peer"])
async def test_desktop_relay_evidence_requires_live_parent_script_and_socket(monkeypatch, failure):
    from types import SimpleNamespace

    from companion import codex_compat as compat

    state = {
        "host_pid": 10,
        "host_created": 100,
        "bridge_pid": 20,
        "bridge_created": 200,
        "desktop_pid": 30,
        "desktop_created": 300,
        "initialized": failure != "initialized",
    }
    output = "p10\nccodex\nf3\nd0x123\nn/test.sock\nf4\nd0x456\nn->0x789\np20\ncPython\nf5\nd0x789\nn->0x456"
    if failure == "peer":
        output = output.replace("n->0x456", "n->0x999")

    async def command(*args, **kwargs):
        return output

    monkeypatch.setattr(compat, "command_output", command)
    monkeypatch.setattr(compat, "read_state", lambda _: state)
    monkeypatch.setattr(compat, "process_matches", lambda *args: failure != "identity")
    monkeypatch.setattr(
        compat.psutil,
        "Process",
        lambda _: SimpleNamespace(
            ppid=lambda: 99 if failure == "parent" else 30,
            cmdline=lambda: [
                "python",
                "wrong" if failure == "script" else str(compat.BRIDGE_SCRIPT),
            ],
        ),
    )
    assert await compat.desktop_socket_evidence("/test.sock", 10) is (failure is None)


@pytest.mark.parametrize("case", ["idle", "active", "unknown", "queued", "background", "page"])
async def test_explicit_refresh_requires_idle_tasks_without_pending_work(case):
    from companion.codex_host_lifecycle import require_idle

    class RPC:
        async def call(self, method, params):
            if method == "thread/loaded/list":
                return {"data": ["one"], "nextCursor": "more" if case == "page" else None}
            if method == "thread/read":
                return {
                    "thread": {
                        "status": {"type": case if case in {"active", "unknown"} else "idle"}
                    }
                }
            if method == "thread/queue/list":
                return {"data": [1] if case == "queued" else []}
            if method == "thread/backgroundTerminals/list":
                return {"data": [1] if case == "background" else []}
            pytest.fail("Unexpected or mutating RPC")

    if case == "idle":
        await require_idle(RPC())
    else:
        with pytest.raises(CodexError, match="left running"):
            await require_idle(RPC())


async def test_explicit_refresh_never_stops_host_with_another_client(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from companion import codex_host_lifecycle as lifecycle

    directory = tmp_path / "bridge"
    directory.mkdir(mode=0o700)
    path = bridge.bridge_socket(directory)
    path.parent.mkdir(mode=0o700)
    path.touch()
    bridge.write_state(directory, {"host_pid": 10, "host_created": 100})
    monkeypatch.setattr(lifecycle, "process_matches", lambda *args: True)
    monkeypatch.setattr(lifecycle, "validate_socket_directory", lambda *args: None)
    monkeypatch.setattr(
        lifecycle.psutil,
        "Process",
        lambda *args: SimpleNamespace(
            terminate=lambda: pytest.fail("Connected host must never be stopped"),
        ),
    )

    class RPC:
        peer_pid = 10
        endpoint = path

        def __init__(self, _):
            pass

        async def connect(self):
            pass

        async def close(self):
            pass

    async def command(*args, **kwargs):
        return f"p10\nccodex\nf3\nd0x111\nn{path}\nf4\nd0x222\nn->0x333\np99\ncPython\nf5\nd0x333\nn->0x222"

    monkeypatch.setattr(lifecycle, "CodexRPC", RPC)
    monkeypatch.setattr(lifecycle, "command_output", command)
    with pytest.raises(CodexError, match="Another client"):
        await lifecycle.restart_idle_host(directory)
    assert path.exists()
