"""Read-only Ari session/history validation; diagnostics never include item text."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from companion.codex_compat import desktop_fingerprint, socket_path
from companion.codex_history import project_item
from companion.codex_rpc import CodexRPC
from companion.codex_session import CodexSession
from companion.codex_validation import validation_thread


class PresentationSettings:
    def __init__(self):
        self.values = {}

    def setting(self, key, default=None):
        return self.values.get(key, default)

    def set_setting(self, key, value):
        self.values[key] = value


async def run(thread):
    rpc = CodexRPC(socket_path())
    counts, maximum, native_sizes = {}, {}, {}
    newest_ids = set()
    call = rpc.call

    async def measured(method, params, **kwargs):
        value = await call(method, params, **kwargs)
        counts[method] = counts.get(method, 0) + 1
        maximum[method] = max(maximum.get(method, 0), len(json.dumps(value).encode()))
        if method == "thread/timeline/list" and not params.get("cursor"):
            newest_ids.update(
                row["id"]
                for entry in value.get("data", [])
                if (row := project_item(entry.get("item", {}))) is not None
            )
        return value

    rpc.call = measured
    history_ids, disconnects = [], []

    async def event(value):
        kind = value["type"]
        native_sizes[kind] = max(
            native_sizes.get(kind, 0), len(json.dumps(value, ensure_ascii=False).encode())
        )
        if kind == "codex_history":
            if not value.get("append"):
                history_ids.clear()
            history_ids.extend(row["id"] for row in value["messages"])
        if kind in {"error", "codex_disconnected"}:
            disconnects.append(kind)

    async def no_audio(_):
        raise RuntimeError("A read-only probe must not receive voice audio")

    session = CodexSession(rpc, PresentationSettings(), event, no_audio, validation_thread=thread)
    try:
        await rpc.connect()
        await session.open()
        checks = {
            "session_connected": not rpc.closed,
            "selected_existing_task": bool(session.thread and session.thread["id"] == thread),
            "history_complete": not session.history_loading and not session.cursor,
            "no_history_duplicates": len(history_ids) == len(set(history_ids)),
            "newest_page_preserved": newest_ids <= set(history_ids),
            "native_messages_under_one_mib": all(
                size < 1024 * 1024 for size in native_sizes.values()
            ),
            "no_disconnect": not disconnects,
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "fingerprint": await desktop_fingerprint(),
            "displayed_rows": len(history_ids),
            "history_window_truncated": session.timeline.truncated,
            "active_turn_observed": session.active_turn is not None,
            "calls": counts,
            "maximum_host_response_bytes": maximum,
            "maximum_native_event_bytes": native_sizes,
            "task_submission": "none",
            "microphone": "not opened",
        }
    finally:
        await session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread", required=True, type=validation_thread)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = asyncio.run(run(args.thread))
    content = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(content)
    print(content)
    raise SystemExit(0 if result["passed"] else 1)
