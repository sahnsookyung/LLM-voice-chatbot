"""Disposable display projection of Codex's canonical timeline, never a second history."""

import json
import re
from collections import OrderedDict
from html import unescape


def voice_delegation_text(text):
    """Recognize the host's complete delegation envelope, never execute its text.

    The user's words already have a canonical voice transcript. This is the
    backend handoff copy, which belongs in task activity rather than a second
    visible user message. Ordinary or malformed text remains a conversation.
    """
    match = re.fullmatch(
        r"<realtime_delegation>\s*<input>([^<]+)</input>\s*"
        r"<transcript_delta>[^<]*</transcript_delta>\s*</realtime_delegation>",
        text.strip(),
    )
    return unescape(match[1]).strip() if match else None


def display_user_text(text):
    """Project structured question replies as answers, without executing payloads."""
    opening, closing = "<send_user_message_question_reply>", "</send_user_message_question_reply>"
    value = text.strip()
    if not (value.startswith(opening) and value.endswith(closing)):
        return text
    try:
        replies = json.loads(value[len(opening) : -len(closing)])
    except (ValueError, TypeError):
        return text
    if not isinstance(replies, list) or not 1 <= len(replies) <= 16:
        return text
    if not all(
        isinstance(row, dict)
        and isinstance(row.get("questionItemId"), str)
        and isinstance(row.get("answer"), str)
        and row["answer"].strip()
        for row in replies
    ):
        return text
    return "\n\n".join(row["answer"] for row in replies)


def display_batches(rows, budget=128 * 1024):
    """Keep native WebSocket messages below its 1 MiB default, including Unicode."""
    batch, size = [], 0
    for row in rows:
        encoded = len(json.dumps(row, ensure_ascii=False).encode("utf-8"))
        if batch and size + encoded > budget:
            yield batch
            batch, size = [], 0
        batch.append(row)
        size += encoded
    # Empty history still has to clear the previous display.
    if batch or not rows:
        yield batch


def tool_catalog_summary(value):
    """The native UI needs availability/counts, never executable tool schemas."""
    return {
        "data": [
            {
                "name": row.get("name"),
                "toolCount": len(row.get("tools") or {}),
                "runtimeStatus": row.get("runtimeStatus"),
                "toolsError": row.get("toolsError"),
            }
            for row in value.get("data", [])
        ],
        "nextCursor": value.get("nextCursor"),
    }


def project_item(item, *, realtime=False):
    kind = item.get("type", "")
    identifier = item.get("id")
    if not identifier:
        return None
    delegation = None
    if kind == "transcriptSegment":
        role, text = item.get("role", "assistant"), item.get("text", "")
    elif kind == "agentMessage":
        role, text = "assistant", item.get("text", "")
    elif kind == "userMessage":
        role = "user"
        text = "\n".join(
            part.get("text", "")
            for part in item.get("content", [])
            if isinstance(part, dict) and part.get("type") == "text"
        )
        delegation = voice_delegation_text(text)
        text = (
            f"Voice request sent to Codex: {delegation}" if delegation else display_user_text(text)
        )
    elif kind in {
        "commandExecution",
        "fileChange",
        "mcpToolCall",
        "dynamicToolCall",
        "webSearch",
        "plan",
    }:
        role = "tool"
        if kind == "commandExecution":
            text = item.get("command", "Command")
        elif kind == "plan":
            text = item.get("text", "Plan")
        elif kind == "fileChange":
            text = "File changes: " + ", ".join(x.get("path", "") for x in item.get("changes", []))
        else:
            text = item.get("tool", item.get("query", kind))
    else:
        # Includes raw reasoning and bemItemPromoted (a reference, not a new message).
        return None
    return {
        "id": identifier,
        "role": role,
        "text": str(text)[:64000],
        "status": item.get("status", "complete"),
        "realtime": realtime,
        "kind": kind,
        # Use protocol metadata, never wording heuristics, to hide progress.
        # Legacy/unknown phases remain visible so an answer cannot disappear.
        "presentation": "activity"
        if role == "tool"
        or delegation
        or (
            kind == "agentMessage"
            and item.get("phase") == "commentary"
            and not item.get("questions")
        )
        else "conversation",
    }


class CodexTimeline:
    def __init__(self, limit=2000):
        self.rows = OrderedDict()
        self.positions = {}
        self.limit = limit
        self.completed = set()
        self.truncated = False

    def upsert(self, item, *, realtime=False, position=None, completed=False):
        value = project_item(item, realtime=realtime)
        if value is None:
            return None
        identifier = value["id"]
        if position is not None:
            self.positions[identifier] = position
        if identifier in self.completed and not completed:
            return self.rows.get(identifier)
        if completed:
            self.completed.add(identifier)
        self.rows[identifier] = value
        while len(self.rows) > self.limit:
            # Pages arrive newest-first, with entries in rollout order within
            # each page. Arrival order would evict recent replies for old ones.
            # Live items have no persisted position yet and belong at the end.
            removed = min(self.rows, key=lambda key: self.positions.get(key, float("inf")))
            del self.rows[removed]
            self.positions.pop(removed, None)
            self.completed.discard(removed)
            self.truncated = True
        return self.rows.get(identifier)

    def load(self, entries):
        for entry in sorted(entries, key=lambda x: x.get("position", 0)):
            if entry.get("type") in {"item", "realtime"}:
                self.upsert(
                    entry.get("item", {}),
                    realtime=entry["type"] == "realtime",
                    position=entry.get("position"),
                    completed=entry.get("item", {}).get("status", "complete")
                    not in {"inProgress", "in_progress"},
                )

    def delta(self, identifier, delta):
        if identifier not in self.rows or identifier in self.completed:
            return None
        row = self.rows[identifier]
        row["text"] = (row["text"] + delta)[:64000]
        return row

    def snapshot(self):
        # Persisted positions trump arrival order after a reconnect or page load.
        return sorted(self.rows.values(), key=lambda x: self.positions.get(x["id"], float("inf")))
