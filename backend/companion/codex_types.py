from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CodexCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal[
        "refresh",
        "select",
        "new",
        "history_more",
        "voice_end",
        "stop_task",
        "settings",
        "approval",
        "answer",
        "voice_select",
        "tasks_more",
        "archived",
    ]
    thread_id: str = Field(default="", max_length=128)
    cwd: str = Field(default="", max_length=4096)
    project_id: str | None = Field(default=None, max_length=128)
    request_id: str = Field(default="", max_length=128)
    decision: Literal["accept", "acceptForSession", "decline", "cancel"] = "decline"
    answers: dict[str, list[str]] = Field(default_factory=dict, max_length=16)
    # Only model/effort/plugin disable-list changes, never permission escalation.
    model: str | None = Field(default=None, max_length=128)
    effort: str | None = Field(default=None, max_length=32)
    disabled_plugins: list[str] | None = Field(default=None, max_length=64)
    voice: str = Field(default="", max_length=64)


class BackendChoice(BaseModel):
    backend: Literal["local", "codex"]


class AvatarCue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    emotion: Literal[
        "neutral", "happy", "curious", "concerned", "annoyed", "surprised", "embarrassed", "sleepy"
    ] = "neutral"
    action: Literal[
        "none", "wave", "nod", "explain", "shift", "step", "twirl", "raise-arms", "shrug", "point"
    ] = "none"


def approval_result(method: str, decision: str, params: dict[str, Any]):
    if method in {"item/commandExecution/requestApproval", "item/fileChange/requestApproval"}:
        allowed = params.get("availableDecisions")
        if allowed and decision not in allowed:
            raise ValueError("This decision is not offered by Codex.")
        return {"decision": decision}
    if method == "item/permissions/requestApproval":
        # Scope is shown before acceptance; never broaden the requested subset.
        return {
            "permissions": params.get("permissions", {}) if decision == "accept" else {},
            "scope": "turn",
        }
    if method == "mcpServer/elicitation/request":
        if decision == "accept" and params.get("mode") != "url":
            raise ValueError("Complete this form in Codex. Ari never invents form answers.")
        return {
            "action": "accept"
            if decision == "accept"
            else "cancel"
            if decision == "cancel"
            else "decline",
            "content": None,
        }
    raise ValueError("Complete this request in Codex.")
