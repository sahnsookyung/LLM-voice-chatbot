"""Explicit one-task validation, independent of release acceptance and preferences."""

import re


def validation_thread(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise ValueError("Choose a valid existing Codex task ID for validation.")
    return value


def validation_ready(readiness):
    return readiness.get("shared_host") is True and all(
        readiness.get("checks", {}).get(key) is True
        for key in (
            "verified_peer",
            "same_home",
            "desktop_socket",
            "host_version_matches",
            "threads",
            "voices",
            "account",
        )
    )


class TransientCodexSettings:
    """Reuse presentation preferences without changing the saved backend or task."""

    def __init__(self, source, thread):
        self.source = source
        self.values = {"codex_thread": thread}

    def setting(self, name, default=None):
        return self.values.get(name, self.source.setting(name, default))

    def set_setting(self, name, value):
        self.values[name] = value
