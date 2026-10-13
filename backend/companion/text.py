import re

from .behavior import INTENTS
from .capabilities import AVATAR_ACTIONS

EXPRESSIONS = {
    "neutral",
    "happy",
    "curious",
    "concerned",
    "surprised",
    "annoyed",
    "embarrassed",
    "sleepy",
}


class SpeechText:
    """Incremental prefix parser and bounded, punctuation-aware clause buffer."""

    def __init__(self):
        self.prefix = ""
        self.started = False
        self.expression = "neutral"
        self.intent = None
        self.intent_prefix = ""
        self.intent_complete = False
        self.action = None
        self.action_prefix = ""
        self.header_complete = False
        self.body_pending = ""
        self.dropping_tag = False
        self.buffer = ""
        self.first = True

    def feed(self, delta: str, final=False) -> tuple[str, list[str]]:
        visible = delta
        if not self.started:
            self.prefix += delta
            candidate = self.prefix.lstrip()
            # A provider can send empty/whitespace deltas before the actual mood tag.
            if not candidate and len(self.prefix) < 64 and not final:
                return "", []
            if (
                candidate.startswith("[")
                and "]" not in candidate
                and len(candidate) < 64
                and not final
            ):
                return "", []
            match = re.match(r"\[\s*mood\s*=\s*([^\]]+?)\s*\]\s*", candidate, re.IGNORECASE)
            if match:
                mood = match[1].strip().lower()
                self.expression = mood if mood in EXPRESSIONS else "neutral"
                visible = candidate[match.end() :]
            elif candidate.startswith("["):
                visible = candidate.split("]", 1)[-1] if "]" in candidate else ""
            else:
                visible = self.prefix
            self.started = True
        if not self.intent_complete:
            self.intent_prefix += visible
            candidate = self.intent_prefix.lstrip()
            if not candidate and len(self.intent_prefix) < 64 and not final:
                return "", []
            if (
                candidate.startswith("[")
                and "]" not in candidate
                and len(candidate) < 64
                and not final
            ):
                return "", []
            match = re.match(r"\[\s*intent\s*=\s*([^\]]+?)\s*\]\s*", candidate, re.IGNORECASE)
            if match:
                intent = match[1].strip().lower()
                self.intent = intent if intent in INTENTS else None
                visible = candidate[match.end() :]
            else:
                visible = candidate
            self.intent_complete = True
            self.prefix = self.intent_prefix = ""
        if not self.header_complete:
            self.action_prefix += visible
            candidate = self.action_prefix.lstrip()
            if not candidate and len(self.action_prefix) < 64 and not final:
                return "", []
            if (
                candidate.startswith("[")
                and "]" not in candidate
                and len(candidate) < 64
                and not final
            ):
                return "", []
            match = re.match(r"\[\s*action\s*=\s*([^\]]+?)\s*\]\s*", candidate, re.IGNORECASE)
            if match:
                action = match[1].strip().lower()
                self.action = action if action in (*AVATAR_ACTIONS, "none") else "invalid"
                visible = candidate[match.end() :]
            else:
                visible = candidate
            self.header_complete = True
            self.action_prefix = ""
        visible = self.clean_body(visible, final)
        self.buffer += visible
        clauses = []
        while self.buffer:
            # Decimal points remain part of a number. Commas are allowed only for the first phrase.
            pattern = r"[。！？!?\n]|(?<!\d)\.(?=\s|$)|(?<=\d)\.(?!\d)(?=\s|$)"
            if self.first:
                pattern += r"|[,，、;；](?=\s|[^\x00-\x7f]|$)"
            match = re.search(pattern, self.buffer)
            end = match.end() if match else 0
            if end and end < 8 and self.buffer[end - 1] in ",，、;；":
                end = 0
            if not end and len(self.buffer) > 240:
                end = self.buffer.rfind(" ", 100, 240)
                if end < 0 and re.search(r"[가-힣ぁ-んァ-ン一-龯]", self.buffer[:240]):
                    end = 240
                end = max(end, 0)  # Never cut through an unbroken word or number.
            if not end:
                break
            clause, self.buffer = self.buffer[:end].strip(), self.buffer[end:].lstrip()
            if clause:
                clauses.append(clause)
                self.first = False
        if final and self.buffer.strip():
            clauses.append(self.buffer.strip())
            self.buffer = ""
        return visible, clauses

    def clean_body(self, delta: str, final: bool) -> str:
        """Models occasionally repeat metadata mid-reply; never speak those tags."""
        text = self.body_pending + delta
        self.body_pending = ""
        if self.dropping_tag:
            if "]" not in text:
                return ""
            text = text.split("]", 1)[1]
            self.dropping_tag = False
        output = []
        markers = ("[mood=", "[intent=", "[action=")
        while "[" in text:
            before, tail = text.split("[", 1)
            output.append(before)
            tail = "[" + tail
            compact = re.sub(r"\s+", "", tail).lower()
            if any(compact.startswith(marker) for marker in markers):
                if "]" in tail:
                    text = tail.split("]", 1)[1]
                    continue
                if not final:
                    if len(tail) < 64:
                        self.body_pending = tail
                    else:
                        self.dropping_tag = True
                return "".join(output)
            if (
                not final
                and len(tail) < 64
                and any(marker.startswith(compact) for marker in markers)
            ):
                self.body_pending = tail
                return "".join(output)
            output.append("[")
            text = tail[1:]
        return "".join(output) + text


def speech_language(text: str, override: str) -> str:
    if override != "auto":
        return override
    if re.search(r"[가-힣]", text):
        return "korean"
    if re.search(r"[ぁ-んァ-ン一-龯]", text):
        return "japanese"
    return "english"
