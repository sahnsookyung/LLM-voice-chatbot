import re

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
            match = re.match(r"\[mood=([^\]]+)\]\s*", candidate)
            if match:
                self.expression = match[1] if match[1] in EXPRESSIONS else "neutral"
                visible = candidate[match.end() :]
            elif candidate.startswith("["):
                visible = candidate.split("]", 1)[-1] if "]" in candidate else ""
            else:
                visible = self.prefix
            self.started = True
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


def speech_language(text: str, override: str) -> str:
    if override != "auto":
        return override
    if re.search(r"[가-힣]", text):
        return "korean"
    if re.search(r"[ぁ-んァ-ン一-龯]", text):
        return "japanese"
    return "english"
