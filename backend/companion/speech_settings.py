"""Validated, versioned speech preferences. Quality never changes under pressure."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Language = Literal["auto", "english", "korean", "japanese"]


class SpeechSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    recognition_mode: Literal["accurate", "fast"] = "accurate"
    input_language: Language = "auto"
    reply_language: Language = "auto"
    vocabulary: list[str] = Field(default_factory=lambda: ["Ari", "Codex"], max_length=64)
    context_ms: Literal[160, 320, 560] = 320
    sensitivity: float = Field(default=0.7, ge=0.3, le=0.95)
    pause_tolerance: float = Field(default=0.3, ge=0.3, le=1.5)
    warm_up: bool = True
    fish_precision: Literal[8] = 8  # 4-bit is deliberately unavailable until validated.
    fish_streaming: bool = (
        True  # Gated by scripts/check_codec_stream.py; explicit whole-segment fallback.
    )

    @field_validator("vocabulary")
    @classmethod
    def validate_vocabulary(cls, words):
        words = list(dict.fromkeys(word.strip() for word in words if word.strip()))
        if any(len(word) > 80 or any(ord(c) < 32 for c in word) for word in words):
            raise ValueError(
                "Vocabulary entries must be single-line phrases of at most 80 characters"
            )
        return words

    @classmethod
    def load(cls, store):
        saved = store.setting("speech_settings", None)
        if saved is None:
            language = store.setting("language", "auto")
            settings = cls(input_language=language, reply_language=language)
            store.set_setting("speech_settings", settings.model_dump())
            return settings
        return cls.model_validate(saved)
