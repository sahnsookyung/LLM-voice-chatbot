import struct
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .codex_types import CodexCommand
from .speech_settings import SpeechSettings

VERSION = 1
# Audio packets: magic, turn UUID, sequence, sample offset, PCM16 little-endian payload.
AUDIO_HEADER = struct.Struct("<4s16sIQ")
MAGIC = b"LVA1"
MIC_MAGIC = b"LVM2"


def microphone_packet(capture: str, sequence: int, offset: int, pcm: bytes) -> bytes:
    return AUDIO_HEADER.pack(MIC_MAGIC, uuid.UUID(capture).bytes, sequence, offset) + pcm


def decode_microphone(packet: bytes):
    if len(packet) < AUDIO_HEADER.size or (len(packet) - AUDIO_HEADER.size) % 2:
        raise ValueError("Invalid microphone packet length")
    magic, capture, sequence, offset = AUDIO_HEADER.unpack_from(packet)
    if magic != MIC_MAGIC or not 0 < len(packet) - AUDIO_HEADER.size <= 8192:
        raise ValueError("Invalid microphone frame")
    return str(uuid.UUID(bytes=capture)), sequence, offset, packet[AUDIO_HEADER.size :]


def audio_packet(turn: str, sequence: int, offset: int, pcm: bytes) -> bytes:
    return AUDIO_HEADER.pack(MAGIC, uuid.UUID(turn).bytes, sequence, offset) + pcm


def decode_audio(packet: bytes):
    if len(packet) < AUDIO_HEADER.size or (len(packet) - AUDIO_HEADER.size) % 2:
        raise ValueError("Invalid audio packet length")
    magic, turn, seq, offset = AUDIO_HEADER.unpack_from(packet)
    if magic != MAGIC:
        raise ValueError("Unsupported audio packet version")
    return str(uuid.UUID(bytes=turn)), seq, offset, packet[AUDIO_HEADER.size :]


class Command(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal[
        "text",
        "start",
        "commit",
        "stop",
        "mute",
        "ack",
        "profile",
        "language",
        "character",
        "pressure",
        "avatar",
        "speech_settings",
        "prepare",
        "transcription_action",
        "telemetry",
        "codex",
    ]
    text: str = Field(default="", max_length=16000)
    mode: Literal["ptt", "handsfree"] = "ptt"
    speak: bool = True
    turn_id: str | None = None
    samples: int = Field(default=0, ge=0)
    profile: Literal["quiet", "everyday"] = "everyday"
    language: Literal["auto", "english", "korean", "japanese"] = "auto"
    character_id: str = Field(default="ari", pattern=r"^[a-z0-9_-]{1,64}$")
    pressure: Literal["normal", "warning", "critical"] = "normal"
    avatar_mode: Literal["illustration", "model"] = "illustration"
    avatar_visible: bool = True
    reduced_motion: bool = False
    speech_settings: SpeechSettings | None = None
    capture_id: uuid.UUID | None = None
    microphone_version: Literal[1, 2] = 1
    transcription_action: Literal["retry", "use_draft", "edit"] = "retry"
    metric: Literal[
        "capture_startup",
        "capture_ended",
        "playback_started",
        "release_to_playback",
        "playback_gap",
        "stop_cutoff",
    ] = "capture_startup"
    seconds: float = Field(default=0, ge=0, le=3600)
    codex: CodexCommand | None = None


class MemoryInput(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    approved: bool = False


class PersonaInput(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    persona: str = Field(min_length=1, max_length=3000)


class VoiceInput(BaseModel):
    voice_id: str = Field(pattern=r"^[a-z0-9_-]{1,64}$")


class CharacterInput(PersonaInput):
    palette: Literal["teal", "plum", "copper", "midnight"] = "teal"
    use_ari_voice: bool = True
