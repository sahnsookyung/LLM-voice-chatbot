import struct
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

VERSION = 1
# Audio packets: magic, turn UUID, sequence, sample offset, PCM16 little-endian payload.
AUDIO_HEADER = struct.Struct("<4s16sIQ")
MAGIC = b"LVA1"


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
