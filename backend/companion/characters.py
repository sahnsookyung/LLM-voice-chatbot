import shutil
import stat
import struct
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .text import EXPRESSIONS


class VoiceReference(BaseModel):
    audio: str
    transcript: str = Field(min_length=1, max_length=3000)


class Layer(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str = Field(pattern=r"^[a-z0-9_-]{1,40}$")
    image: str
    x: float = 0
    y: float = 0
    z: int = 0
    pivot_x: float = 0
    pivot_y: float = 0
    role: str = Field(default="static", pattern=r"^(static|head|hair|eyes|mouth|brows|overlay)$")
    expression: str | None = None
    openness: float | None = Field(default=None, ge=0, le=1)


class Character(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = Field(default=1, ge=1, le=1)
    id: str = Field(pattern=r"^[a-z0-9_-]{1,64}$")
    name: str = Field(min_length=1, max_length=80)
    persona: str = Field(min_length=1, max_length=3000)
    attribution: str = Field(min_length=1, max_length=3000)
    width: int = Field(default=800, ge=64, le=2048)
    height: int = Field(default=1000, ge=64, le=2048)
    voice: VoiceReference | None = None
    layers: list[Layer] = Field(default_factory=list, max_length=100)
    procedural: bool = False
    palette: Literal["teal", "plum", "copper", "midnight"] = "teal"


def safe_asset(root: Path, name: str) -> Path:
    part = PurePosixPath(name)
    if part.is_absolute() or ".." in part.parts or "\\" in name:
        raise ValueError("Unsafe asset path")
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        raise ValueError("Missing or unsafe character asset")
    return target


def validate_character(root: Path) -> Character:
    manifest = safe_asset(root, "character.json")
    if manifest.stat().st_size > 64 * 1024:
        raise ValueError("Character manifest is too large")
    character = Character.model_validate_json(manifest.read_text())
    decoded_bytes = 0
    unique_images = set()
    ids = set()
    for layer in character.layers:
        if layer.id in ids:
            raise ValueError("Duplicate layer identifier")
        ids.add(layer.id)
        if layer.expression and layer.expression not in EXPRESSIONS:
            raise ValueError("Unsupported expression")
        asset = safe_asset(root, layer.image)
        if asset.suffix.lower() != ".png":
            raise ValueError("Imported layers must be PNG images")
        if asset not in unique_images:
            unique_images.add(asset)
            with asset.open("rb") as stream:
                header = stream.read(24)
            if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
                raise ValueError("Invalid PNG")
            width, height = struct.unpack(">II", header[16:24])
            if not 1 <= width <= 4096 or not 1 <= height <= 4096:
                raise ValueError("Image dimensions exceed the character budget")
            decoded_bytes += width * height * 4
    if decoded_bytes > 64 * 1024 * 1024:
        raise ValueError("Character textures exceed 64 MiB decoded")
    if not character.layers and not character.procedural:
        raise ValueError("Character requires layers")
    if character.voice:
        import wave

        voice = safe_asset(root, character.voice.audio)
        with wave.open(str(voice)) as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise ValueError("Voice reference must be mono PCM16 WAV")
            if not 2 <= wav.getnframes() / wav.getframerate() <= 20:
                raise ValueError("Voice reference must contain 2–20 seconds")
    return character


def import_bundle(archive: Path, destination: Path) -> Character:
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination) as temp:
        root = Path(temp)
        with zipfile.ZipFile(archive) as bundle:
            entries = bundle.infolist()
            if len(entries) > 150 or sum(i.file_size for i in entries) > 128 * 1024 * 1024:
                raise ValueError("Character bundle is too large")
            names = set()
            for entry in entries:
                path = PurePosixPath(entry.filename)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or "\\" in entry.filename
                    or stat.S_ISLNK(entry.external_attr >> 16)
                    or entry.filename in names
                ):
                    raise ValueError("Unsafe character archive")
                names.add(entry.filename)
                if not entry.is_dir() and path.suffix.lower() not in {
                    ".json",
                    ".png",
                    ".wav",
                    ".txt",
                }:
                    raise ValueError("Character bundles may contain only data and media")
                bundle.extract(entry, root)
        character = validate_character(root)
        target = destination / character.id
        if target.exists() or character.id == "ari":
            raise ValueError("A character with this identifier already exists")
        shutil.copytree(root, target)
        return character


def character_path(config, identifier: str) -> Path:
    return (
        config.root / "characters" / "ari"
        if identifier == "ari"
        else config.data / "characters" / identifier
    )


def load_character(config, identifier: str) -> Character:
    return validate_character(character_path(config, identifier))
