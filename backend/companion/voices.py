"""Small, local reference clips shared across characters; no resident design model."""

import json
import re

from .characters import character_path, load_character, safe_asset


def voice_catalog(config, character_id):
    character = load_character(config, character_id)
    result = []
    if character.voice:
        result.append(
            {
                "id": "character",
                "name": "Original",
                "description": "This character's original voice.",
                "path": safe_asset(character_path(config, character_id), character.voice.audio),
                "transcript": character.voice.transcript,
            }
        )
    for manifest in sorted((config.root / "voices").glob("*/voice.json")):
        value = json.loads(manifest.read_text())
        if not re.fullmatch(r"[a-z0-9_-]{1,64}", value["id"]) or value["id"] == "character":
            raise ValueError("Invalid bundled voice identifier")
        result.append(
            {
                "id": value["id"],
                "name": value["name"],
                "description": value["description"],
                "path": safe_asset(manifest.parent, value["audio"]),
                "transcript": value["transcript"],
            }
        )
    return result


def resolve_voice(config, character_id, identifier):
    return next((v for v in voice_catalog(config, character_id) if v["id"] == identifier), None)
