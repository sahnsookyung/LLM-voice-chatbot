#!/usr/bin/env python3
"""Create an original reference with VoiceDesign; never runs in the chat process."""

import argparse
import json
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_audio.tts.utils import load_model

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--character", type=Path, default=ROOT / "characters/ari")
    parser.add_argument(
        "--description",
        default="An adult woman with a warm, clear, gently expressive voice, a comfortable middle pitch, subtle breathiness, and a natural conversational pace. Calm, curious, and lightly playful. Speak with an original voice, without imitating any real person.",
    )
    parser.add_argument(
        "--text",
        default="Hello, I'm Ari. It's nice to have a little company. Take your time, and tell me what's on your mind today.",
    )
    args = parser.parse_args()
    model_path = ROOT / ".models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
    if not (model_path / "model.safetensors").is_file():
        raise SystemExit("Run: .venv/bin/python scripts/download_models.py design")
    manifest = args.character / "character.json"
    data = json.loads(manifest.read_text())
    mx.random.seed(42)
    mx.set_cache_limit(256 * 1024 * 1024)
    model = load_model(str(model_path))
    chunks = [
        np.asarray(result.audio).reshape(-1)
        for result in model.generate(
            text=args.text,
            instruct=args.description,
            lang_code="english",
            max_tokens=300,
            verbose=False,
        )
    ]
    audio = np.concatenate(chunks)
    duration = len(audio) / model.sample_rate
    if not 2 <= duration <= 20:
        raise SystemExit(f"Reference duration {duration:.1f}s is outside 2–20s; revise the text.")
    path = args.character / "voice.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(model.sample_rate)
        output.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    data["voice"] = {"audio": "voice.wav", "transcript": args.text}
    manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    (args.character / "voice-provenance.json").write_text(
        json.dumps(
            {
                "model": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
                "revision": "7d3824abff87e49756bb0f83fb5411de75d160c4",
                "seed": 42,
                "description": args.description,
                "transcript": args.text,
                "duration_seconds": duration,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved original voice reference: {path} ({duration:.2f}s)")


if __name__ == "__main__":
    main()
