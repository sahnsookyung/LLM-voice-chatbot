#!/usr/bin/env python3
"""Generate small reusable original references with the installed design model."""

import json
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_audio.tts.utils import load_model

ROOT = Path(__file__).resolve().parents[1]
PRESETS = [
    {
        "id": "bright",
        "name": "Bright",
        "description": "A slightly higher, clear voice with a cheerful, playful lift.",
        "language": "english",
        "transcript": "Hi, I'm Ari! It's lovely to see you. Tell me about your day. I'm listening.",
        "seed": 72,
        "instruct": "An original adult woman in her twenties, with a sweet, bright, slightly high-pitched soprano speaking voice. Cute and lightly playful, clear and natural, with smiling warmth and gentle melodic intonation. Relaxed conversational pace, crisp consonants, smooth vowels. Expressive but not exaggerated; no whisper, no rasp, no baby talk. Natural fluent English. Do not imitate a real person.",
    },
    {
        "id": "soft",
        "name": "Soft",
        "description": "An airy, gentle higher voice with a soft, affectionate tone.",
        "language": "japanese",
        "transcript": "こんにちは、アリです。会えてうれしいです。今日はどんな一日でしたか？ゆっくり聞かせてくださいね。",
        "seed": 108,
        "instruct": "An original adult Japanese woman in her twenties. A clear, sweet, gently high-pitched voice, soft and cute with a little airy warmth, like an approachable adult anime heroine in a calm everyday conversation. Light melodic intonation, delicate but intelligible consonants, natural Japanese rhythm. Warm and softly playful, never childish or squeaky. Normal conversational volume, no whisper, no exaggerated acting. Do not imitate any real person.",
    },
]


def main():
    mx.set_cache_limit(256 * 1024 * 1024)
    model = load_model(str(ROOT / ".models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"))
    for preset in PRESETS:
        folder = ROOT / "voices" / preset["id"]
        if (folder / "voice.json").exists():
            print("Already generated:", preset["id"], flush=True)
            continue
        mx.random.seed(preset["seed"])
        chunks = [
            np.asarray(r.audio).reshape(-1)
            for r in model.generate(
                text=preset["transcript"],
                instruct=preset["instruct"],
                lang_code=preset["language"],
                max_tokens=300,
                verbose=False,
            )
        ]
        audio = np.concatenate(chunks)
        duration = len(audio) / model.sample_rate
        if not 2 <= duration <= 20:
            raise ValueError(f"{preset['id']}: reference must be 2–20 seconds, got {duration:.2f}")
        folder.mkdir(parents=True, exist_ok=True)
        with wave.open(str(folder / "voice.wav"), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(model.sample_rate)
            output.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
        value = {
            **preset,
            "audio": "voice.wav",
            "duration_seconds": duration,
            "model": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
            "revision": "7d3824abff87e49756bb0f83fb5411de75d160c4",
        }
        (folder / "voice.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        print(f"Saved {preset['name']}: {duration:.2f}s", flush=True)
    mx.synchronize()


if __name__ == "__main__":
    main()
