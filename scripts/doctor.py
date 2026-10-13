#!/usr/bin/env python3
"""Read-only setup diagnostics. Does not silently download or activate a microphone."""

import subprocess
import sys
from pathlib import Path

from companion.config import model_ready

ROOT = Path(__file__).resolve().parents[1]
checks = {
    "Backend Python": ".venv/bin/python",
    "Dialogue runtime": ".runtime/omlx/bin/omlx",
    "Speech runtime": ".runtime/tts/bin/python",
    "Recognition runtime": ".runtime/nemo/bin/nemo-speech",
    "Quiet model": ".models/Qwen3.5-4B-4bit",
    "Everyday model": ".models/Qwen3.5-9B-4bit",
    "Recognition model": ".models/nemotron-asr.gguf",
    "Voice model": ".models/Qwen3-TTS-12Hz-0.6B-Base-bf16",
    "Original voice reference": "characters/ari/voice.wav",
    "macOS app": "build/Local Companion.app/Contents/MacOS/Companion",
}


def main():
    missing = []
    for label, path in checks.items():
        target = ROOT / path
        found = model_ready(target) if target.is_dir() else target.is_file()
        print(f"{'OK     ' if found else 'MISSING'} {label}")
        if not found:
            missing.append(label)
    result = subprocess.run(["xcrun", "metal", "--version"], capture_output=True, check=False)
    print(("OK     " if result.returncode == 0 else "MISSING") + " Metal compiler")
    if missing:
        print(
            "Run scripts/setup.sh --models, scripts/create_voice.py with the TTS Python, then scripts/build.sh."
        )
    else:
        print("All required local components are installed.")
    return 1 if missing or result.returncode else 0


if __name__ == "__main__":
    sys.exit(main())
