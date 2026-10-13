#!/usr/bin/env python3
"""Download only pinned model revisions; generation runs offline afterward."""

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "quiet": ("mlx-community/Qwen3.5-4B-4bit", "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c"),
    "everyday": ("mlx-community/Qwen3.5-9B-4bit", "8b2b98c00a6b4d291155e4890773ca8f769aee53"),
    "voice": (
        "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
        "1eccf1cb2519b5a4e8a95b5f0544f3303568164f",
    ),
    "design": (
        "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
        "7d3824abff87e49756bb0f83fb5411de75d160c4",
    ),
    "asr": ("nvidia/nemotron-3.5-asr-streaming-0.6b", "ea30d66debe3740a08b573244286791d423d6b3e"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="*", default=["quiet", "asr", "voice", "everyday"])
    a = p.parse_args()
    dest = ROOT / ".models"
    dest.mkdir(exist_ok=True)
    for name in a.models:
        repo, revision = MODELS[name]
        print(f"Downloading {name}: {repo} @ {revision}", flush=True)
        if name == "asr":
            path = hf_hub_download(
                repo,
                "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf",
                revision=revision,
                local_dir=dest / "asr",
            )
            link = dest / "nemotron-asr.gguf"
            if not link.exists():
                link.symlink_to(Path(path).relative_to(dest))
        else:
            snapshot_download(
                repo,
                revision=revision,
                local_dir=dest / repo.split("/")[-1],
                max_workers=3,
                ignore_patterns=["*.bin", "*.pt", "*.pth", ".gitattributes"],
            )
    (dest / "manifest.json").write_text(json.dumps(MODELS, indent=2))


if __name__ == "__main__":
    main()
