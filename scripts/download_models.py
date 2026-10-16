#!/usr/bin/env python3
"""Download only pinned model revisions; generation runs offline afterward."""

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "voice-4bit-evaluation": (
        "majentik/fishaudio-s2-pro-MLX-4bit",
        "dda28aa6e33fe946826792e3feb643e9c463b75f",
    ),
    "final-asr": ("mlx-community/Qwen3-ASR-1.7B-8bit", "a8379a2e2f9e313c9292cdf1af4055ab56d50d55"),
    "quiet": ("mlx-community/Qwen3.5-4B-4bit", "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c"),
    "everyday": ("mlx-community/Qwen3.5-9B-4bit", "8b2b98c00a6b4d291155e4890773ca8f769aee53"),
    "voice": (
        "mlx-community/fish-audio-s2-pro-8bit",
        "c8d4481b3f7cbfe64d855c8b7cda7739502fc3ff",
    ),
    "design": (
        "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
        "7d3824abff87e49756bb0f83fb5411de75d160c4",
    ),
    "asr": ("nvidia/nemotron-3.5-asr-streaming-0.6b", "ea30d66debe3740a08b573244286791d423d6b3e"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="*", default=["quiet", "asr", "final-asr", "voice", "everyday"])
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
        elif name == "voice-4bit-evaluation":
            # Evaluation only: use Ari's existing full-precision codec, never the
            # third-party pickle. This variant is not an application preference.
            codec = dest / "fish-audio-s2-pro-8bit/codec.safetensors"
            if not codec.is_file():
                raise SystemExit("Download the default voice model before its 4-bit comparison.")
            folder = dest / "fish-s2-pro-4bit-evaluation"
            snapshot_download(
                repo,
                revision=revision,
                local_dir=folder,
                max_workers=2,
                allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.md", "NOTICE"],
            )
            link = folder / "codec.safetensors"
            if not link.exists():
                link.symlink_to("../fish-audio-s2-pro-8bit/codec.safetensors")
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
