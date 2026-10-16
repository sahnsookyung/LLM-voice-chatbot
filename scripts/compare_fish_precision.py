"""Separate local voice samples; never changes the application's selected model."""

import argparse
import json
import threading
import time
import wave
from pathlib import Path

import mlx.core as mx
from companion.fish_stream import stream_pcm
from companion.speech import fish_text
from companion.tts_worker import install_fish_hooks
from mlx_audio.tts.utils import load_model
from mlx_audio.utils import load_audio

parser = argparse.ArgumentParser()
parser.add_argument("model", type=Path)
parser.add_argument("--bits", type=int, choices=[4, 8], required=True)
args = parser.parse_args()
output = Path("reports/fish-precision-evaluation")
output.mkdir(exist_ok=True)
began = time.monotonic()
model = load_model(str(args.model))
mx.set_cache_limit(256 * 1024 * 1024)
install_fish_hooks(model, threading.Event())
reference = load_audio("characters/ari/voice.wav", sample_rate=model.sample_rate)
transcript = json.loads(Path("characters/ari/character.json").read_text())["voice"]["transcript"]
report = {
    "bits": args.bits,
    "load_seconds": time.monotonic() - began,
    "human_voice_quality_validated": False,
    "enabled_in_app": args.bits == 8,
    "cases": [],
}
weights = {}
cases = [
    ("en-neutral", "english", "neutral", "Ari and Codex finished three API tests today."),
    ("ko-neutral", "korean", "neutral", "아리와 코덱스가 오늘 API 테스트 세 개를 마쳤어요."),
    (
        "ja-neutral",
        "japanese",
        "neutral",
        "アリとコーデックスは今日、APIのテストを三つ終えました。",
    ),
    ("mixed", "auto", "neutral", "Ari, 오늘 API 테스트 세 개를 했어요. 次は日本語です。"),
    ("en-happy", "english", "happy", "You passed your exam! I'm so happy for you!"),
    ("en-concerned", "english", "concerned", "I'm sorry you're hurting. I'll stay here with you."),
]
for name, language, emotion, text in cases:
    instruction = None if language == "auto" else f"Speak naturally in {language.capitalize()}."
    start = time.monotonic()
    first = None
    frames = 0
    with wave.open(str(output / f"{args.bits}bit-{name}.wav"), "wb") as destination:
        destination.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        for pcm in stream_pcm(
            model, fish_text(text, emotion), reference, transcript, instruction, weights
        ):
            first = first if first is not None else time.monotonic() - start
            frames += len(pcm) // 2
            destination.writeframes(pcm)
    seconds = time.monotonic() - start
    row = {
        "case": name,
        "first_audio_seconds": first,
        "generation_seconds": seconds,
        "audio_seconds": frames / 24000,
        "rtf": seconds / (frames / 24000),
        "peak_gpu_memory_bytes": mx.get_peak_memory(),
    }
    report["cases"].append(row)
    print(json.dumps(row), flush=True)
    (output / f"{args.bits}bit.json").write_text(json.dumps(report, indent=2))
