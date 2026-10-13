#!/usr/bin/env python3
"""Offline Fish acceptance: three languages/voices, delivery, cancellation, WAV evidence.

Run with PYTHONPATH=backend .venv/bin/python scripts/check_fish_speech.py.
Uses only bundled synthetic references; never opens a microphone.
"""

import argparse
import asyncio
import audioop
import json
import time
import wave
from pathlib import Path

import numpy as np
import psutil
from companion.adapters import LocalSpeech, NemoRecognition
from companion.config import Config
from companion.voices import resolve_voice

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("original-en", "character", "english", "neutral", "Hello, I'm Ari. It's good to see you."),
    ("bright-en", "bright", "english", "happy", "Hi! I'm happy you're here."),
    ("bright-ko", "bright", "korean", "concerned", "많이 힘들었겠어요. 천천히 이야기해 주세요."),
    ("soft-ja", "soft", "japanese", "happy", "こんにちは、アリです。今日はどんな一日でしたか？"),
    ("bright-laugh", "bright", "english", "happy", "Haha!"),
    ("bright-sad", "bright", "english", "concerned", "I'm sorry. That sounds really hard."),
    ("soft-en", "soft", "english", "neutral", "Hello, I'm Ari. It's good to see you."),
]


async def main(transcribe=False):
    output = ROOT / "reports/local-fish-audio"
    output.mkdir(parents=True, exist_ok=True)
    speech = LocalSpeech(
        Config(root=ROOT, data=ROOT / ".runtime/fish-check", token="fish-check-" + "x" * 64)
    )
    report = {"model": "fish-audio-s2-pro-8bit", "cases": []}
    try:
        start = time.perf_counter()
        await speech.prepare()
        report["load_seconds"] = time.perf_counter() - start
        print(f"Loaded in {report['load_seconds']:.2f}s", flush=True)
        for name, voice_id, language, emotion, text in CASES:
            voice = resolve_voice(speech.config, "ari", voice_id)
            start, first, chunks, peak = time.perf_counter(), None, [], [0]

            async def sample_rss(peak=peak):
                while True:
                    peak[0] = max(peak[0], psutil.Process(speech.process.pid).memory_info().rss)
                    await asyncio.sleep(0.1)

            sampler = asyncio.create_task(sample_rss())
            try:
                async for chunk in speech.generate(
                    text, language, voice["path"], voice["transcript"], emotion
                ):
                    first = first if first is not None else time.perf_counter() - start
                    chunks.append(chunk)
            finally:
                sampler.cancel()
                await asyncio.gather(sampler, return_exceptions=True)
            elapsed = time.perf_counter() - start
            pcm = b"".join(chunks)
            samples = np.frombuffer(pcm, dtype="<i2").astype(float) / 32768
            assert len(samples) > 2400 and np.sqrt(np.mean(samples**2)) > 0.002
            assert max(map(len, chunks)) <= 15360
            with wave.open(str(output / f"{name}.wav"), "wb") as wav:
                wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                wav.writeframes(pcm)
            result = {
                "name": name,
                "language": language,
                "expression": emotion,
                "text": text,
                "first_pcm_seconds": first,
                "generation_seconds": elapsed,
                "audio_seconds": len(samples) / 24000,
                "peak_worker_rss_gb": peak[0] / 1e9,
                "mlx_peak_gb": speech.peak_memory_bytes / 1e9,
            }
            report["cases"].append(result)
            (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(json.dumps(result, ensure_ascii=False), flush=True)
        voice = resolve_voice(speech.config, "ari", "bright")

        async def long_reply():
            async for _ in speech.generate(
                "Let me tell you a long story about a journey across the mountains. " * 3,
                "english",
                voice["path"],
                voice["transcript"],
            ):
                pass

        task = asyncio.create_task(long_reply())
        await asyncio.sleep(1)
        start, pid = time.perf_counter(), speech.process.pid
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await speech.cancel()
        report["cancel_seconds"] = time.perf_counter() - start
        report["cancel_kept_worker"] = speech.process is not None and speech.process.pid == pid
        pcm = [
            part
            async for part in speech.generate(
                "Hello again!", "english", voice["path"], voice["transcript"], "happy"
            )
        ]
        assert sum(map(len, pcm)) > 4800
        report["post_cancel_recovery"] = True
    finally:
        await speech.close()
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if transcribe:

        async def callback(*args, **kwargs):
            pass

        recognition = NemoRecognition(speech.config, callback)
        try:
            for result in report["cases"]:
                if result["name"].endswith("laugh"):
                    continue  # ASR is not an evaluator for a nonverbal chuckle.
                await recognition.start(result["language"])
                with wave.open(str(output / f"{result['name']}.wav")) as wav:
                    pcm, _ = audioop.ratecv(
                        wav.readframes(wav.getnframes()), 2, 1, 24000, 16000, None
                    )
                for offset in range(0, len(pcm), 32000):
                    await recognition.feed(pcm[offset : offset + 32000])
                await recognition.feed(bytes(16000))
                result["recognized"] = await recognition.commit()
                print(result["name"], result["recognized"], flush=True)
                await recognition.disconnect()
        finally:
            await recognition.close()
            (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcribe", action="store_true")
    asyncio.run(main(parser.parse_args().transcribe))
