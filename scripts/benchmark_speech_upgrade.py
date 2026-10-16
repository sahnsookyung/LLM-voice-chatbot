#!/usr/bin/env python3
"""Real local ASR→LLM→Fish→AVAudioPlayerNode benchmark, no live microphone.

Run with backend Python. The 30 synthetic conversational turns exercise actual
playback; they do NOT establish accuracy on human speech or microphone latency.
Only timings, configuration and aggregate error counts are exported.
"""

import argparse
import asyncio
import audioop
import base64
import json
import re
import time
import unicodedata
import wave
from pathlib import Path

from companion.adapters import LocalDialogue, LocalSpeech
from companion.config import Config
from companion.final_asr import REVISION
from companion.session import Session
from companion.store import Store
from companion.voices import resolve_voice

ROOT = Path(__file__).resolve().parents[1]


def tokens(text, language):
    value = unicodedata.normalize("NFKC", text).casefold()
    if language == "english":
        return re.findall(r"[^\W_]+(?:'[^\W_]+)?", value)
    if language == "mixed":
        return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?|[가-힣ぁ-んァ-ン一-龯]", value)
    return [c for c in value if c.isalnum()]


def error_count(reference, hypothesis, language):
    expected, actual = tokens(reference, language), tokens(hypothesis, language)
    previous = list(range(len(actual) + 1))
    for i, word in enumerate(expected, 1):
        row = [i]
        for j, other in enumerate(actual, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (word != other)))
        previous = row
    return {"errors": previous[-1], "reference_tokens": len(expected)}


async def main(turns, target):
    data = ROOT / ".runtime/speech-benchmark"
    data.mkdir(exist_ok=True)
    config = Config(root=ROOT, data=data, token="benchmark-" + "x" * 64)
    store = Store(data / "history.sqlite")
    store.set_setting("profile", "quiet")
    speech = LocalSpeech(config)
    events, native = [], []
    probe = await asyncio.create_subprocess_exec(
        str(ROOT / ".build/speech-playback-probe"),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )

    async def tell(value):
        probe.stdin.write((json.dumps(value) + "\n").encode())
        await probe.stdin.drain()

    async def send(event):
        events.append(event)
        if event["type"] == "turn":
            await tell({"type": "begin", "turn": event["turn_id"]})
        elif event["type"] == "cancel":
            await tell({"type": "cancel"})

    async def audio(packet):
        await tell({"type": "audio", "data": base64.b64encode(packet).decode()})

    session = Session(config, store, send, audio, LocalDialogue(config), speech)

    async def acknowledgments():
        while line := await probe.stdout.readline():
            event = json.loads(line)
            if event["type"] == "ack":
                session.acknowledge(event["turn"], event["samples"])
            else:
                native.append(event)

    reader = asyncio.create_task(acknowledgments())
    cases = []
    fixture_report = json.loads((ROOT / "reports/local-fish-audio/report.json").read_text())
    for name in ("original-en", "bright-ko", "soft-ja"):
        case = next(row for row in fixture_report["cases"] if row["name"] == name)
        with wave.open(str(ROOT / f"reports/local-fish-audio/{name}.wav")) as source:
            pcm = audioop.ratecv(
                source.readframes(source.getnframes()), 2, 1, source.getframerate(), 16000, None
            )[0]
        cases.append((case["language"], case["text"], pcm))
    report = {
        "input": "synthetic fixtures; actual local playback; no live microphone",
        "profile": "quiet",
        "fish_precision": 8,
        "final_asr_revision": REVISION,
        "runs": [],
        "human_accuracy_validated": False,
        "fish_pre_warmed_by_fixture_generation": True,
    }
    try:
        start = time.monotonic()
        await session.recognizer.prepare(320)
        await session.final_recognizer.prepare()
        await speech.prepare()
        await session.dialogue.prepare("quiet")
        report["sequential_load_seconds"] = time.monotonic() - start
        voice = resolve_voice(config, "ari", "character")
        speech.streaming = True
        mixed = "Ari, 오늘 Codex에서 API 테스트 세 개를 했어요. 次は日本語で話して。"
        parts = [
            part
            async for part in speech.generate(mixed, "auto", voice["path"], voice["transcript"])
        ]
        mixed_pcm = audioop.ratecv(b"".join(parts), 2, 1, 24000, 16000, None)[0]
        cases.append(("mixed", mixed, mixed_pcm))
        for index in range(turns + 3):
            # Three compatibility runs provide the whole-segment comparator.
            streaming = index >= 3
            language, reference, pcm = cases[index % len(cases)]
            session.conversation = store.create_conversation()
            session.settings.fish_streaming = streaming
            events.clear()
            native.clear()
            session.timings.records.clear()
            await session.start_listening("ptt")
            await asyncio.wait_for(session.recognition_ready.wait(), 10)
            for offset in range(0, len(pcm), 2048):
                await session.input_audio(pcm[offset : offset + 2048])
            await session.audio_queue.join()
            await tell({"type": "release"})
            start = time.monotonic()
            await session.request_commit()
            await asyncio.wait_for(session.final_task, 40)
            if session.task:
                await asyncio.wait_for(session.task, 90)
            final = next(
                (e.get("text", "") for e in reversed(events) if e.get("status") == "final"), ""
            )
            row = {
                "index": index,
                "streaming": streaming,
                "first_turn_for_mode": index in {0, 3},
                "language": language,
                "input_seconds": len(pcm) / 32000,
                "turn_seconds": time.monotonic() - start,
                "fast_errors": error_count(reference, session.draft, language),
                "accurate_errors": error_count(reference, final, language),
                "stages": list(session.timings.records),
                "native": list(native),
                "failed": any(e["type"] == "error" or e.get("status") == "failed" for e in events),
            }
            report["runs"].append(row)
            target.write_text(json.dumps(report, indent=2))
            print(
                json.dumps({k: v for k, v in row.items() if k not in {"stages", "native"}}),
                flush=True,
            )
            if row["failed"]:
                print("A pipeline failure occurred; see local worker diagnostics.", flush=True)
                break
    finally:
        await session.close()
        await tell({"type": "quit"})
        await asyncio.wait_for(probe.wait(), 5)
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        store.close()
        target.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=30)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/speech-upgrade-benchmark.json"
    )
    args = parser.parse_args()
    asyncio.run(main(args.turns, args.output))
