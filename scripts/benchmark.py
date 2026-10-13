#!/usr/bin/env python3
"""Exercise the real backend with clocked playback acknowledgments; never opens a mic.

This measures the server path, not speaker/AEC latency. Use --turns 100 --duration 1800
for the half-hour soak. Reports remain local and contain no user conversation data.
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import psutil
from companion.adapters import DemoDialogue, DemoSpeech, LocalDialogue, LocalSpeech
from companion.config import Config
from companion.protocol import decode_audio
from companion.session import Session
from companion.store import Store

ROOT = Path(__file__).resolve().parents[1]


async def run(args):
    data = ROOT / ".runtime/benchmark"
    data.mkdir(parents=True, exist_ok=True)
    store = Store(data / "history.sqlite")
    store.set_setting("profile", args.profile)
    results = []
    audio_queue = asyncio.Queue(maxsize=16)
    events = []

    async def event(value):
        events.append(value)

    async def audio(packet):
        await audio_queue.put(decode_audio(packet))

    config = Config(root=ROOT, data=data, token="benchmark-" + ("x" * 64), demo=args.demo)
    session = Session(
        config,
        store,
        event,
        audio,
        DemoDialogue() if args.demo else LocalDialogue(config),
        DemoSpeech() if args.demo else LocalSpeech(config),
    )

    async def play():
        while True:
            turn, _seq, offset, pcm = await audio_queue.get()
            await asyncio.sleep(len(pcm) / 48000)
            session.acknowledge(turn, offset + len(pcm) // 2)
            audio_queue.task_done()

    playback = asyncio.create_task(play())
    prompts = [
        "Say hello in one short sentence.",
        "한국어로 한 문장만 인사해 줘.",
        "日本語で一文だけ挨拶してください。",
    ]
    begin = time.monotonic()
    try:
        for index in range(args.seed_history):
            store.add_message(
                session.conversation,
                "user",
                f"Practice conversation {index}: tell me about taking a relaxing break and learning another language.",
            )
            identifier = store.add_message(session.conversation, "assistant", "")
            response = "A short walk can help you relax. You can also practice a few phrases slowly and enjoy the process without rushing."
            store.update_message(identifier, response, response, "complete")
        if args.with_recognition:
            await session.recognizer.start("auto")
        for index in range(args.turns):
            events.clear()
            start = time.monotonic()
            await session.reply(prompts[index % 3], speak=not args.text_only)
            interrupted = bool(args.interrupt_every and (index + 1) % args.interrupt_every == 0)
            cancel_seconds = None
            if interrupted:
                async with asyncio.timeout(120):
                    while not session.sent_samples and not session.task.done():
                        await asyncio.sleep(0.01)
                cancel_start = time.monotonic()
                await session.stop()
                cancel_seconds = time.monotonic() - cancel_start
            else:
                await asyncio.wait_for(session.task, 120)
            done = next((e for e in events if e["type"] == "done"), {})
            errors = [e.get("message") for e in events if e["type"] == "error"]
            row = {
                "turn": index + 1,
                "language": ["en", "ko", "ja"][index % 3],
                "first_audio_s": done.get("first_audio_seconds"),
                "elapsed_s": time.monotonic() - start,
                "errors": errors,
                "interrupted": interrupted,
                "cancel_control_s": cancel_seconds,
                "generated_text": session.generated,
                "delivered_text": session.delivered,
                "audio_seconds": session.sent_samples / 24000,
            }
            process = psutil.Process()
            row["worker_rss_bytes"] = process.memory_info().rss + sum(
                child.memory_info().rss
                for child in process.children(recursive=True)
                if child.is_running()
            )
            results.append(row)
            print(json.dumps(row), flush=True)
            wait = (index + 1) * args.duration / max(1, args.turns) - (time.monotonic() - begin)
            if wait > 0:
                await asyncio.sleep(wait)
    finally:
        playback.cancel()
        await asyncio.gather(playback, return_exceptions=True)
        await session.close()
        store.close()
    warm = [r["first_audio_s"] for r in results[1:] if r["first_audio_s"] is not None]
    report = {
        "profile": args.profile,
        "seed_history_exchanges": args.seed_history,
        "interrupt_every": args.interrupt_every,
        "requested_duration_s": args.duration,
        "demo": args.demo,
        "text_only": args.text_only,
        "asr_resident": args.with_recognition,
        "elapsed_s": time.monotonic() - begin,
        "turns": results,
        "warm_median_s": statistics.median(warm) if warm else None,
        "warm_p95_s": sorted(warm)[min(len(warm) - 1, int(len(warm) * 0.95))] if warm else None,
        "measurement": "Server reply start to first PCM chunk; client playback is clocked simulation. Excludes ASR and physical device latency.",
    }
    path = ROOT / "reports" / f"local-benchmark-{args.profile}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(path)
    return int(any(r["errors"] for r in results))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--turns", type=int, default=6)
    p.add_argument("--duration", type=float, default=0)
    p.add_argument("--profile", choices=["quiet", "everyday"], default="quiet")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--with-recognition", action="store_true")
    p.add_argument("--interrupt-every", type=int, default=0)
    p.add_argument("--seed-history", type=int, default=0)
    args = p.parse_args()
    raise SystemExit(asyncio.run(run(args)))
