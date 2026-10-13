#!/usr/bin/env python3
"""Two real hands-free turns from the included synthetic reference. No microphone."""

import asyncio
import audioop
import json
import time
import wave
from pathlib import Path

from companion.adapters import LocalDialogue, LocalSpeech
from companion.config import Config
from companion.protocol import decode_audio
from companion.session import Session
from companion.store import Store

ROOT = Path(__file__).resolve().parents[1]


async def main():
    with wave.open(str(ROOT / "characters/ari/voice.wav")) as source:
        pcm = source.readframes(source.getnframes())
        rate = source.getframerate()
    if rate != 16000:
        pcm, _ = audioop.ratecv(pcm, 2, 1, rate, 16000, None)
    data = ROOT / ".runtime/pipeline-test"
    config = Config(root=ROOT, data=data, token="pipeline-" + ("x" * 64))
    store = Store(data / "history.sqlite")
    store.set_setting("profile", "quiet")
    queue = asyncio.Queue(maxsize=16)
    events = []
    first_audio = {}

    async def send(event):
        events.append({**event, "received_at": time.monotonic()})

    async def audio(packet):
        decoded = decode_audio(packet)
        first_audio.setdefault(decoded[0], time.monotonic())
        await queue.put(decoded)

    session = Session(config, store, send, audio, LocalDialogue(config), LocalSpeech(config))

    async def player():
        while True:
            turn, _seq, offset, chunk = await queue.get()
            await asyncio.sleep(len(chunk) / 48000)
            session.acknowledge(turn, offset + len(chunk) // 2)
            queue.task_done()

    playback = asyncio.create_task(player())
    results = []
    try:
        await session.start_listening("handsfree")
        for index in range(2):
            events.clear()
            first_audio.clear()
            start = time.monotonic()
            speech_end = None
            for offset in range(0, len(pcm), 2048):
                chunk = pcm[offset : offset + 2048]
                await session.input_audio(chunk)
                await asyncio.sleep(len(chunk) / 32000)
            speech_end = time.monotonic()
            for _ in range(48):
                await session.input_audio(bytes(2048))
                await asyncio.sleep(0.064)
            await asyncio.wait_for(session.audio_queue.join(), 10)
            # A sentence-boundary pause can legitimately start an earlier reply,
            # then interrupt it when the reference continues. Await the final turn.
            async with asyncio.timeout(90):
                while not session.task:
                    if any(e["type"] == "error" for e in events):
                        break
                    await asyncio.sleep(0.05)
                if session.task:
                    await session.task
            latest_audio = first_audio.get(session.turn_id)
            last_turn = next((e for e in reversed(events) if e["type"] == "turn"), None)
            row = {
                "turn": index + 1,
                "recognition": [e["user_text"] for e in events if e["type"] == "turn"],
                "reply": session.generated,
                "turn_ready_after_speech_s": last_turn["received_at"] - speech_end
                if last_turn
                else None,
                "speech_end_to_first_pcm_s": latest_audio - speech_end if latest_audio else None,
                "elapsed_s": time.monotonic() - start,
                "errors": [e["message"] for e in events if e["type"] == "error"],
            }
            results.append(row)
            print(json.dumps(row), flush=True)
            if row["errors"]:
                break
    finally:
        playback.cancel()
        await asyncio.gather(playback, return_exceptions=True)
        await session.close()
        store.close()
    path = ROOT / "reports/local-voice-pipeline.json"
    path.write_text(json.dumps(results, indent=2))
    return int(
        any(
            r["errors"] or not r["recognition"] or r["speech_end_to_first_pcm_s"] is None
            for r in results
        )
    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
