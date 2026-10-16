"""Exercise real Fish cancellation, cache trimming and a fresh turn, without playback."""

import asyncio
import json
import time
from pathlib import Path

from companion.adapters import LocalSpeech
from companion.config import Config
from companion.voices import resolve_voice


async def main():
    config = Config(data=Path(".runtime/speech-worker-check"), token="x" * 64)
    config.data.mkdir(exist_ok=True)
    speech = LocalSpeech(config)
    speech.streaming = True
    voice = resolve_voice(config, "ari", "character")
    try:
        await speech.prepare()
        stream = speech.generate(
            "Hello. This is a local cancellation test with a longer second sentence.",
            "english",
            voice["path"],
            voice["transcript"],
        )
        first = await anext(stream)
        assert 0 < len(first) <= 3840
        start = time.monotonic()
        await stream.aclose()
        await speech.cancel()
        cancellation = time.monotonic() - start
        await speech.trim()
        count = samples = 0
        async for pcm in speech.generate(
            "Hello again!", "english", voice["path"], voice["transcript"]
        ):
            assert 0 < len(pcm) <= 3840 and len(pcm) % 2 == 0
            count += 1
            samples += len(pcm) // 2
        await speech.trim()
        report = {
            "cancel_seconds": cancellation,
            "next_turn_packets": count,
            "next_turn_samples": samples,
            "trim_acknowledged": True,
            "playback_or_microphone_opened": False,
        }
        assert count > 1 and samples > 2400
        Path("reports/speech-worker-lifecycle.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
    finally:
        await speech.close()


if __name__ == "__main__":
    asyncio.run(main())
