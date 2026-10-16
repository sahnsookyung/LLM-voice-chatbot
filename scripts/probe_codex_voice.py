#!/usr/bin/env python3
"""Developer acceptance probe, never enables production Codex mode.

Uses the real Ari adapter and native player against an already shared host.
Explicit --text or --audio is sent to the selected Codex task/OpenAI services.
No microphone is opened. Use a dedicated test task, not an active work task.
Reports counts/timings only; it cannot attest desktop history or Computer Use.
"""

import argparse
import asyncio
import base64
import json
import sys
import tempfile
import time
import uuid
import wave
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from companion.codex_compat import probe, socket_path
from companion.codex_rpc import CodexError, CodexRPC
from companion.codex_session import CodexSession
from companion.protocol import decode_audio, microphone_packet
from companion.store import Store


async def run(args):
    readiness = await probe(ROOT / ".runtime/codex/acceptance.json")
    if not readiness["shared_host"] or not all(
        readiness["checks"].get(k) for k in ("threads", "voices", "account")
    ):
        raise SystemExit(
            "No verified shared desktop host/account. Nothing was submitted. Run check_codex.py first."
        )
    player_path = ROOT / ".build/speech-playback-probe"
    if not player_path.is_file():
        raise SystemExit(
            "Build scripts/speech_playback_probe.swift with Audio.swift and SpeechTransport.swift first."
        )
    pcm = None
    if args.audio:
        with wave.open(str(args.audio), "rb") as source:
            if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (
                16000,
                1,
                2,
            ):
                raise SystemExit("Probe input must be mono 16 kHz PCM16 WAV.")
            if source.getnframes() > 16000 * 60:
                raise SystemExit("Probe input must be at most 60 seconds.")
            pcm = source.readframes(source.getnframes())
    if not args.text and not pcm:
        raise SystemExit("Provide nonempty --text or --audio. No implicit test message is sent.")
    counts = Counter()
    played = []
    baseline_ids, live_reply_ids = set(), set()
    expected_reply_ids = set()
    native_lock = asyncio.Lock()
    player = await asyncio.create_subprocess_exec(
        str(player_path),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    session = None
    rpc = CodexRPC(socket_path())

    async def native(value):
        async with native_lock:
            player.stdin.write((json.dumps(value) + "\n").encode())
            await player.stdin.drain()

    async def event(value):
        kind = value["type"]
        counts[kind] += 1
        row = value.get("item", {})
        if (
            kind == "codex_item"
            and row.get("realtime")
            and row.get("kind") == "transcriptSegment"
            and row.get("role") == "assistant"
            and row.get("id") not in baseline_ids
        ):
            live_reply_ids.add(row["id"])
            if (
                args.expected_response
                and args.expected_response.casefold() in row.get("text", "").casefold()
            ):
                expected_reply_ids.add(row["id"])
        if kind == "codex_audio_begin":
            await native({"type": "begin", "turn": value["turn_id"]})
        elif kind == "cancel":
            await native({"type": "cancel"})
        elif kind in {"error", "codex_disconnected", "codex_audio_failed"}:
            counts["failures"] += 1

    async def output(packet):
        _, _, _, pcm = decode_audio(packet)
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.int32)
        counts["decoded_samples"] += len(samples)
        counts["decoded_non_silent_samples"] += int((np.abs(samples) > 100).sum())
        await native({"type": "audio", "data": base64.b64encode(packet).decode()})

    async def read_player():
        while line := await player.stdout.readline():
            value = json.loads(line)
            if value.get("type") == "ack" and session:
                session.audio.acknowledge(value["turn"], value["samples"])
                counts["played_ack"] += 1
            elif value.get("type") in {"audible", "non_silent_playback", "gap", "cutoff_requested"}:
                played.append(value)
            elif value.get("type") == "error":
                counts["failures"] += 1

    reader = asyncio.create_task(read_player())
    try:
        with tempfile.TemporaryDirectory(prefix="ari-codex-voice-probe-") as directory:
            store = Store(Path(directory) / "preferences.sqlite")
            try:
                store.set_setting("codex_thread", args.thread)
                await rpc.connect()
                session = CodexSession(rpc, store, event, output, validation_thread=args.thread)
                await session.open()
                baseline_ids.update(session.timeline.rows)
                capture = str(uuid.uuid4())
                await session.start_voice(capture)
                await session.voice_start
                if not session.owns_voice:
                    raise CodexError("Voice was not acknowledged. No test utterance was submitted.")
                await native({"type": "release"})
                if pcm:
                    for sequence, offset in enumerate(range(0, len(pcm), 2560)):
                        await session.input_audio(
                            microphone_packet(
                                capture, sequence, offset // 2, pcm[offset : offset + 2560]
                            )
                        )
                        await asyncio.sleep(0.08)
                    await session.input_queue.join()
                    # A queued native packet is not necessarily on the RTP track
                    # yet. Muting clears that track, so wait for its bounded tail.
                    async with asyncio.timeout(2):
                        while session.rtc and session.rtc.microphone.pending:
                            await asyncio.sleep(0.02)
                    await session.mute()
                else:
                    await session.mute()
                    await session.text(args.text)
                deadline = time.monotonic() + args.observe_seconds
                while time.monotonic() < deadline and not counts["failures"]:
                    await asyncio.sleep(0.1)
                await session.end_voice()
                await session.refresh_history()
                persisted_replies = live_reply_ids & set(session.timeline.rows)
                heard_reply = any(v["type"] == "non_silent_playback" for v in played)
                matched_reply = bool(expected_reply_ids & persisted_replies)
                passed = (
                    not counts["failures"]
                    and heard_reply
                    and bool(persisted_replies)
                    and (not args.expected_response or matched_reply)
                )
                report = {
                    "kind": "developer voice probe; not release acceptance",
                    "passed": passed,
                    "fingerprint": readiness.get("fingerprint"),
                    "input": "supplied WAV; provenance not inferred" if pcm else "explicit text",
                    "counts": dict(counts),
                    "native_timings": played,
                    "adapter_timings": session.timings.export(),
                    "live_assistant_transcripts": len(live_reply_ids),
                    "persisted_assistant_transcripts": len(persisted_replies),
                    "expected_response_matches": len(expected_reply_ids & persisted_replies),
                    "non_silent_native_playback": heard_reply,
                    "live_desktop_history": "unverified",
                    "computer_use": "unverified",
                    "human_microphone": "unverified",
                }
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "output": str(args.output),
                            "played_ack": counts["played_ack"],
                            "failures": counts["failures"],
                            "passed": passed,
                        }
                    )
                )
                return passed
            finally:
                if session:
                    await session.close()
                else:
                    await rpc.close()
                store.close()
    finally:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        player.stdin.close()
        try:
            await asyncio.wait_for(player.wait(), 3)
        except TimeoutError:
            player.kill()
            await player.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread", required=True, help="An existing dedicated Codex test task ID")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Explicit message to send to OpenAI through this task")
    source.add_argument(
        "--audio", type=Path, help="Explicit recording to send to OpenAI, 16 kHz mono PCM16 WAV"
    )
    parser.add_argument("--observe-seconds", type=int, choices=range(1, 61), default=30)
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/codex/voice-probe.json")
    parser.add_argument(
        "--expected-response", help="Check this phrase without saving transcript text"
    )
    raise SystemExit(0 if asyncio.run(run(parser.parse_args())) else 2)
