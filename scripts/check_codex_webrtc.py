"""Probe host-owned WebRTC with existing sign-in; never opens a microphone.

Only session signaling and silent RTP are sent. No credentials, SDP, audio,
transcripts or provider response bodies are written to the diagnostic report.
Run against an explicitly selected existing task with no active voice owner.
"""

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

from aiortc import AudioStreamTrack, RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from companion.codex_compat import probe, socket_path
from companion.codex_rpc import CodexError, CodexRPC


async def run(thread, seconds, text=None):
    rpc = CodexRPC(socket_path())
    peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    peer.addTrack(AudioStreamTrack())  # aiortc's built-in silent source; no device access.
    channel = peer.createDataChannel("oai-events")
    counts, errors, states = Counter(), set(), []
    opened, terminal = asyncio.Event(), asyncio.Event()
    requested = False
    typed_input_sent = False
    account = {}
    report = {}
    started_at = time.monotonic()
    subscribed = False
    media_tasks = []
    audio_frames = 0
    non_silent_frames = 0
    event_shapes = {}
    error_terms = set()

    @peer.on("track")
    def receive_track(track):
        async def consume():
            nonlocal audio_frames, non_silent_frames
            try:
                while True:
                    frame = await track.recv()
                    audio_frames += 1
                    if abs(frame.to_ndarray().astype("float32")).max() > 100:
                        non_silent_frames += 1
            except MediaStreamError:
                counts["media_ended"] += 1
            except (OSError, RuntimeError, ValueError):
                errors.add("media_read_failed")
                terminal.set()

        media_tasks.append(asyncio.create_task(consume()))

    @peer.on("connectionstatechange")
    async def connection_changed():
        states.append(peer.connectionState)
        if peer.connectionState == "failed":
            terminal.set()

    @channel.on("open")
    def channel_open():
        opened.set()

    @channel.on("message")
    def received(data):
        try:
            value = json.loads(data)
            kind = value.get("type", "unknown")
            if isinstance(kind, str) and len(kind) < 100:
                counts["data:" + kind] += 1
                event_shapes[kind] = sorted(value.keys())
        except (ValueError, TypeError, AttributeError):
            counts["invalid_data_event"] += 1

    async def events():
        while True:
            message = await rpc.events.get()
            kind, params = message.get("method", ""), message.get("params", {})
            if params.get("threadId") != thread:
                continue
            if not kind.startswith("thread/realtime/"):
                continue
            counts[kind] += 1
            if kind == "thread/realtime/sdp":
                await peer.setRemoteDescription(RTCSessionDescription(params["sdp"], "answer"))
            elif kind == "thread/realtime/error":
                body = str(params.get("message", "")).lower()
                error_terms.update(
                    term
                    for term in (
                        "already",
                        "active",
                        "realtime",
                        "webrtc",
                        "websocket",
                        "sdp",
                        "auth",
                        "api key",
                        "timeout",
                        "failed",
                        "request",
                        "connection",
                        "reset",
                        "invalid",
                        "model",
                        "session",
                        "call",
                        "rate",
                        "limit",
                        "not found",
                        "400",
                        "401",
                        "403",
                        "429",
                        "500",
                        "502",
                        "503",
                        "speech",
                        "version",
                    )
                    if term in body
                )
                if "requires api key auth" in body:
                    errors.add("api_key_required")
                elif "401" in body or "unauthorized" in body:
                    errors.add("unauthorized")
                elif "403" in body or "forbidden" in body:
                    errors.add("forbidden")
                elif "429" in body:
                    errors.add("rate_limited")
                else:
                    errors.add("provider_error_not_retained")
                terminal.set()
            elif kind == "thread/realtime/closed":
                terminal.set()

    reader = None
    try:
        readiness = await probe(ROOT / ".runtime/codex/acceptance.json")
        if not readiness["shared_host"]:
            raise CodexError("Shared desktop attachment is unavailable")
        await rpc.connect()
        account = await rpc.call("account/read", {})
        await rpc.call("thread/read", {"threadId": thread, "includeTurns": False})
        # Resume subscribes to this existing task; it does not submit a new turn.
        await rpc.call("thread/resume", {"threadId": thread, "excludeTurns": True})
        subscribed = True
        cursor, owner, newest_position = None, None, -1
        for _ in range(500):
            page = await rpc.call(
                "thread/timeline/list",
                {"threadId": thread, "limit": 20, **({"cursor": cursor} if cursor else {})},
            )
            page_owner = page.get("activeRealtimeSessionAtPageStart")
            entries = sorted(page.get("data", []), key=lambda entry: entry["position"])
            for entry in entries:
                item = entry.get("item", {})
                if entry.get("type") == "realtime":
                    if item.get("type") == "realtimeSessionStarted":
                        page_owner = item.get("realtimeSessionId") or "unknown"
                    elif item.get("type") == "realtimeSessionClosed":
                        page_owner = None
            position = entries[-1]["position"] if entries else -1
            if position >= newest_position:
                owner, newest_position = page_owner, position
            cursor = page.get("nextCursor")
            if not cursor:
                break
        if cursor:
            raise RuntimeError("History is incomplete; cannot determine voice ownership")
        if owner:
            raise RuntimeError("Another voice session owns this task; probe refused to start")
        offer = await peer.createOffer()
        await peer.setLocalDescription(offer)
        reader = asyncio.create_task(events())
        requested = True
        await rpc.call(
            "thread/realtime/start",
            {
                "threadId": thread,
                "version": "v3",
                "outputModality": "audio",
                "transport": {"type": "webrtc", "sdp": peer.localDescription.sdp},
                "includeStartupContext": False,
                "clientManagedHandoffs": False,
                "backendReasoningStatus": False,
                "delegationAckFiller": False,
                "prompt": "This is a connection check. Stay silent unless the user speaks. Do not start tasks.",
            },
            timeout=30,
        )
        waits = [asyncio.create_task(opened.wait()), asyncio.create_task(terminal.wait())]
        try:
            await asyncio.wait(waits, timeout=seconds, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for wait in waits:
                wait.cancel()
            await asyncio.gather(*waits, return_exceptions=True)
        if reader.done():
            await reader
        if opened.is_set() and text:
            await rpc.call(
                "thread/realtime/appendText", {"threadId": thread, "role": "user", "text": text}
            )
            typed_input_sent = True
            try:
                await asyncio.wait_for(terminal.wait(), timeout=seconds)
            except TimeoutError:
                pass
        report = {
            "kind": "host-owned WebRTC signaling probe; not voice acceptance",
            "authentication_type": (account.get("account") or {}).get("type"),
            "data_channel_open": opened.is_set(),
            "connection_states": states[:],
            "events": dict(counts),
            "errors": sorted(errors),
            "error_terms": sorted(error_terms),
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
            "microphone_opened": False,
            "spoken_input_sent": False,
            "typed_input_sent": typed_input_sent,
            "received_audio_frames": audio_frames,
            "received_non_silent_frames": non_silent_frames,
            "data_event_field_names": event_shapes,
            "credentials_accessed": False,
            "raw_content_saved": False,
        }
    except (CodexError, OSError, RuntimeError, ValueError, TimeoutError):
        errors.add("setup_or_signaling_failed")
    finally:
        if requested:
            try:
                await rpc.call("thread/realtime/stop", {"threadId": thread}, timeout=5)
            except (CodexError, OSError, RuntimeError):
                errors.add("stop_not_acknowledged")
        if reader:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
        await peer.close()
        for task in media_tasks:
            task.cancel()
        await asyncio.gather(*media_tasks, return_exceptions=True)
        if subscribed:
            try:
                await rpc.call("thread/unsubscribe", {"threadId": thread}, timeout=5)
            except (CodexError, OSError, RuntimeError):
                errors.add("unsubscribe_not_acknowledged")
        await rpc.close()
    report.update(
        {
            "errors": sorted(errors),
            "events": dict(counts),
            "typed_input_sent": typed_input_sent,
            "microphone_opened": False,
            "credentials_accessed": False,
            "raw_content_saved": False,
            "received_audio_frames": audio_frames,
            "received_non_silent_frames": non_silent_frames,
        }
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread", required=True)
    parser.add_argument("--seconds", type=int, choices=range(1, 31), default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--text", help="Explicit test message to send through the existing voice task"
    )
    args = parser.parse_args()
    result = asyncio.run(run(args.thread, args.seconds, args.text))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
