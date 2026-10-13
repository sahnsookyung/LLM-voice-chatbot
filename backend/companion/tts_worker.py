"""Isolated MLX worker. stdout is a JSON protocol; diagnostics go to stderr."""

import base64
import contextlib
import json
import os
import queue
import sys
import threading
import traceback

from .speech import fish_text, pcm_chunks


class SpeechCancelled(Exception):
    pass


def install_fish_hooks(model, cancelled):
    """Bound reference encoding to one voice and make the pinned AR loop stoppable."""
    reference_cache = {}
    original_reference = model._prepare_reference_prompt
    original_sample = model._sample_semantic

    def check_cancel():
        if cancelled.is_set():
            raise SpeechCancelled

    def cached_reference(audio, transcript):
        check_cancel()
        key = id(audio), transcript
        if key not in reference_cache:
            reference_cache.clear()
            reference_cache[key] = original_reference(audio, transcript)
        check_cancel()
        return reference_cache[key]

    def cancellable_sample(*args, **kwargs):
        check_cancel()
        return original_sample(*args, **kwargs)

    model._prepare_reference_prompt = cached_reference
    model._sample_semantic = cancellable_sample


def main():
    output = sys.stdout
    commands = queue.Queue(maxsize=2)
    cancelled = threading.Event()

    def emit(data):
        output.write(json.dumps(data) + "\n")
        output.flush()

    def read():
        for line in sys.stdin:
            try:
                command = json.loads(line)
                if command.get("type") == "cancel":
                    cancelled.set()
                else:
                    cancelled.clear()
                    commands.put(command)
            except (ValueError, KeyError):
                cancelled.set()
        cancelled.set()
        commands.put(None)

    with contextlib.redirect_stdout(sys.stderr):
        try:
            import mlx.core as mx
            from mlx_audio.tts.utils import load_model
            from mlx_audio.utils import load_audio

            model = load_model(sys.argv[1])
            mx.set_cache_limit(256 * 1024 * 1024)
            install_fish_hooks(model, cancelled)
        except Exception:  # noqa: BLE001 — isolate provider failure at the service boundary
            traceback.print_exc()
            emit(
                {
                    "type": "error",
                    "message": "Unable to load MLX speech model; check runtime and model files.",
                }
            )
            return
        emit({"type": "ready", "engine": "Fish Audio S2 Pro", "sample_rate": 24000})
        threading.Thread(target=read, daemon=True).start()
        references = {}
        while (command := commands.get()) is not None:
            try:
                path = command["reference"]
                stat = os.stat(path)
                key = path, stat.st_mtime_ns, stat.st_size
                if key not in references:
                    # Load before releasing the previous array: its id is the
                    # reference-token cache key and must not be immediately reused.
                    audio = load_audio(path, sample_rate=model.sample_rate)
                    references = {key: audio}
                text = fish_text(command["text"], command.get("expression", "neutral"))
                if not text or cancelled.is_set():
                    emit({"type": "done"})
                    continue
                results = model.generate(
                    text=text,
                    ref_audio=references[key],
                    ref_text=command["transcript"],
                    # Explicit language helps keep a Japanese reference from
                    # pulling an English reply toward its original pronunciation.
                    instruct={
                        "english": "Speak naturally in English.",
                        "korean": "Speak naturally in Korean.",
                        "japanese": "Speak naturally in Japanese.",
                    }.get(command["language"]),
                    stream=False,
                    max_tokens=512,
                    verbose=False,
                )
                try:
                    for result in results:
                        if cancelled.is_set():
                            break
                        for pcm in pcm_chunks(result.audio, result.sample_rate):
                            if cancelled.is_set():
                                break
                            emit({"type": "audio", "pcm": base64.b64encode(pcm).decode()})
                finally:
                    results.close()
                mx.synchronize()
                emit({"type": "done", "peak_memory_bytes": mx.get_peak_memory()})
            except SpeechCancelled:
                mx.synchronize()
                emit({"type": "done"})
            except Exception as error:  # noqa: BLE001 — isolate provider failure at the service boundary
                # Provider exception messages may contain the user's utterance.
                # Keep stack locations and exception type, not private speech text.
                traceback.print_tb(error.__traceback__)
                print(f"Fish generation failed: {type(error).__name__}", file=sys.stderr)
                emit(
                    {
                        "type": "error",
                        "message": "Local speech generation failed. Check the voice reference and model installation.",
                    }
                )


if __name__ == "__main__":
    main()
