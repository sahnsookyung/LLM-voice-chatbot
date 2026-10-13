"""Isolated MLX worker. stdout is a JSON protocol; diagnostics go to stderr."""

import base64
import contextlib
import json
import os
import queue
import sys
import threading
import traceback


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
                    commands.put(command)
            except (ValueError, KeyError):
                cancelled.set()
        cancelled.set()
        commands.put(None)

    with contextlib.redirect_stdout(sys.stderr):
        try:
            import mlx.core as mx
            import numpy as np
            from mlx_audio.tts.utils import load_model
            from mlx_audio.utils import load_audio

            model = load_model(sys.argv[1])
            mx.set_cache_limit(256 * 1024 * 1024)
        except Exception:  # noqa: BLE001 — isolate provider failure at the service boundary
            traceback.print_exc()
            emit(
                {
                    "type": "error",
                    "message": "Unable to load MLX speech model; check runtime and model files.",
                }
            )
            return
        emit({"type": "ready"})
        threading.Thread(target=read, daemon=True).start()
        references = {}
        embeddings = {}
        original_extract = model.extract_speaker_embedding

        def cached_embedding(audio, sr=24000):
            # Hash the small reference once per cached array, not generated speech.
            key = id(audio), sr
            if key not in embeddings:
                embeddings[key] = original_extract(audio, sr)
            return embeddings[key]

        model.extract_speaker_embedding = cached_embedding
        while (command := commands.get()) is not None:
            cancelled.clear()
            try:
                path = command["reference"]
                stat = os.stat(path)
                key = path, stat.st_mtime_ns, stat.st_size
                if key not in references:
                    references.clear()
                    embeddings.clear()
                    model._icl_cache.clear()
                    references[key] = load_audio(path, sample_rate=24000)
                results = model.generate(
                    text=command["text"],
                    ref_audio=references[key],
                    ref_text=command["transcript"],
                    lang_code=command["language"],
                    stream=True,
                    streaming_interval=0.32,
                    max_tokens=512,
                    verbose=False,
                )
                try:
                    for result in results:
                        if cancelled.is_set():
                            break
                        samples = np.asarray(result.audio).reshape(-1)
                        pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
                        if pcm:
                            emit({"type": "audio", "pcm": base64.b64encode(pcm).decode()})
                finally:
                    results.close()
                mx.synchronize()
                emit({"type": "done"})
            except Exception:  # noqa: BLE001 — isolate provider failure at the service boundary
                traceback.print_exc()
                emit(
                    {
                        "type": "error",
                        "message": "Local speech generation failed. Check the voice reference and model installation.",
                    }
                )


if __name__ == "__main__":
    main()
