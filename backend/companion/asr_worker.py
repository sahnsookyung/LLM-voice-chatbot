"""Private stdin/stdout ASR protocol; audio stays in RAM and is released per request."""

import base64
import contextlib
import json
import sys


def main():
    output = sys.stdout

    def emit(value):
        output.write(json.dumps(value) + "\n")
        output.flush()

    with contextlib.redirect_stdout(sys.stderr):
        import mlx.core as mx
        import numpy as np
        from mlx_audio.stt.utils import load_model

        try:
            model = load_model(sys.argv[1])
            mx.set_cache_limit(128 * 1024 * 1024)
            emit({"type": "ready"})
        except Exception:  # noqa: BLE001 — contain provider failures without private text
            emit({"type": "error"})
            return
        for line in sys.stdin:
            pcm = audio = request = result = None
            try:
                request = json.loads(line)
                pcm = base64.b64decode(request.pop("pcm"), validate=True)
                if len(pcm) % 2 or not 0 < len(pcm) <= 60 * 32000:
                    raise ValueError("Invalid recording length")
                audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
                pcm = None
                language = request["language"]
                result = model.generate(
                    audio,
                    language=None if language == "auto" else language.capitalize(),
                    temperature=0,
                    max_tokens=2048,
                    verbose=False,
                    hotwords=request["vocabulary"],
                )
                emit({"type": "final", "text": result.text})
            except Exception as error:  # noqa: BLE001 — worker protocol boundary
                print(f"Final recognition failed: {type(error).__name__}", file=sys.stderr)
                emit({"type": "error"})
            finally:
                del pcm
                audio = request = result = line = None
                mx.clear_cache()


if __name__ == "__main__":
    main()
