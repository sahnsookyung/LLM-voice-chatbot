"""CPU-only parity check in the pinned TTS runtime; no microphone or saved audio."""

import json
from pathlib import Path

import numpy as np
from companion.resampling import StreamingResampler
from scipy.signal import resample_poly

rng = np.random.default_rng(173)
rows = []
for length in [1, 19, 2048, 10003, 44100 * 12 + 137]:
    audio = (rng.standard_normal(length) * 0.15).astype(np.float32)
    expected = resample_poly(audio, 80, 147)
    for size in [1, 127, 8192]:
        # Sample-sized packets exercise startup/tail boundaries; long input uses
        # practical packets so the test itself does not dominate validation time.
        if size == 1 and length > 10003:
            continue
        stream = StreamingResampler()
        pieces = []
        max_history = 0
        for start in range(0, length, size):
            pieces.append(stream.feed(audio[start : start + size]))
            max_history = max(max_history, len(stream.data))
        pieces.append(stream.feed([], final=True))
        actual = np.concatenate(pieces)
        error = float(np.max(np.abs(expected - actual)))
        assert len(actual) == len(expected) and error < 2e-6
        assert max_history <= 38, "Resampler history grew with utterance length"
        try:
            stream.feed([0])
            raise AssertionError("Accepted audio after final flush")
        except ValueError:
            pass
        rows.append(
            {
                "input_samples": length,
                "chunk_samples": size,
                "output_samples": len(actual),
                "max_abs_error": error,
                "max_history_samples": max_history,
            }
        )
Path("reports/speech-resampler-parity.json").write_text(json.dumps(rows, indent=2))
print(json.dumps({"passed": len(rows), "max_abs_error": max(row["max_abs_error"] for row in rows)}))
