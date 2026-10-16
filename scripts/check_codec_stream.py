import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from companion.fish_codec_stream import CodecStream
from mlx_audio.codec.models.fish_s1_dac.fish_s1_dac import DAC
from mlx_audio.utils import load_audio

codec = DAC.from_pretrained(".models/fish-audio-s2-pro-8bit")
audio = load_audio("characters/ari/voice.wav", sample_rate=44100)
codes, lens = codec.encode(audio[None, None, :], mx.array([audio.size]))
codes = mx.concatenate([codes] * 3, axis=-1)
mx.eval(codes)
expected, length = codec.decode(codes, mx.array([codes.shape[-1]]))
mx.eval(expected)
rows = []
for size in [1, 4, 7, 16]:
    start = time.monotonic()
    stream = CodecStream(codec)
    result = mx.concatenate(
        [stream.decode(codes[:, :, i : i + size]) for i in range(0, codes.shape[-1], size)]
    )
    mx.eval(result)
    actual = np.asarray(result)
    reference = np.asarray(expected[0, 0])
    delta = actual - reference
    row = {
        "batch_frames": size,
        "frames": codes.shape[-1],
        "audio_seconds": len(actual) / 44100,
        "seconds": time.monotonic() - start,
        "max_abs_error": float(np.max(np.abs(delta))),
        "rms_error": float(np.sqrt(np.mean(delta * delta))),
        "snr_db": float(
            20 * np.log10(np.linalg.norm(reference) / max(np.linalg.norm(delta), 1e-12))
        ),
    }
    rows.append(row)
    print(json.dumps(row), flush=True)
    assert len(actual) == len(reference) and row["snr_db"] > 50 and row["max_abs_error"] < 0.005
Path("reports/speech-codec-parity.json").write_text(
    json.dumps(
        {
            "provider_revision": "94c7716212b2228f178d2f9c7619a591fd1b0b78",
            "source": "bundled synthetic reference, repeated three times",
            "cases": rows,
        },
        indent=2,
    )
)
