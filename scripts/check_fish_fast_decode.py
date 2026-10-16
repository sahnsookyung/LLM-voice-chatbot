"""Compare a compiled residual-codebook pass against the pinned provider loop."""

import json
import time
from pathlib import Path

import mlx.core as mx
from companion.fish_stream import make_fast_decode
from mlx_audio.tts.models.fish_qwen3_omni.fish_speech import _sample_logits
from mlx_audio.tts.utils import load_model

model = load_model(".models/fish-audio-s2-pro-8bit").model
mx.set_cache_limit(256 * 1024 * 1024)


def frame(hidden, semantic, temperature):
    cache = model.make_fast_cache()
    model.fast_forward_cached(hidden, cache)
    value = model.fast_embeddings(semantic)
    tokens = [semantic]
    for _ in range(model.num_codebooks - 1):
        logits = model.fast_forward_cached(value, cache)
        token = _sample_logits(logits, temperature=temperature, top_p=0.7, top_k=30)
        tokens.append(token)
        value = model.fast_embeddings(token)
    return mx.stack(tokens, axis=1)


compiled_frame = make_fast_decode(model)


def compiled(hidden, semantic, temperature):
    return compiled_frame(hidden, semantic, temperature, 0.7, 30)


hidden = mx.random.normal((1, model.config.audio_decoder_config.dim)).astype(mx.bfloat16)
semantic = mx.array([143], mx.int32)
rows = []
for temperature in [0.0, 0.7]:
    outputs = []
    for name, function in [("provider", frame), ("compiled", compiled)]:
        mx.random.seed(41)
        result = function(hidden, semantic, temperature)
        mx.eval(result)
        sequence = [result.tolist()]
        start = time.monotonic()
        for _ in range(50):
            result = function(hidden, semantic, temperature)
            mx.eval(result)
            sequence.append(result.tolist())
        outputs.append(sequence)
        rows.append(
            {
                "implementation": name,
                "temperature": temperature,
                "seconds_per_frame": (time.monotonic() - start) / 50,
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    assert outputs[0] == outputs[1], "Compiled sampling changed generated codebooks"
Path("reports/fish-fast-decode-check.json").write_text(json.dumps(rows, indent=2))
