"""Speech-only delivery controls; these never enter chat text or history."""

import re

FISH_MODEL = "fish-audio-s2-pro-8bit"
PCM_RATE = 24000
PCM_CHUNK_SAMPLES = 7680

DELIVERY = {
    "neutral": "",
    "happy": "[happy]",
    "curious": "[curious]",
    "concerned": "[soft, empathetic tone]",
    "surprised": "[surprised]",
    "annoyed": "[annoyed]",
    "embarrassed": "[shy]",
    "sleepy": "[sleepy]",
}


def fish_text(text, expression="neutral"):
    # Do not let quoted/user-supplied Fish control or speaker tags become acting
    # instructions. Only the app's bounded mood and chuckle mapping adds tags.
    plain = re.sub(r"<\|[^<>]*\|>", "", text)
    plain = plain.replace("[", "").replace("]", "").strip()
    if not plain:
        return ""
    # Only a complete, happy laughter interjection becomes a vocal event. A
    # quoted laugh, serious reply, or sentence discussing laughter stays speech.
    if expression == "happy" and re.fullmatch(
        r"(?:(?:ha[ -]*){2,}|(?:he[ -]*){2,}|(?:하|호|헤){2,}|(?:は|ハ|ふ|フ|へ){2,}[っッ]?)[.!！。~～…]*",
        plain,
        re.IGNORECASE,
    ):
        return "[chuckle]"
    return (DELIVERY.get(expression, "") + " " + plain).strip()


def fish_model_ready(path):
    from .config import model_ready

    return model_ready(path) and all(
        (path / name).is_file()
        for name in ("codec.safetensors", "tokenizer.json", "tokenizer_config.json")
    )


def pcm_chunks(audio, sample_rate):
    """Resample each complete Fish segment before splitting into wire packets.

    Fish's pinned MLX provider returns complete segments, not streaming codec
    frames. Resampling once avoids filter resets/clicks at transport boundaries.
    """
    import math

    import numpy as np
    from scipy.signal import resample_poly

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not len(samples):
        return
    if sample_rate <= 0 or not np.isfinite(samples).all():
        raise ValueError("Invalid speech samples or sample rate")
    if sample_rate != PCM_RATE:
        divisor = math.gcd(sample_rate, PCM_RATE)
        samples = resample_poly(samples, PCM_RATE // divisor, sample_rate // divisor)
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
    for offset in range(0, len(pcm), PCM_CHUNK_SAMPLES * 2):
        yield pcm[offset : offset + PCM_CHUNK_SAMPLES * 2]
