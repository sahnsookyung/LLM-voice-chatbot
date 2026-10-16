"""Bounded, phase-continuous version of scipy's default resample_poly filter."""

import math

import numpy as np
from scipy.signal import firwin


class StreamingResampler:
    def __init__(self, source=44100, target=24000):
        divisor = math.gcd(source, target)
        self.up, self.down = target // divisor, source // divisor
        self.half = 10 * max(self.up, self.down)
        self.filter = (
            (
                firwin(2 * self.half + 1, 1 / max(self.up, self.down), window=("kaiser", 5.0))
                * self.up
            ).astype(np.float32)
            if source != target
            else None
        )
        self.data = np.empty(0, np.float32)
        self.base = self.total = self.output = 0
        self.closed = False

    def feed(self, samples, final=False):
        if self.closed:
            raise ValueError("Resampler already flushed")
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not np.isfinite(samples).all():
            raise ValueError("Invalid audio samples")
        self.closed = final
        if self.filter is None:
            return samples
        self.data = np.concatenate([self.data, samples])
        self.total += len(samples)
        stop = (
            (self.total * self.up + self.down - 1) // self.down
            if final
            else max(0, ((self.total * self.up - 1 - self.half) // self.down) + 1)
        )
        count = stop - self.output
        if count <= 0:
            return np.empty(0, np.float32)
        centers = np.arange(self.output, stop, dtype=np.int64) * self.down + self.half
        highest = centers // self.up
        taps = np.arange((2 * self.half) // self.up + 1)
        index = highest[:, None] - taps[None, :]
        filter_index = centers[:, None] - index * self.up
        valid = (index >= 0) & (index < self.total) & (filter_index < len(self.filter))
        values = self.data[np.clip(index - self.base, 0, max(0, len(self.data) - 1))]
        weights = self.filter[np.clip(filter_index, 0, len(self.filter) - 1)]
        result = np.sum(np.where(valid, values * weights, 0), axis=1, dtype=np.float32)
        self.output = stop
        keep = max(0, (stop * self.down - self.half) // self.up)
        drop = min(len(self.data), max(0, keep - self.base))
        self.data = self.data[drop:].copy()
        self.base += drop
        return result
