"""Also run in the TTS environment: python -m unittest discover -s tests -p test_pcm_conversion.py."""

import importlib.util
import unittest

import numpy as np
from companion.speech import pcm_chunks


@unittest.skipUnless(importlib.util.find_spec("scipy"), "resampling requires the TTS runtime")
class PCMConversionTests(unittest.TestCase):
    def test_native_fish_rate_preserves_pitch_duration_and_packet_continuity(self):
        original = np.sin(2 * np.pi * 1000 * np.arange(44100) / 44100) * 0.25
        chunks = list(pcm_chunks(original, 44100))
        result = np.frombuffer(b"".join(chunks), dtype="<i2") / 32767
        self.assertEqual(len(result), 24000)
        self.assertLessEqual(max(map(len, chunks)), 15360)
        self.assertEqual(np.argmax(np.abs(np.fft.rfft(result))), 1000)
        expected = np.sin(2 * np.pi * 1000 * np.arange(24000) / 24000) * 0.25
        self.assertLess(np.max(np.abs(result[100:-100] - expected[100:-100])), 0.001)

    def test_downsampling_rejects_frequencies_above_new_nyquist(self):
        original = np.sin(2 * np.pi * 18000 * np.arange(44100) / 44100) * 0.8
        result = np.frombuffer(b"".join(pcm_chunks(original, 44100)), dtype="<i2") / 32767
        self.assertLess(np.sqrt(np.mean(result[100:-100] ** 2)), 0.001)

    def test_invalid_audio_rejected_and_empty_result_is_safe(self):
        self.assertEqual(list(pcm_chunks([], 44100)), [])
        for samples, rate in (([np.nan], 44100), ([np.inf], 44100), ([0.1], 0)):
            with self.assertRaises(ValueError):
                list(pcm_chunks(samples, rate))


if __name__ == "__main__":
    unittest.main()
