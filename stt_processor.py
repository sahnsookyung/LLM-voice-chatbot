import asyncio
import torch
import numpy as np
import sounddevice as sd
import tempfile
import wave
import threading
import queue
import time
from pathlib import Path
from faster_whisper import WhisperModel


class AudioProcessor:
    """Handles Voice Activity Detection and Speech Recognition"""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.chunk_duration = 3.0
        self.silence_duration = 1.5

        print("🎤 Loading Voice Activity Detection...")
        torch.set_num_threads(1)
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        self.get_speech_timestamps, _, _, _, _ = self.vad_utils

        print("🗣️ Loading Speech Recognition...")
        self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

        print("✅ Audio processor initialized!")

    def save_audio_chunk(self, audio_data, filename):
        """Save audio data to WAV file"""
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())

    def detect_speech(self, audio_data):
        """Use SileroVAD to detect if there's speech in the audio"""
        try:
            audio_tensor = torch.from_numpy(audio_data.flatten()).float()
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.vad_model,
                sampling_rate=self.sample_rate,
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=100
            )
            return len(speech_timestamps) > 0
        except Exception as e:
            print(f"VAD Error: {e}")
            return False

    def transcribe_audio(self, audio_file):
        """Transcribe audio using Whisper"""
        try:
            segments, _ = self.whisper_model.transcribe(
                audio_file,
                beam_size=1,
                language="en"
            )
            transcription = " ".join([segment.text.strip() for segment in segments])
            return transcription.strip()
        except Exception as e:
            print(f"ASR Error: {e}")
            return ""

    def record_with_vad(self, on_speech_callback, is_speaking_callback):
        """Record audio with voice activity detection"""
        print("\n🎤 Listening... (Press Ctrl+C to stop)")

        audio_buffer = []
        silence_counter = 0
        speech_detected = False
        recording = False

        def audio_callback(indata, frames, time, status):
            nonlocal audio_buffer, silence_counter, speech_detected, recording

            # Don't record while AI is speaking
            if is_speaking_callback():
                return

            audio_buffer.extend(indata[:, 0])

            # Keep buffer manageable (last 5 seconds)
            max_buffer_size = int(self.sample_rate * 5.0)
            if len(audio_buffer) > max_buffer_size:
                audio_buffer = audio_buffer[-max_buffer_size:]

            # Check for speech in the last chunk
            if len(audio_buffer) >= self.sample_rate * 0.5:
                recent_audio = np.array(audio_buffer[-int(self.sample_rate * 0.5):])
                has_speech = self.detect_speech(recent_audio)

                if has_speech:
                    if not recording:
                        print("🗣️ Speech detected, recording...")
                        recording = True
                    speech_detected = True
                    silence_counter = 0
                else:
                    if recording:
                        silence_counter += 1

        try:
            with sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    callback=audio_callback,
                    blocksize=int(self.sample_rate * 0.1)
            ):
                while True:
                    sd.sleep(100)

                    if (recording and
                            silence_counter > (self.silence_duration / 0.1) and
                            len(audio_buffer) > self.sample_rate and
                            not is_speaking_callback()):

                        print("🔄 Processing speech...")

                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                            audio_array = np.array(audio_buffer)
                            self.save_audio_chunk(audio_array, tmp_file.name)

                            transcription = self.transcribe_audio(tmp_file.name)
                            Path(tmp_file.name).unlink()

                        if transcription:
                            on_speech_callback(transcription)

                        # Reset for next recording
                        audio_buffer = []
                        silence_counter = 0
                        speech_detected = False
                        recording = False

                        print("🎤 Listening...")

        except KeyboardInterrupt:
            print("\n⌨️ Stopping voice input...")
            return False

        return True