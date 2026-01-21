import queue
import re
import threading
import sys
import os
from pathlib import Path
import time
import numpy as np

import sounddevice as sd
import torch

# Check for Kokoro TTS availability
TTS_AVAILABLE = False
KOKORO_TYPE = None

try:
    # Try to import Kokoro TTS with correct imports
    try:
        print("Checking Kokoro TTS availability...")
        # Method 1: Correct Kokoro import
        from kokoro import KModel, KPipeline

        TTS_AVAILABLE = True
        KOKORO_TYPE = "kokoro"
        print("✅ Kokoro TTS found!")
    except ImportError:
        try:
            # Method 2: Alternative import path
            from kokoro_tts import KModel, KPipeline

            TTS_AVAILABLE = True
            KOKORO_TYPE = "kokoro_alt"
            print("✅ Kokoro TTS (alternative) found!")
        except ImportError:
            pass # No Kokoro TTS found via alternative path
except Exception as e:
    print(f"❌ TTS initialization error: {e}")
finally:
    if not TTS_AVAILABLE:
        print("❌ Kokoro TTS not found. Please install it:")
        print("pip install kokoro")


class TTSProcessor:
    """Handles Text-to-Speech using Kokoro TTS"""

    def __init__(self, enable_tts=True, stream_by_sentence=False):
        self.enable_tts = enable_tts and TTS_AVAILABLE
        self.stream_by_sentence = stream_by_sentence
        self.audio_queue = queue.Queue()
        self.tts_thread = None
        self._is_speaking_event = threading.Event()  # Use an event for thread-safe flag
        self.model = None
        self.voice = "af_heart"  # Default voice
        self.k_model = None
        self.k_pipeline = None
        self.voice_pack = None

        if self.enable_tts:
            self._initialize_tts()
            # Start TTS worker thread
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()
        else:
            print("🔇 TTS disabled or not available")

    def _initialize_tts(self):
        """Initialize the appropriate TTS model"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🎵 Loading Kokoro TTS on {device}...")

            if KOKORO_TYPE in ["kokoro", "kokoro_alt"]:
                from kokoro import KModel, KPipeline
                self.k_model = KModel().to(device).eval()
                lang_code = self.voice[0]
                self.k_pipeline = KPipeline(lang_code=lang_code, model=self.k_model)
                self.voice_pack = self.k_pipeline.load_voice(self.voice)
                print("✅ Kokoro TTS loaded successfully!")
                print(f"🎵 Using KModel, KPipeline, and voice '{self.voice}'")
            else:
                raise Exception("No valid Kokoro TTS implementation found")

        except Exception as e:
            print(f"❌ Failed to initialize Kokoro TTS: {e}")
            self.enable_tts = False

    def _tts_worker(self):
        """Background worker for TTS processing"""
        while True:
            text = self.audio_queue.get()  # Blocks until an item is available
            if text is None:  # Shutdown signal
                break

            self._is_speaking_event.set()  # Set the flag to True
            self._generate_and_play_speech(text)
            self._is_speaking_event.clear()  # Set the flag back to False

            self.audio_queue.task_done()  # Mark the task as done for queue.join()

    def _clean_text_for_tts(self, text):
        """Clean text to make it more suitable for TTS"""
        if not text:
            return ""

        # Remove or replace problematic characters/patterns
        clean_text = text.replace("*", "").replace("_", "")
        # Removed the line that replaced "..." with " pause "
        clean_text = clean_text.replace("—", " ")
        clean_text = clean_text.replace("–", " ")

        # Remove multiple spaces
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        # Remove any remaining markdown or formatting
        clean_text = re.sub(r'\[.*?\]', '', clean_text)  # Remove [text]
        clean_text = re.sub(r'\(.*?\)', '', clean_text)  # Remove (text) - be careful with this

        return clean_text

    def _generate_and_play_speech(self, text):
        """
        Generate speech with Kokoro TTS and play it.
        This function is called by the worker thread.
        """
        try:
            clean_text = self._clean_text_for_tts(text)
            if not clean_text:
                print("❌ No valid text to speak")
                return

            if KOKORO_TYPE in ["kokoro", "kokoro_alt"]:
                audio_tensors = []
                for _, ps, _ in self.k_pipeline(text=clean_text, voice=self.voice):
                    ref_s = self.voice_pack[len(ps) - 1]
                    audio_tensor = self.k_model(ps, ref_s, speed=1.0)
                    audio_tensors.append(audio_tensor)

                if audio_tensors:
                    combined_audio_tensor = torch.cat(audio_tensors).squeeze()
                    audio_array = combined_audio_tensor.cpu().numpy()
                else:
                    audio_array = None

                sample_rate = 24000
                if audio_array is None or audio_array.size == 0:
                    print("❌ Audio generation returned empty data.")
                    return

            # Normalize audio
            max_val = max(abs(audio_array.max()), abs(audio_array.min()))
            if max_val > 0:
                audio_array = audio_array / max_val * 0.7

            # Play the audio using an explicit OutputStream
            # This is more robust on macOS than sd.play() when multiple streams are active
            with sd.OutputStream(samplerate=sample_rate, channels=1) as stream:
                stream.write(audio_array)

        except Exception as e:
            print(f"❌ TTS Generation Error: {e}")
            self._fallback_tts_generation(clean_text)

    def _fallback_tts_generation(self, text):
        """Fallback when primary TTS fails (simply prints)"""
        print(f"🔊 FALLBACK: {text}")

    def _process_full_response(self, text_generator):
        """Buffer the entire response before sending to TTS."""
        full_response = ""
        for chunk in text_generator:
            print(chunk, end="", flush=True)
            if chunk:
                full_response += chunk
        
        if full_response.strip():
            self.audio_queue.put(full_response.strip())

    def _handle_sentence_boundaries(self, parts):
        """Helper to process complete sentences and return the remaining buffer."""
        # Process all complete sentences (pairs of sentence text and separator)
        for i in range(0, len(parts) - 1, 2):
            sentence = parts[i] + parts[i+1]
            if sentence.strip():
                self.audio_queue.put(sentence.strip())
        
        # Return the remaining incomplete part
        return parts[-1]

    def _process_sentence_streaming(self, text_generator):
        """Process text chunk by chunk, splitting into sentences for faster TTS."""
        sentence_buffer = ""
        # Regex to split text into sentences (handles ., !, ? followed by space)
        sentence_split_pattern = re.compile(r'([.!?]\s+)')

        for chunk in text_generator:
            print(chunk, end="", flush=True)
            if not chunk:
                continue
                
            sentence_buffer += chunk
            parts = sentence_split_pattern.split(sentence_buffer)
            
            if len(parts) > 1:
                sentence_buffer = self._handle_sentence_boundaries(parts)

        if sentence_buffer.strip():
            self.audio_queue.put(sentence_buffer.strip())

    def speak_stream(self, text_generator):
        """
        Takes a text generator (from the LLM) and processes it according to
        the configured streaming mode.
        """
        if not self.enable_tts:
            full_response = "".join(text_generator)
            print(f"\n🤖 AI said: {full_response}")
            return

        print("\n🤖 AI said: ", end="", flush=True)

        try:
            if self.stream_by_sentence:
                self._process_sentence_streaming(text_generator)
            else:
                self._process_full_response(text_generator)

            # Wait for the audio to complete
            self.audio_queue.join()
            print()

        except Exception as e:
            print(f"❌ Streaming TTS Error: {e}")


    def speak_text(self, text):
        """
        Directly speak a complete text string without streaming.
        Useful when you already have the full text.
        """
        if not self.enable_tts:
            print(f"\n🤖 AI said: {text}")
            return

        if text.strip():
            self.audio_queue.put(text.strip())
            self.audio_queue.join()

    def is_currently_speaking(self):
        """Check if TTS is currently playing using a thread-safe event."""
        return self._is_speaking_event.is_set()

    def cleanup(self):
        """Clean up TTS resources"""
        if self.tts_thread and self.tts_thread.is_alive():
            self.audio_queue.put(None)  # Signal shutdown
            self.tts_thread.join(timeout=2)

    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup()