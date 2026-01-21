#!/usr/bin/env python3
"""
Voice Chat Application with Kokoro TTS
Handles voice input and audio output for conversational AI
"""

import sys
import time
import argparse
from stt_processor import AudioProcessor
from tts_processor import TTSProcessor
from llm_processor import LLMProcessor


class VoiceChatApp:
    """Voice-based conversational AI application"""

    def __init__(self,
                 model_name="hf.co/mlabonne/gemma-3-12b-it-abliterated-v2-GGUF:Q4_K_M",
                 personality_file="personality-template.txt",
                 enable_tts=True,
                 stream_by_sentence=False):

        print("🎮 Initializing Voice Chat with Kokoro TTS...")
        print("=" * 50)

        # Initialize components
        try:
            self.stt_processor = AudioProcessor()
            self.tts_processor = TTSProcessor(enable_tts=enable_tts, 
                                              stream_by_sentence=stream_by_sentence)
            self.llm_processor = LLMProcessor(model_name=model_name,
                                              personality_file=personality_file)

            print("=" * 50)
            print("✅ All components loaded successfully!")

        except Exception as e:
            print(f"❌ Failed to initialize components: {e}")
            sys.exit(1)

    def on_speech_detected(self, transcription):
        """Callback when speech is detected and transcribed"""
        print(f"\n👤 You said: {transcription}")

        # Generate AI response using the streaming generator
        response_generator = self.llm_processor.generate_response(transcription)

        # Speak the response in a streaming fashion
        if self.tts_processor.enable_tts:
            # Pass the generator directly to the new streaming method
            self.tts_processor.speak_stream(response_generator)
        else:
            # If TTS is disabled, we still need to consume the generator and print the full text
            full_response = "".join(response_generator)
            print(f"\n🤖 AI said: {full_response}")

    def is_ai_speaking(self):
        """Check if AI is currently speaking"""
        return self.tts_processor.is_currently_speaking()

    def run(self):
        """Main application loop"""
        print("\n🎤 Voice Chat Mode")
        print("💭 Let's have a genuine conversation! I'm curious about everything...")
        print("\nControls:")
        print("- Just speak naturally")
        print("- Press Ctrl+C to exit")
        if self.tts_processor.enable_tts:
            print("- AI responses will be spoken aloud")
        print("\n" + "=" * 50)

        try:
            # Start voice recording loop
            while True:
                try:
                    # Record with VAD and handle speech
                    still_running = self.stt_processor.record_with_vad(
                        on_speech_callback=self.on_speech_detected,
                        is_speaking_callback=self.is_ai_speaking
                    )

                    if not still_running:
                        break

                except Exception as e:
                    print(f"❌ Voice processing error: {e}")
                    print("🔄 Restarting voice input...")
                    time.sleep(1)

        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down...")
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up resources...")
        if hasattr(self, 'tts_processor'):
            self.tts_processor.cleanup()
        print("✅ Cleanup completed")

    def __del__(self):
        """Destructor"""
        self.cleanup()


def check_requirements():
    """Check if required packages are installed"""
    missing_packages = []

    packages_to_check = [
        ('torch', 'torch'),
        ('sounddevice', 'sounddevice'),
        ('faster_whisper', 'faster-whisper'),
        ('langchain_ollama', 'langchain-ollama'),
        ('soundfile', 'soundfile'),
        ('numpy', 'numpy')
    ]

    for module_name, package_name in packages_to_check:
        try:
            __import__(module_name)
        except ImportError:
            missing_packages.append(package_name)

    if missing_packages:
        print("❌ Missing required packages. Please install:")
        print("pip install " + " ".join(missing_packages))
        print("\nFor Kokoro TTS, also try:")
        print("pip install kokoro-tts")
        print("or")
        print("pip install transformers")
        return False

    return True


def main():
    """Main entry point"""
    if not check_requirements():
        sys.exit(1)

    #         models_to_test = [
    #         "hf.co/mlabonne/gemma-3-12b-it-abliterated-v2-GGUF:Q4_K_M",
    #         "hf.co/mlabonne/gemma-3-4b-it-abliterated-v2-GGUF:Q4_K_M",
    #         "hf.co/mlabonne/gemma-3-27b-it-abliterated-GGUF:Q4_K_M"
    #     ]

    # Parse command line arguments (basic)
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Voice Chat Application with Kokoro TTS")
    parser.add_argument(
        "--model", 
        type=str, 
        default="hf.co/mlabonne/gemma-3-27b-it-abliterated-GGUF:Q4_K_M",
        help="LLM model name to use"
    )
    parser.add_argument(
        "--personality", 
        type=str, 
        default="personality-template.txt",
        help="Path to personality template file"
    )
    parser.add_argument(
        "--no-tts", 
        action="store_false", 
        dest="enable_tts",
        help="Disable Text-to-Speech"
    )
    parser.add_argument(
        "--stream-by-sentence", 
        action="store_true",
        help="Stream TTS by sentence instead of waiting for full response"
    )
    
    # Set defaults for flags if not provided
    parser.set_defaults(enable_tts=True)
    
    args = parser.parse_args()

    model_name = args.model
    personality_file = args.personality
    enable_tts = args.enable_tts
    stream_by_sentence = args.stream_by_sentence

    print("🚀 Starting Voice Chat...")
    print(f"📝 Model: {model_name}")
    print(f"🎭 Personality: {personality_file}")
    print(f"🎵 TTS: {'Enabled' if enable_tts else 'Disabled'}")
    if enable_tts:
        print(f"📡 TTS Streaming: {'Sentence-based' if stream_by_sentence else 'Full-response'}")

    # Create and run the application
    app = VoiceChatApp(
        model_name=model_name,
        personality_file=personality_file,
        enable_tts=enable_tts,
        stream_by_sentence=stream_by_sentence
    )

    app.run()


if __name__ == "__main__":
    main()