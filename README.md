# LLM Voice Chatbot

A voice-based conversational AI application using Kokoro TTS, Whisper for STT, and Ollama for LLM processing.

## Features

- **Voice Input**: Speech-to-Text using Faster-Whisper.
- **Voice Activity Detection**: Efficient speech detection using Silero VAD.
- **Conversational AI**: Powered by Ollama (Gemma 3 or other models).
- **High-Quality TTS**: Streaming text-to-speech using Kokoro.

## Prerequisites

- **Python 3.10+**
- **Ollama**: [Download and install Ollama](https://ollama.ai/).
- **Models**: Ensure you have the required models pulled in Ollama.
  ```bash
  ollama pull gemma-3-27b-it-abliterated-GGUF:Q4_K_M
  ```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd LLM-voice-chatbot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main application:
```bash
python voice_chat.py
```

### Options
- `--model=<model_name>`: Specify a different Ollama model.
- `--personality=<file>`: Use a custom personality template.
- `--no-tts`: Disable Text-to-Speech output.

## Project Structure

- `voice_chat.py`: Main entry point and application loop.
- `stt_processor.py`: Handles audio recording and transcription.
- `tts_processor.py`: Handles text-to-speech generation and playback.
- `llm_processor.py`: Manages interactions with the LLM via Ollama.
- `personality-template.txt`: Template defining the AI's personality.
