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
   - Note: There's been some controversy around Ollama being a bit shady. If you want to use other things I've heard better things about [llama.cpp](https://github.com/ggml-org/llama.cpp), but you might need to refactor things a bit.
- **Models**: Ensure you have the required models pulled in Ollama. e.g. This abliterated model from the original gemma 3 model provides a somewhat uncensored experience. Read more about abliteration here.
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
   -  Note that the `{context}Human: {question}\nAssistant:` format at the end of the template is necessary for Ollama when using a simple prompt template. This is because Ollama expects a completion-style prompt. It needs explicit markers like "Human:" and "Assistant:" to understand the conversation structure.

## TODO
- Context is currently retained to 10 messages. This is a bit arbitrary and should be configurable. How effective this is also depends on the model size.
- We could look into compacting the context periodically to preserve context.
- If we want to have a chatbot that retains long-term context, then we'd need a vector database of some sort to store the context and retrieve the relevant memories.