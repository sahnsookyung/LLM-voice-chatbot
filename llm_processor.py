from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path


class LLMProcessor:
    """Handles Language Model interactions"""

    def __init__(self, model_name, personality_file="personality-template.txt"):
        self.model_name = model_name
        self.conversation_history = []
        self.max_history_pairs = 10

        # Load personality template
        self.template = self._load_personality_template(personality_file)

        print("🧠 Loading Language Model...")
        try:
            self.llm_model = OllamaLLM(
                model=self.model_name
            )
            self.prompt = ChatPromptTemplate.from_template(self.template)
            self.chain = self.prompt | self.llm_model
            print(f"✅ LLM loaded successfully! Using model: {self.model_name}")

        except Exception as e:
            print(f"❌ Failed to load LLM: {e}")
            print("💡 Make sure Ollama is running and the model is installed:")
            print(f"   ollama pull {self.model_name}")
            raise

    def _load_personality_template(self, filename):
        """Load personality template from file"""
        try:
            personality_path = Path(filename)
            if not personality_path.exists():
                raise FileNotFoundError(f"Personality file not found: {filename}")

            with open(personality_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            print(f"✅ Loaded personality template from {filename}")
            return template_content
        except Exception as e:
            print(f"❌ Failed to load personality template: {e}")
            print("💡 Using default template.")
            return "You are a friendly and helpful AI assistant. Always be conversational and curious."

    def _build_context(self):
        """Build context from recent history"""
        context = ""
        for human_msg, ai_msg in self.conversation_history[-self.max_history_pairs:]:
            context += f"Human: {human_msg}\nAssistant: {ai_msg}\n\n"
        return context

    def generate_response(self, text):
        """Generate AI response using the LLM in a streaming fashion."""
        context = self._build_context()

        try:
            full_response_text = ""
            # The .stream() method returns a generator
            response_stream = self.chain.stream({"context": context, "question": text})

            # This loop yields each chunk as it comes in
            for chunk in response_stream:
                if isinstance(chunk, str):
                    content = chunk
                elif hasattr(chunk, 'content'):
                    content = chunk.content
                else:
                    continue
                full_response_text += content
                yield content

            # Add the complete response to history
            self.conversation_history.append((text, full_response_text))

            # Manage history size
            if len(self.conversation_history) > self.max_history_pairs:
                self.conversation_history = self.conversation_history[-self.max_history_pairs:]

        except Exception as e:
            print(f"❌ LLM Error: {e}")
            error_response = "Hmm, something weird happened in my brain just now. Can you try that again?"
            self.conversation_history.append((text, error_response))
            yield error_response

    def generate_one_time_response(self, text):
        """Generate AI response in a single, non-streaming call."""
        context = self._build_context()

        try:
            # The .invoke() method waits for the full response
            response = self.chain.invoke({"context": context, "question": text})

            # Add the complete response to history
            self.conversation_history.append((text, response))

            # Manage history size
            if len(self.conversation_history) > self.max_history_pairs:
                self.conversation_history = self.conversation_history[-self.max_history_pairs:]

            return response

        except Exception as e:
            print(f"❌ LLM Error: {e}")
            error_response = "Hmm, something weird happened in my brain just now. Can you try that again?"
            self.conversation_history.append((text, error_response))
            return error_response

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("🗑️ Conversation history cleared")

    def get_history_summary(self):
        """Get a summary of conversation history"""
        if not self.conversation_history:
            return "No conversation history"

        return f"Conversation history: {len(self.conversation_history)} exchanges"

    def set_model(self, model_name):
        """Switch to a different model"""
        try:
            print(f"🔄 Switching to model: {model_name}")
            self.model_name = model_name
            self.llm_model = OllamaLLM(
                model=self.model_name
            )
            self.prompt = ChatPromptTemplate.from_template(self.template)
            self.chain = self.prompt | self.llm_model
            print(f"✅ Successfully switched to {model_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to switch model: {e}")
            return False

    def reload_personality(self, filename="personality-template.txt"):
        """Reload personality template from file"""
        self.template = self._load_personality_template(filename)
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.chain = self.prompt | self.llm_model
        print("🔄 Personality template reloaded")
