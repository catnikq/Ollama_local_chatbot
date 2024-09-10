from model.model import LLM  # Import your LLM class

class ConversationManager:
    def __init__(self):
        self.llm = LLM()  # Initialize the LLM model
        self.history = []  # Initialize the conversation history

    def add_user_message(self, user_input):
        """Add user's message to the conversation history."""
        self.history.append({"role": "user", "content": user_input})

    def add_assistant_message(self, response_text):
        """Add assistant's message to the conversation history."""
        self.history.append({"role": "assistant", "content": response_text})

    def generate_response(self, user_input):
        """Generate a response using the LLM model and update history."""
        self.add_user_message(user_input)
        response_text = self.llm.generate_response(self.history)
        self.add_assistant_message(response_text)
        return response_text

    def get_history(self):
        """Return the conversation history with only the content."""
        return [{"role": msg["role"], "content": msg["content"]} for msg in self.history]