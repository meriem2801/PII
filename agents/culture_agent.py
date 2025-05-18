import openai
from config import OPENAI_API_KEY

class CultureAgent:
    def __init__(self):
        self.model = "gpt-4o"
        self.conversation_history = [
            {"role": "system", "content": "Réponds en expert du patrimoine et de l'histoire locale."}
        ]

    def handle_request(self, user_input):
        self.conversation_history.append({"role": "user", "content": user_input})
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.conversation_history,
            api_key=OPENAI_API_KEY
        )
        reply = response["choices"][0]["message"]["content"]
        self.conversation_history.append({"role": "assistant", "content": reply})
        return reply
