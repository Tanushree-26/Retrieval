import os
from groq import Groq
from src.config import MODEL, GROQ_API_KEY

class GenerationClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or GROQ_API_KEY
        if not self.api_key:
            # print("Warning: GROQ_API_KEY not found.")
            pass
            
        self.client = Groq(api_key=self.api_key)

    def generate_response(self, query, context_chunks):
        """
        Generates a response using Groq API (Llama 3.3).
        """
        if not self.api_key:
            return "Error: API Key is missing. Please provide a Groq API Key."

        context_text = "\n\n".join([chunk for chunk in context_chunks])
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion:\n{query}"}
        ]

        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=MODEL
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
