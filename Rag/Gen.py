import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class Gen:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)

    def format_prompt(self, context, prompt):
        """
        Format the context into a prompt for the LLM.
        
        Args:
            context (str): The context or information to be included in the prompt.
        
        Returns:
            str: The formatted prompt.
        """
        prompt = f"You are part of a RAG pipeline. Using the information below, answer the following question as naturally as possible. If the context is insufficient to fully answer, state that clearly, but still provide the best possible response. When quoting, specify the document source.\nContext: {context}\n\nQuestion: {prompt}"
        return prompt

    def generate_response(self, prompt):
        """
        Generate a response from Groq's LLM using the provided prompt.
        
        Args:
            prompt (str): The prompt to send to the LLM.
        
        Returns:
            str: The response from the LLM.
        """
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.2-3b-preview",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return "FAILED"

    def process_and_respond(self, context, prompt):
        """
        Takes context, formats it, and then returns the LLM response.
        
        Args:
            context (str): The context to be processed and used for generating the response.
        
        Returns:
            str: The LLM's response.
        """
        prompt = self.format_prompt(context, prompt)
        response = self.generate_response(prompt)
        return response
