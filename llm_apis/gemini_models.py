import os
from llm_apis.base import BaseLLMAPI
from openai import OpenAI
from typing import Optional

class GeminiLLM(BaseLLMAPI):
    REASONING_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro"]
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model_name: str = "gemini-2.5-flash", 
                 reasoning: str = "medium"):
        super().__init__()

        self.client = OpenAI(
            api_key=api_key if api_key else os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.model_name = model_name
        self.reasoning = reasoning


    def generate(self, system_prompt: str, user_prompt: str, max_output_tokens: int = 2048) -> str:
        kwargs = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
        if self.model_name in self.REASONING_MODELS:
            # reasoning model
            kwargs["reasoning_effort"] = self.reasoning

        resp = self.client.chat.completions.create(**kwargs)
        result = resp.choices[0].message.content.strip()

        return result