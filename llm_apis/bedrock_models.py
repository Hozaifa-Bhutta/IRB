import os
from llm_apis.base import BaseLLMAPI
from openai import OpenAI
from typing import Optional


class BedRockLLM(BaseLLMAPI):
    REASONING_MODELS = []
    def __init__(self, api_key: Optional[str] = None, model_name: str = None, reasoning: str = "medium"):
        super().__init__()

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"
        )
        self.model_name = model_name


    def generate(self, system_prompt, user_prompt, max_output_tokens):
        kwargs = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
        if self.model_name in self.REASONING_MODELS:
            # reasoning model
            kwargs["reasoning_effort"] = "medium"

        resp = self.client.chat.completions.create(**kwargs)
        result = resp.choices[0].message.content.strip()

        return result