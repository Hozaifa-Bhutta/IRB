import os
from openai import OpenAI
from llm_apis.base import BaseLLMAPI
from typing import Optional

class OpenAILLM(BaseLLMAPI):
    REASONING_MODELS = ["gpt-5"]
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model_name: str = "gpt-4o-mini",
                 reasoning: str = "medium"):
        super().__init__()
        print(f"Initializing '{model_name}'")
        self.model_name = model_name
        self.client = OpenAI(api_key = api_key if api_key else os.environ["OPENAI_API_KEY"])
        self.reasoning = reasoning

    def generate(self, system_prompt, user_prompt, max_output_tokens = 2048, temperature = None, top_p = None):
        kwargs = {
            "model": self.model_name,
            "instructions": system_prompt,
            "input": user_prompt,
            "max_output_tokens": max_output_tokens,
            "tool_choice": "none",
        }
        if self.model_name in self.REASONING_MODELS:
            # reasoning model
            kwargs["reasoning"] = {"effort": self.reasoning}

        if temperature is not None:
            kwargs["temperature"] = temperature

        resp = self.client.responses.create(**kwargs)
        result = resp.output_text.strip()

        return result