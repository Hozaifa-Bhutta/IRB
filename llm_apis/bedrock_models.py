import os
import boto3
from llm_apis.base import BaseLLMAPI
from typing import Optional, List, Dict, Any

class BedRockLLM(BaseLLMAPI):
    REASONING_MODELS = []

    def __init__(
        self, 
        api_key: Optional[str] = None, 
        region_name: str = "us-east-2", 
        model_name: str = None, 
        is_reasoning_model: bool = False,
        reasoning: str = "medium",
    ):
        super().__init__()

        if api_key:
            os.environ['AWS_BEARER_TOKEN_BEDROCK'] = api_key

        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region_name
        )
        
        self.model_name = model_name
        self.reasoning = reasoning

        self.is_reasoning_model = is_reasoning_model
        # similar to gemini 2.5 pro, see https://ai.google.dev/gemini-api/docs/openai
        self.reasoning_budgets = {
            "none": 0,
            "low": 1024,
            "medium": 8192,
            "high": 24576
        }

    def generate(self, system_prompt: str, user_prompt: str, max_output_tokens: int, return_dict: bool = False) -> str:
        system_prompts = [{"text": system_prompt}]
        messages = [
            {
                "role": "user",
                "content": [{"text": user_prompt}]
            }
        ]

        inference_config = {
            "maxTokens": max_output_tokens,
        }

        additional_fields = {}
        if self.is_reasoning_model:
            budget = self.reasoning_budgets.get(self.reasoning_effort, 4096)
            additional_fields["reasoning_config"] = {
                "type": "enabled", 
                "budget_tokens": budget
            }

        response = self.client.converse(
            modelId=self.model_name,
            messages=messages,
            system=system_prompts,
            inferenceConfig=inference_config,
            additionalModelRequestFields=additional_fields
        )

        content_blocks = response['output']['message']['content']
        for block in content_blocks:
            if 'text' in block:
                # This is the final visible answer
                final_answer = block['text']

        if return_dict:
            return {
                "text": final_answer.strip(),
                "raw": response
            }
        else:
            output_content = final_answer
            return output_content.strip()