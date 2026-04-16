from .bedrock_models import BedRockLLM
from .gemini_models import GeminiLLM
from .openai_models import OpenAILLM
from .base import BaseLLMAPI


MODEL_NAME_2_MODEL_CLASS = {
    "gpt-5-mini": OpenAILLM,
    "gpt-5": OpenAILLM,
    "gpt-5_low": OpenAILLM,
    "gpt-4.1": OpenAILLM,
    "gpt-4.1-mini": OpenAILLM,
    "gpt-4.1-nano": OpenAILLM,
    "gpt-4o-mini": OpenAILLM,
    "gemini-2.5-flash": GeminiLLM,
    "gemini-2.5-flash-for-eval": GeminiLLM,
    "gemini-2.5-pro": GeminiLLM,
    "llama-4-scout": BedRockLLM,
    "llama-3.3-70B": BedRockLLM,
    "gpt-oss-120b": BedRockLLM,
    "qwen3-next-80B-A3B": BedRockLLM,
    "deepseek-r1": BedRockLLM
}

MODEL_NAME_2_INIT_DICT = {
    "gpt-5-mini": {
        "model_name": "gpt-5-mini",
        "reasoning": "medium",
    },
    "gpt-5": {
        "model_name": "gpt-5",
        "reasoning": "medium",
    },
    "gpt-5_low": {
        "model_name": "gpt-5",
        "reasoning": "low",
    },
    "gpt-4.1": {
        "model_name": "gpt-4.1"
    },
    "gpt-4.1-mini": {
        "model_name": "gpt-4.1-mini"
    },
    "gpt-4.1-nano": {
        "model_name": "gpt-4.1-nano"
    },
    "gpt-4o-mini": {
        "model_name": "gpt-4o-mini"
    },
    "gemini-2.5-flash": {
        "model_name": "gemini-2.5-flash",
        "reasoning": "medium"
    },
    "gemini-2.5-flash-for-eval": {
        "model_name": "gemini-2.5-flash",
        "reasoning": "none"
    },
    "gemini-2.5-pro": {
        "model_name": "gemini-2.5-pro",
        "reasoning": "medium"
    },
    "llama-4-scout": {
        "model_name": "us.meta.llama4-scout-17b-instruct-v1:0",
    },
    "llama-3.3-70B": {
        "model_name": "us.meta.llama3-3-70b-instruct-v1:0"
    },
    "gpt-oss-120b": {
        "model_name": "openai.gpt-oss-120b-1:0",
        "reasoning": "medium"
    },
    "qwen3-next-80B-A3B": {
        "model_name": "qwen.qwen3-next-80b-a3b",
    },
    "deepseek-r1": {
        "model_name": "us.deepseek.r1-v1:0",
        "reasoning": "medium"
    }
}

def init_llm(model_name):
    init_dict = MODEL_NAME_2_INIT_DICT[model_name]
    print(init_dict)
    model = MODEL_NAME_2_MODEL_CLASS[model_name](**init_dict)
    return model
