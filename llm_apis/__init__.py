from .bedrock_models import BedRockLLM
from .gemini_models import GeminiLLM
from .openai_models import OpenAILLM
from .base import BaseLLMAPI


MODEL_NAME_2_MODEL_CLASS = {
    "gpt-5-mini": OpenAILLM,
    "gpt-5": OpenAILLM,
    "gpt-4.1": OpenAILLM,
    "gpt-4.1-mini": OpenAILLM,
    "gpt-4o-mini": OpenAILLM,
    "gemini-2.5-flash": GeminiLLM,
    "gemini-2.5-pro": GeminiLLM
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
    "gpt-4.1": {
        "model_name": "gpt-4.1"
    },
    "gpt-4.1-mini": {
        "model_name": "gpt-4.1-mini"
    },
    "gpt-4o-mini": {
        "model_name": "gpt-4o-mini"
    },
    "gemini-2.5-flash": {
        "model_name": "gemini-2.5-flash",
        "reasoning": "medium"
    },
    "gemini-2.5-pro": {
        "model_name": "gemini-2.5-pro",
        "reasoning": "medium"
    }
}

def init_llm(model_name):
    init_dict = MODEL_NAME_2_INIT_DICT[model_name]
    print(init_dict)
    model = MODEL_NAME_2_MODEL_CLASS[model_name](**init_dict)
    return model
