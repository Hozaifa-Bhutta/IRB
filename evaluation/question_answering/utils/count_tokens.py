import tiktoken
from transformers import AutoTokenizer


TOKENIZER = {}
def count_tokens(text: str, model: str) -> int:
    try:
        if model == "gpt-oss-120b":
            if model not in TOKENIZER:
                encoding = tiktoken.get_encoding("cl100k_base")
                TOKENIZER[model] = encoding
            return len(TOKENIZER[model].encode(text))

        elif model == "deepseek-r1":
            if model not in TOKENIZER:
                tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)
                TOKENIZER[model] = tokenizer
            return len(TOKENIZER[model].encode(text))

        else:
            return 0

    except Exception as e:
        return 0