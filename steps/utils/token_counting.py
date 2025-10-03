import tiktoken


TIKTOKEN_ENC = {}


def init_enc(model_name):
    if model_name not in TIKTOKEN_ENC:
        TIKTOKEN_ENC[model_name] = {
            "enc": tiktoken.encoding_for_model("gpt-4o")
        }


def token_count_tiktoken(text, model_name):
    init_enc(model_name)

    return len(TIKTOKEN_ENC[model_name]["enc"].encode(text))