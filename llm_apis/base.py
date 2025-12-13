

class BaseLLMAPI:
    def __init__(self):
        pass


    def generate(self, system_prompt: str, user_prompt: str, max_output_tokens: int, return_dict: bool) -> str:
        raise NotImplementedError