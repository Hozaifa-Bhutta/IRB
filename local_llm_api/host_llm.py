import subprocess, hydra, os
from omegaconf import DictConfig
# This script runs a VLLM server with a specified model and port.

# model = "Qwen/Qwen2-7B-Instruct" # replace this with the huggingface model you want to use
# port = 8000

@hydra.main(version_base=None, config_path="../conf/llm_api", config_name=os.getenv("LLM_API_CONFIG_NAME"))
def main(cfg: DictConfig):
    model_name = cfg.model_name
    port = cfg.port

    subprocess.run([
        "python3",
        "-m", "vllm.entrypoints.openai.api_server", # runs the module
        "--model", model_name, # specify the HF model you want to load
        "--port", str(port),
        "--gpu-memory-utilization", "0.8", # set GPU memory utilization to 80% cap
        "--max-model-len", "8192",  # maximum number of tokens model can handle in input+output
        "--enable-prefix-caching"
    ])



    # we can now run the OpenAI client against this server at  "http://localhost:{port}/v1"


if __name__ == "__main__":
    main()