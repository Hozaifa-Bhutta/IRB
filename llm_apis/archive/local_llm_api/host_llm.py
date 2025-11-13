import subprocess, hydra, os
from omegaconf import DictConfig
# This script runs a VLLM server with a specified model and port.

# model = "Qwen/Qwen2-7B-Instruct" # replace this with the huggingface model you want to use
# port = 8000

@hydra.main(
    version_base=None,
    config_path="../conf/llm_api",
    config_name=os.getenv("LLM_API_CONFIG_NAME")
)
def main(cfg: DictConfig):
    model_name = cfg.model_name
    port = cfg.port
    tensor_parallel_size = cfg.get("tensor_parallel_size", None)

    # Build the base command for running the vLLM API server
    cmd = [
        "python3",
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--port", str(port),
        "--gpu-memory-utilization", "0.9",
        "--max-model-len", "32000",
        "--enable-prefix-caching"
    ]

    # Append tensor parallel size if specified
    if tensor_parallel_size:
        cmd += ["--tensor-parallel-size", str(tensor_parallel_size)]

    # Execute the server command
    subprocess.run(cmd)

    # Server available at "http://localhost:{port}/v1"


if __name__ == "__main__":
    main()