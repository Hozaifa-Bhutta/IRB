export LLM_API_CONFIG_NAME="qwen3_8b" #"qwen7binstruct"
export CUDA_VISIBLE_DEVICES="1"

python local_llm_api/host_llm.py