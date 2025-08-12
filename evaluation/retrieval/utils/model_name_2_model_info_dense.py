from transformers import AutoTokenizer, AutoModel

model_name_2_model_path = {
    "e5_base": "intfloat/e5-base-v2"
}

model_name_2_model_class = {
    "e5_base": AutoModel
}

model_name_2_tokenizer_class = {
    "e5_base": AutoTokenizer
}

model_name_2_prefix = {
    "e5_base": {"query": "query:", "doc": "document:"}
}