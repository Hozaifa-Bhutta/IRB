from transformers import AutoTokenizer, AutoModel

model_name_2_model_path = {
    "e5_base": "intfloat/e5-base-v2",
    "bge_m3": "BAAI/bge-m3"
}

model_name_2_model_class = {
    "e5_base": AutoModel,
    "bge_m3": AutoModel
}

model_name_2_tokenizer_class = {
    "e5_base": AutoTokenizer,
    "bge_m3": AutoTokenizer
}

model_name_2_prefix = {
    "e5_base": {"query": "query:", "doc": "document:"}
}


def text_embedding_batch(batch, model, tokenizer, model_name, prefix = None, device = None):
    if prefix is not None:
        batch = [prefix + " " + text for text in batch]
    
    inputs = tokenizer(batch, 
                        padding=True, 
                        truncation=True,
                        return_tensors="pt", 
                        return_token_type_ids=False, 
                        max_length=256).to(device)
        
    output = model(**inputs)

    if model_name in ["specter2", "bge_m3"]:
        return output.last_hidden_state[:, 0, :].cpu()
    
    elif model_name in ["e5_base"]:
        attention_mask = inputs["attention_mask"]
        last_hidden = output.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    else:
        raise NotImplementedError