import torch, time
from openai import OpenAI, BadRequestError
from transformers import AutoTokenizer, AutoModel

model_name_2_model_path = {
    "e5_base": "intfloat/e5-base-v2",
    "bge_m3": "BAAI/bge-m3",
    "grit_lm": "GritLM/GritLM-7B"
}

model_name_2_model_class = {
    "e5_base": AutoModel,
    "bge_m3": AutoModel,
    "grit_lm": AutoModel
}

model_name_2_tokenizer_class = {
    "e5_base": AutoTokenizer,
    "bge_m3": AutoTokenizer,
    "grit_lm": AutoTokenizer
}

model_name_2_prefix = {
    "e5_base": {"query": "query:", "doc": "document:"},
    "grit_lm": {"query": "<|embed|>\n", "doc": "<|embed|>\n"}
}


def init_model(retrieval_model, device):
    if retrieval_model in ["e5_base", "bge_m3", "grit_lm"]:
        model = model_name_2_model_class[retrieval_model].from_pretrained(
            model_name_2_model_path[retrieval_model], 
        )
        tokenizer = model_name_2_tokenizer_class[retrieval_model].from_pretrained(
            model_name_2_model_path[retrieval_model]
        )

        model.eval()
        model.to(device)

        return model, tokenizer
    else:
        return None, None


def text_embedding_batch_api(batch, model, tokenizer, model_name, prefix, device = None, max_length = 512):
    client = OpenAI()

    OPENAI_EMBEDDING_HIDDEN_SIZE = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536
    }

    try:
        response = client.embeddings.create(
            model=model_name,
            input=batch  
        )
        embeddings = torch.tensor([item.embedding for item in response.data])
        return embeddings

    except BadRequestError:
        embeddings = []
        for doc in batch:
            try:
                response = client.embeddings.create(
                    input=doc,
                    model=model_name
                )
                embedding = torch.tensor(response.data[0].embedding)
            except BadRequestError:
                hidden_size = OPENAI_EMBEDDING_HIDDEN_SIZE.get(model_name, 1536)
                embedding = torch.zeros([hidden_size])
            
            embeddings.append(embedding)
        
        embeddings = torch.stack(embeddings)

        return embeddings


def text_embedding_batch_hf(batch, model, tokenizer, model_name, prefix = None, device = None, max_length = 512):
    if prefix is not None:
        batch = [prefix + " " + text for text in batch]
    
    inputs = tokenizer(batch, 
                        padding=True, 
                        truncation=True,
                        return_tensors="pt", 
                        return_token_type_ids=False, 
                        max_length=max_length).to(device)
        
    output = model(**inputs)

    if model_name in ["specter2", "bge_m3"]:
        return output.last_hidden_state[:, 0, :].cpu().float()
    
    elif model_name in ["e5_base", "grit_lm"]:
        attention_mask = inputs["attention_mask"]
        last_hidden = output.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return (last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]).float()
    
    else:
        raise NotImplementedError
    


def text_embedding_batch(batch, model, tokenizer, model_name, prefix = None, device = None, max_length = 512):
    if model:
        return text_embedding_batch_hf(batch, model, tokenizer, model_name, prefix, device, max_length)
    else: 
        res = text_embedding_batch_api(batch, model, tokenizer, model_name, prefix, device, max_length)
        print(res.shape)

        time.sleep(0.2)

        return res



def create_query_encoder(model, tokenizer, model_name, prefix=None, device=None, max_length=512):
    def encoder(query: str):
        # 1. Wrap the single query in a list to make it a batch of 1
        batch = [query]
        
        # 2. Call your existing batch function
        batch_embeddings = text_embedding_batch(
            batch=batch, 
            model=model, 
            tokenizer=tokenizer, 
            model_name=model_name, 
            prefix=prefix, 
            device=device, 
            max_length=max_length
        )
        
        # 3. Extract the first (and only) embedding from the batch to return a 1D array
        return batch_embeddings[0] 
        
    return encoder