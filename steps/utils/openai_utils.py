from openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

OPENAI_CLIENT = {
    "client": None
}

OPENAI_EMBEDDING = {
    "model": None,
    "name": None
}


def init_client(openai_api_key, local = False, port = None, model_name = None):
    if OPENAI_CLIENT["client"] is None:
        print("Initializing OpenAI client")
        if not local:
            assert openai_api_key is not None
            client = OpenAI(api_key=openai_api_key)
            model = "gpt-4o"
        else:
            client = OpenAI(api_key="test", base_url=f"http://localhost:{port}/v1")
            model = model_name
        
        
        OPENAI_CLIENT["client"] = client
        OPENAI_CLIENT["model"] = model



def init_openai_embedding(openai_api_key, 
                          embedding_model_name = "text-embedding-3-small"):
    if embedding_model_name != OPENAI_EMBEDDING["name"]:
        print(f"Initializing OpenAI embedding: {embedding_model_name}")
        embedding_model = OpenAIEmbedding(embed_model=embedding_model_name, 
                                      api_key = openai_api_key)
        
        OPENAI_EMBEDDING["model"] = embedding_model
        OPENAI_EMBEDDING["name"] = embedding_model_name