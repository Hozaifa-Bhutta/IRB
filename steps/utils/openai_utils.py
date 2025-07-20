from openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

OPENAI_CLIENT = {
    "client": None
}

OPENAI_EMBEDDING = {
    "model": None,
    "name": None
}


def init_client(openai_api_key):
    if OPENAI_CLIENT["client"] is None:
        print("Initializing OpenAI client")
        client = OpenAI(api_key=openai_api_key)
        OPENAI_CLIENT["client"] = client


def init_openai_embedding(openai_api_key, 
                          embedding_model_name = "text-embedding-3-small"):
    if embedding_model_name != OPENAI_EMBEDDING["name"]:
        print(f"Initializing OpenAI embedding: {embedding_model_name}")
        embedding_model = OpenAIEmbedding(embed_model=embedding_model_name, 
                                      api_key = openai_api_key)
        
        OPENAI_EMBEDDING["model"] = embedding_model
        OPENAI_EMBEDDING["name"] = embedding_model_name