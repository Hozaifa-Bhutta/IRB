import torch
from evaluation.question_answering.deep_research.search_engine.bm25.searcher import BM25Searcher
from evaluation.question_answering.deep_research.search_engine.dense.searcher import DenseSearcher
from evaluation.retrieval.utils.text_embeddings import create_query_encoder, init_model


def init_ds_searcher(model_name, index_folder):
    if model_name != "bm25":
        model, tokenizer = init_model(retrieval_model = model_name, 
                                      device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        query_encoder = create_query_encoder(model, 
                                            tokenizer, 
                                            model_name = model_name)
        return DenseSearcher(index_folder = index_folder, query_encoder = query_encoder)
    else:
        # TODO: do this
        raise NotImplementedError
    
