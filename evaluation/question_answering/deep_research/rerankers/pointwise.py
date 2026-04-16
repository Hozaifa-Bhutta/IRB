import torch
from FlagEmbedding import FlagReranker
from typing import List


class BGEReranker:
    def __init__(self, model_name: str = 'BAAI/bge-reranker-v2-m3', batch_size: int = 32):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.reranker = FlagReranker(model_name, use_fp16=True)

        self.model_name = model_name
        self.batch_size = batch_size


    def __str__(self):
        return f"BGEReranker({self.model_name})"
    
    def score_query(self, query: str, texts: List[str]) -> List[float]:
        scores = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_pairs = [[query, text] for text in batch_texts]

            batch_scores = self.reranker.compute_score(batch_pairs, normalize = True)
            scores.extend(batch_scores)


        return scores