from transformers import AutoModel
from typing import List


class JinaReranker:
    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size

        self.reranker = AutoModel.from_pretrained(
            model_name,
            dtype = "auto",
            trust_remote_code = True
        )
        self.reranker.eval()


    def __str__(self):
        return f"JinaReranker({self.model_name})"
    
    def score_query(self, query: str, texts: List[str]) -> List[float]:
        scores = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            batch_scores = self.reranker.rerank(query, batch_texts)
            batch_scores = [item["relevance_score"] for item in batch_scores]
            scores.extend(batch_scores)


        return scores