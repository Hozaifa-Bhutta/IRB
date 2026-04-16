import os, json, faiss
import numpy as np
from typing import List, Dict, Any, Callable
from evaluation.question_answering.deep_research.search_engine.base import BaseSearcher

class DenseSearcher(BaseSearcher):
    def __init__(self, index_folder: str, query_encoder: Callable, reranker = None, reranking_depth: int = 50):
        super().__init__()
        self.index_folder = index_folder
        self.query_encoder = query_encoder
        self.reranker = reranker
        self.reranking_depth = reranking_depth

        index_path = os.path.join(index_folder, "faiss_index_flatip.index")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        self.index = faiss.read_index(index_path)
        
        id_map_path = os.path.join(index_folder, "id_map.json")
        if os.path.exists(id_map_path):
            with open(id_map_path, "r") as f:
                self.id_map = {int(k): str(v) for k, v in json.load(f).items()}
        else:
            self.id_map = None
            
        raw_path = os.path.join(index_folder, "raw.json")
        if os.path.exists(raw_path):
            with open(raw_path, "r") as f:
                self.raw_data = json.load(f)
        else:
            self.raw_data = {}

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        query_vec = self.query_encoder(query)
        
        query_vec = np.array(query_vec, dtype='float32').reshape(1, -1)
        
        faiss.normalize_L2(query_vec)
        

        if self.reranker is None:
            scores, indices = self.index.search(query_vec, k)
        else:
            scores, indices = self.index.search(query_vec, self.reranking_depth)
        
        results = []
        for score, int_id in zip(scores[0], indices[0]):
            if int_id == -1:
                continue
            
            doc_id = self.id_map[int_id] if self.id_map else str(int_id)

            doc = self.get_doc(doc_id)
            url = doc_id.split("--__--")[0]
            text = doc['contents']
            published_date = doc["published_date"]
            
            results.append({
                "docid": doc_id,
                "url": url,
                "published_date": published_date,
                "score": float(score),
                "text": text,
            })

        if not self.reranker: return results

        texts = [item["text"] for item in results]
        reranking_scores = self.reranker.score_query(query, texts)
        reranked_pairs = sorted(
            zip(results, reranking_scores), key=lambda x: x[1], reverse=True
        )
        reranked = [
            {"docid": item["docid"], "score": float(score), "text": item["text"]}
            for item, score in reranked_pairs
        ]

        return reranked[:k]
    

    def get_doc(self, _id: str) -> Dict[str, Any]:
        return self.raw_data.get(_id, None)

    def get_doc_vec(self, _id: str):
        pass