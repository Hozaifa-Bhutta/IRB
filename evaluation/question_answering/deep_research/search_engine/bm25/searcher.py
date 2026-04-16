import json
from evaluation.question_answering.deep_research.search_engine.base import BaseSearcher
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexReader

class BM25Searcher(BaseSearcher):
    def __init__(self, index_path: str, reranker = None, reranking_depth: int = 50):
        super().__init__()
        self.searcher = LuceneSearcher(index_path)
        
        self.index_reader = LuceneIndexReader(index_path)
        self.reranker = reranker
        self.reranking_depth = reranking_depth

    def search(self, query: str, k: int = 10):
        if not self.reranker:
            hits = self.searcher.search(query, k=k)
        else: 
            hits = self.searcher.search(query, k = self.reranking_depth)
        
        results = []
        for hit in hits:
            doc = self.get_doc(hit.doc_id)
            url = hit.doc_id.split("--__--")[0]
            text = doc['contents']
            published_date = doc["published_date"]

            results.append({
                "docid": hit.docid,
                "url": url,
                "published_date": published_date,
                "score": hit.score,
                "text": text
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


    def get_doc(self, docid: str):
        doc = self.searcher.doc(docid)
        if doc is None:
            return None
        
        raw_content = doc.raw()
        try:
            return json.loads(raw_content)
        except json.JSONDecodeError:
            return raw_content 

    def get_doc_vec(self, docid: str):
        doc_vector = self.index_reader.get_document_vector(docid)
        return doc_vector