import json


def process_search_results(queries_ids, all_hits):
    predictions_metadata = {"chunk": {}, "full": {}}
    for query_id in queries_ids:
        hits = all_hits[query_id]
        temp = {}
        predictions_metadata["chunk"][query_id] = []
        for hit in hits:
            docid = hit.docid.split("--__--")[0]
            if docid not in temp: temp[docid] = 0
            temp[docid] = max(temp[docid], hit.score)
            
            predictions_metadata["chunk"][query_id].append(json.loads(hit.lucene_document.get('raw')))

        formatted_results = list(sorted([{"docid": k, "score": v} for k,v in temp.items()], key = lambda x: -x["score"]))
        predictions_metadata["full"][query_id] = formatted_results
    
    return predictions_metadata


