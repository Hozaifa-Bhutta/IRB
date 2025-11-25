import json, torch, argparse, os, pytrec_eval, logging, faiss, hydra
import pandas as pd
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
from typing import List, Dict, Tuple
from dataclasses import dataclass
from evaluation.retrieval.utils.text_embeddings import model_name_2_prefix, text_embedding_batch, init_model
from evaluation.retrieval.utils.generic import process_search_results
from evaluation.retrieval.utils.allowed_datasets import ALLOWED_DATASETS


logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@dataclass
class DenseSearchResult:
    docid: str
    score: float
    lucene_document: dict


def convert_to_pytrec_eval_format(queries, all_search_results, type = "relevance"):
    """
    queries: [q1, q2, ...]
    all_search_results: [[{'docid': '22711954', 'score': 4.043900012969971}, ...]]
    """

    assert len(queries) == len(all_search_results)

    score_converter = {
        "relevance": lambda x: int(x),
        "prediction": lambda x:float(x)
    }

    res = {}
    for query, search_results in zip(queries, all_search_results):
        if query not in res:
            res[query] = {}
        

        for sr in search_results:
            docid = str(sr["docid"])
            score = score_converter[type](sr["score"])

            res[query][docid] = score

    return res


def mrr(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: list[int],
) -> tuple[dict[str, float]]:
    MRR = {}

    for k in k_values:
        MRR[f"MRR@{k}"] = 0.0

    k_max, top_hits = max(k_values), {}
    logging.info("\n")

    for query_id, doc_scores in results.items():
        top_hits[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]

    for query_id in top_hits:
        if query_id in qrels:
            query_relevant_docs = set([doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0])
        else: query_relevant_docs = set([])
        for k in k_values:
            for rank, hit in enumerate(top_hits[query_id][0:k]):
                if hit[0] in query_relevant_docs:
                    MRR[f"MRR@{k}"] += 1.0 / (rank + 1)
                    break

    for k in k_values:
        MRR[f"MRR@{k}"] = round(MRR[f"MRR@{k}"] / len(qrels), 5)
        logging.info("MRR@{}: {:.4f}".format(k, MRR[f"MRR@{k}"]))

    return MRR


def evaluate(qrels: Dict[str, Dict[str, int]], 
                results: Dict[str, Dict[str, float]], 
                k_values: List[int],
                ignore_identical_ids: bool=True) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    
    if ignore_identical_ids:
        logger.info('For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this.')
        popped = []
        for qid, rels in results.items():
            for pid in list(rels):
                if qid == pid:
                    results[qid].pop(pid)
                    popped.append(pid)

    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    
    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0
    
    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
    scores = evaluator.evaluate(results)

    
    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_"+ str(k)]
    
    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"]/len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"]/len(scores), 5)
    
    for eval in [ndcg, _map, recall, precision]:
        logger.info("\n")
        for k in eval.keys():
            logger.info("{}: {:.4f}".format(k, eval[k]))

    return ndcg, _map, recall, precision

def read_qrels(qrel_path):
    _qrels = pd.read_csv(qrel_path, sep='\t').to_dict("records")
    
    metadata = {}
    for line in _qrels:
        query_id = str(line["query-id"])
        doc_id = str(line["corpus-id"])
        score = line["score"]

        if query_id not in metadata:
            metadata[query_id] = []

        metadata[query_id].append({
            "docid": doc_id,
            "score": score
        })

    queries_ids = list(metadata.keys())
    queries_all_labels = [metadata[k] for k in queries_ids]

    qrels = convert_to_pytrec_eval_format(queries = queries_ids, all_search_results=queries_all_labels)

    return qrels


def batch_search(embeddings, q_ids, index, k, id_map=None, id2raw = None):
    faiss.normalize_L2(embeddings)
    D, I = index.search(embeddings, k)
    
    results = {
        key: [
            DenseSearchResult(
                docid=id_map[int(idx)] if id_map is not None else int(idx),
                score=float(score),
                lucene_document={"raw": json.dumps(id2raw[id_map[int(idx)] if id_map is not None else int(idx)])}if id2raw else None
            )
            for score, idx in zip(distances, indexes) if idx != -1
        ]
        for key, distances, indexes in zip(q_ids, D, I)
    }
    return results



def load_index(index_folder):
    index = faiss.read_index(os.path.join(index_folder, "faiss_index_flatip.index"))
    id_map_path = os.path.join(index_folder, "id_map.json")
    if os.path.exists(id_map_path):
        import json
        with open(id_map_path, "r") as f:
            id_map = json.load(f)
        # JSON keys are strings, convert them back to int
        id_map = {int(k): v for k, v in id_map.items()}
    else:
        id_map = None

    raw_path = os.path.join(index_folder, "raw.json")
    if os.path.exists(raw_path):
        import json
        with open(raw_path) as f:
            id2raw = json.load(f)
    else: id2raw = None
    return index, id_map, id2raw



@hydra.main(version_base=None, config_path="../../conf/evaluation/", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig):
    retrieval_model = cfg.general.retrieval_model
    index_folder = cfg.general.index_folder
    work_dir = cfg.general.work_dir
    dataset = cfg.general.dataset
    dataset_date = cfg.general.dataset_date
    batch_size = cfg.retrieval.eval.batch_size
    threads = cfg.retrieval.eval.threads
    retrieval_metadata_path = cfg.general.retrieval_metadata_path


    dataset_name_2_relative_path = {
        dn: os.path.join("benchmarks", dataset_date, dn) if dataset_date is not None else os.path.join("data", dn) \
            for dn in ALLOWED_DATASETS
    }

    queries_path = os.path.join(
        work_dir, 
        dataset_name_2_relative_path[dataset],
        "queries.jsonl")
    
    qrel_path = os.path.join(
        work_dir,
        dataset_name_2_relative_path[dataset],
        "qrels/test.tsv" if dataset != "msmarco" else "qrels/dev.tsv"
    )
    qrels = read_qrels(qrel_path=qrel_path)


    index, id_map, id2raw = load_index(index_folder)

    model, tokenizer = init_model(retrieval_model, device = DEVICE)

    prefix_ = model_name_2_prefix.get(retrieval_model)
    if not prefix_:
        prefix = None
    else: prefix = prefix_["query"]

    # load queries
    with open(queries_path) as f:
        queries = [json.loads(line) for line in f]
        queries = [line for line in queries if line["_id"] in qrels]


    queries_texts = [line["text"] for line in queries]
    queries_ids = [line["_id"] for line in queries]

    k = 1000
    all_hits = {}
    for i in tqdm(range(0, len(queries), batch_size), desc = f"Searching ({dataset})"):
        batch_queries = queries_texts[i:i+batch_size]
        batch_queries_ids = queries_ids[i:i+batch_size]


        batch_queries_embeddings = text_embedding_batch(batch = batch_queries, model = model, 
                                                        tokenizer = tokenizer, model_name = retrieval_model, prefix = prefix, device = DEVICE).cpu().detach().numpy()
        

        batch_search_results = batch_search(embeddings = batch_queries_embeddings, 
                                            q_ids = batch_queries_ids, index = index, k = k, id_map = id_map, id2raw=id2raw)
        
        all_hits.update(batch_search_results)


    predictions_metadata = process_search_results(
        queries_ids = queries_ids,
        all_hits = all_hits
    )
    all_search_results = [predictions_metadata["full"][query_id] for query_id in queries_ids]


    predictions = convert_to_pytrec_eval_format(queries = queries_ids, all_search_results=all_search_results, type = "prediction")
    evaluation_result = evaluate(qrels = qrels, results = predictions, k_values = [5, 10, 50, 100, 1000])
    mrr_result = mrr(qrels = qrels, results = predictions, k_values = [5, 10, 50, 100, 1000])

    print(evaluation_result, mrr_result)

    if retrieval_metadata_path is not None:
        with open(os.path.join(retrieval_metadata_path, f"{dataset}__{retrieval_model}.json"), "w") as f:
            json.dump(predictions_metadata, f)


if __name__ == "__main__":
    main()