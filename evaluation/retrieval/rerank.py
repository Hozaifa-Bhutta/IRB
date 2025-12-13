# currently only supports cohere reranker, code mostly from 'https://aws.amazon.com/blogs/machine-learning/cohere-rerank-3-5-is-now-available-in-amazon-bedrock-through-rerank-api/'
import boto3, json, hydra, os
from omegaconf import DictConfig
from evaluation.retrieval.eval_dense import read_qrels
from evaluation.retrieval.utils.allowed_datasets import ALLOWED_DATASETS
from steps.utils.generic import read_json_or_jsonl
from evaluation.retrieval.eval_dense import mrr, evaluate, convert_to_pytrec_eval_format
from tqdm import tqdm

region = "us-east-1"

bedrock_agent_runtime = boto3.client('bedrock-agent-runtime',region_name=region)

modelId = "cohere.rerank-v3-5:0"
model_package_arn = f"arn:aws:bedrock:{region}::foundation-model/{modelId}"

NUM_CANDIDATES = 100


def rerank_text_helper(text_query, text_sources, num_results, model_package_arn):
    response = bedrock_agent_runtime.rerank(
        queries=[
            {
                "type": "TEXT",
                "textQuery": {
                    "text": text_query
                }
            }
        ],
        sources=text_sources,
        rerankingConfiguration={
            "type": "BEDROCK_RERANKING_MODEL",
            "bedrockRerankingConfiguration": {
                "numberOfResults": num_results,
                "modelConfiguration": {
                    "modelArn": model_package_arn,
                }
            }
        }
    )
    return response['results']

def rerank(text_query, chunks, chunks_ids):
    text_sources = []
    for text in chunks:
        text_sources.append({
            "type": "INLINE",
            "inlineDocumentSource": {
                "type": "TEXT",
                "textDocument": {
                    "text": text,
                }
            }
        })

    response = rerank_text_helper(text_query, text_sources, len(chunks), model_package_arn)
    reranked_chunks = []
    temp = {}
    for line in response:
        index = line["index"]
        score = line["relevanceScore"]

        chunk_id = chunks_ids[index]
        docid, _ = chunk_id.split("--__--")

        reranked_chunks.append({"id": chunk_id, "score": score})

        if docid not in temp: temp[docid] = 0
        temp[docid] = max(temp[docid], score)

    reranked_docs = list(sorted([{"docid": k, "score": v} for k,v in temp.items()], key = lambda x: -x["score"]))

    return reranked_chunks, reranked_docs


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
    
    corpus_path= os.path.join(
        work_dir, 
        dataset_name_2_relative_path[dataset],
        "corpus.jsonl"
    )
    
    qrel_path = os.path.join(
        work_dir,
        dataset_name_2_relative_path[dataset],
        "qrels/test.tsv" if dataset != "msmarco" else "qrels/dev.tsv"
    )
    qrels = read_qrels(qrel_path=qrel_path)

    queries = read_json_or_jsonl(queries_path)
    first_stage_retrieval_metadata = read_json_or_jsonl(os.path.join(retrieval_metadata_path, f"{dataset}__{retrieval_model}.json"))

    queries_ids = [line["_id"] for line in queries]

    # load corpus
    with open(corpus_path) as f:
        docid2info = {}
        for line in f:
            jline = json.loads(line)
            docid = jline["_id"]
            text = jline["text"]
            published_date = jline["published_date"]
            docid2info[docid] = {"text": text, "published_date": published_date}

    reranked_metadata = {"chunk": {}, "full": {}}

    for line in tqdm(queries, desc = "Reranking"):
        query_id = line["_id"]
        query = line["text"]

        chunks_ids = [item["id"] for item in first_stage_retrieval_metadata["chunk"][query_id]]
        chunks = [item["contents"] for item in first_stage_retrieval_metadata["chunk"][query_id]][:NUM_CANDIDATES]

        reranked_chunks, reranked_docs = rerank(text_query = query, chunks = chunks, chunks_ids = chunks_ids)
        reranked_metadata["chunk"][query_id] = reranked_chunks
        reranked_metadata["full"][query_id] = reranked_docs


    all_search_results = [reranked_metadata["full"][query_id] for query_id in queries_ids]


    predictions = convert_to_pytrec_eval_format(queries = queries_ids, all_search_results=all_search_results, type = "prediction")
    evaluation_result = evaluate(qrels = qrels, results = predictions, k_values = [5, 10, 50, 100, 1000])
    mrr_result = mrr(qrels = qrels, results = predictions, k_values = [5, 10, 50, 100, 1000])

    print(evaluation_result, mrr_result)

    if retrieval_metadata_path is not None:
        with open(os.path.join(retrieval_metadata_path, f"{dataset}__{retrieval_model}__reranked.json"), "w") as f:
            json.dump(reranked_metadata, f)


if __name__ == "__main__":
    main()