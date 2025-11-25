import json, hydra, os, itertools
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from evaluation.question_answering.utils.allowed_datasets import ALLOWED_DATASETS
from evaluation.question_answering.eval import metadata_folder_name_creation, data_relative_path
from steps.utils.generic import read_json_or_jsonl
from typing import List, Dict
from tqdm import tqdm


def read_qrels(qrel_path):
    _qrels = pd.read_csv(qrel_path, sep='\t').to_dict("records")
    
    metadata = {}
    for line in _qrels:
        query_id = str(line["query-id"])
        doc_id = str(line["corpus-id"])
        score = line["score"]

        if query_id not in metadata:
            metadata[query_id] = {}

        metadata[query_id][doc_id] = int(score)

    return metadata

def get_average_performance(eval_res_list: List[List[float]], split: str):
    correct, incorrect, not_attempted = [], [], []
    truthfulness = []

    for corr, incorr, not_att in eval_res_list:
        correct.append(corr)
        incorrect.append(incorr)
        not_attempted.append(not_att)


        truthfulness.append(corr - incorr)
    
    return {
        "split": split,
        "support": len(eval_res_list),
        "correct": round(np.mean(correct), 3),
        "incorrect": round(np.mean(incorrect), 3),
        "not_attempted": round(np.mean(not_attempted), 3),
        "truthfulness": round(np.mean(truthfulness), 3)
    }


def filter_by_num_keypoints(att: Dict, choice: int = 1):
    nkp = att.get("num_keypoints")

    return nkp == choice


def filter_by_language(att: Dict, choice: str = "english_only"):
    assert choice in ["english_only", "multilingual"]

    langs = (att.get("evidence_attr").get("langs"))
    langs = list(langs)

    _type = None
    if any([l != "en" for l in langs]):
        _type = "multilingual"
    else: _type = "english_only"

    return _type == choice


def filter_by_freshness(att: Dict, choice: int = 2024):
    create_timestamp = att.get("wiki_create_timestamp")
    published_dates = att.get("evidence_attr", {}).get("published_dates")

    create_timestamp = int(create_timestamp[:4])
    published_dates = [int(item[:4]) for item in list(published_dates)]

    all_years = published_dates + [create_timestamp]

    min_year = min(all_years)

    return min_year == choice


def filter_by_topic(att: Dict, choice: str = "History_and_Society"):
    topics = att.get("topics")

    return any([choice in top for top in topics])


def filter_by_numhop(att: Dict, choice: int = 1):
    nh = att.get("num_hops")

    return nh == choice


def general_filter_func(att: Dict, choice_dict: Dict[str, str]):
    # the keys are 'language', 'freshness', 'topic', 'keypoints'

    filter_mapper = {
        "language": filter_by_language,
        "freshness": filter_by_freshness,
        "topic": filter_by_topic,
        "keypoints": filter_by_num_keypoints,
        "numhops": filter_by_numhop
    }

    return all([filter_mapper[k](att, v) for k, v in choice_dict.items()])

def check_retrieval_correctness(qrels_query, retrieval_metadata_query, num_retrieval_contexts):
    retrieved_doc_ids = set([item["docid"] for item in retrieval_metadata_query[:num_retrieval_contexts]])
    required_docids = qrels_query.keys()

    if all([required_docid in retrieved_doc_ids for required_docid in required_docids]): return "correct"
    elif any([required_docid in retrieved_doc_ids for required_docid in required_docids]): return "partially-correct"
    else: return "wrong"


def show_results(eval_results, configurations, attributes, eval_metadata):
    general_performance = get_average_performance([item for item in eval_results if item], split = "general")

    all_performances = []
    all_reasoning_tokens = []
    for config_dict in configurations:
        temp = []
        config_reasoning_tokens = []
        for att, eval_res in zip(attributes, eval_results):
            query_id = att.get("_id")
            if eval_res is None: continue
            if general_filter_func(att, config_dict): 
                temp.append(eval_res)
                try:
                    reasoning_tokens = eval_metadata["raw_response"][query_id]["usage"]["output_tokens_details"]["reasoning_tokens"]
                except Exception: reasoning_tokens = 0
                config_reasoning_tokens.append(reasoning_tokens)
                all_reasoning_tokens.append(reasoning_tokens)

        config_performance = get_average_performance(temp, split = str(config_dict))
        config_performance["avg_reasoning_tokens"] = np.mean(config_reasoning_tokens)

        all_performances.append(config_performance)

    general_performance["avg_reasoning_tokens"] = np.mean(all_reasoning_tokens)

    all_performances.append(general_performance)

    df = pd.DataFrame(columns=["split", "support", "avg_reasoning_tokens", "correct", "incorrect", "not_attempted", "truthfulness"], data = all_performances)

    return df

    

@hydra.main(version_base=None, config_path="../../conf/evaluation", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig):
    retrieval_model = cfg.general.retrieval_model
    work_dir = cfg.general.work_dir
    dataset = cfg.general.dataset
    dataset_date = cfg.general.dataset_date
    retrieval_metadata_folder = cfg.general.retrieval_metadata_path
    outfolder = cfg.qa.outfolder
    use_retrieval_contexts = cfg.qa.use_retrieval_contexts
    use_chunk = cfg.qa.use_chunk
    max_num_contexts = cfg.qa.max_num_contexts

    qa_llm_model_name = cfg.qa.qa_llm_model_name

    configurations = cfg.view.configurations

    # open attributes, queries, and eval_results file


    dataset_name_2_relative_path = {
        dn: data_relative_path(dn, dataset_date) \
            for dn in ALLOWED_DATASETS
    }

    _metadata_folder = metadata_folder_name_creation(
        dataset = dataset, llm_model_name = qa_llm_model_name, retrieval_model = retrieval_model,
        use_retrieval_contexts = use_retrieval_contexts, use_chunk = use_chunk
    )
    eval_result_outfile = os.path.join(outfolder, _metadata_folder, "eval_result.json")
    eval_metadata_outfile = os.path.join(outfolder, _metadata_folder, "eval_metadata.json")
    
    attributes_path = os.path.join(
        work_dir, 
        dataset_name_2_relative_path[dataset],
        "attributes.jsonl")
    
    answers_path = os.path.join(
         work_dir, 
        dataset_name_2_relative_path[dataset],
        "answers.jsonl"
    )

    queries_path = os.path.join(
        work_dir, 
        dataset_name_2_relative_path[dataset],
        "queries.jsonl"
    )
    
    qrels_path = os.path.join(
        work_dir,
        dataset_name_2_relative_path[dataset],
        "qrels",
        "test.tsv"
    )
    retrieval_metadata_path = os.path.join(
        retrieval_metadata_folder,
        f"{dataset}__{retrieval_model}.json"
    )

    attributes = read_json_or_jsonl(attributes_path)
    answers = read_json_or_jsonl(answers_path)
    queries = read_json_or_jsonl(queries_path)

    # load qrels
    qrels = read_qrels(qrels_path)

    retrieval_metadata = read_json_or_jsonl(retrieval_metadata_path).get("chunk" if use_chunk else "full", {})
    eval_metadata = read_json_or_jsonl(eval_metadata_outfile)

    with open(eval_result_outfile) as f:
        eval_results_ = json.load(f)
        eval_results = []
        for line in tqdm(attributes):
            query_id = line["_id"]
            eval_res = eval_results_.get(query_id, {})

            if eval_res:
                # corr, incorr, not_att = float(eval_res.get("CORRECT", 0)), float(eval_res.get("INCORRECT", 0)), float(eval_res.get("NOT_ATTEMPTED", 0))
                
                # optimistic scheme
                corr = eval_res.get("CORRECT", 0) > 0
                if corr: corr, incorr, not_att = 1, 0, 0
                else:
                    not_att = eval_res.get("NOT_ATTEMPTED", 0) > 0
                    if not_att: corr, incorr, not_att = 0, 0, 1
                    else: corr, incorr, not_att = 0, 1, 0

                eval_results.append([corr, incorr, not_att])
            else: eval_results.append(None)

    eval_results_all = show_results(eval_results, configurations, attributes, eval_metadata)
    eval_results_all.to_csv(os.path.join(os.environ["RESULT_DIR"], "all.csv"), index = False)


    eval_results_correct_retrieval = []
    for i, line in enumerate(attributes):
        query_id = line["_id"]
        if query_id not in qrels or query_id not in retrieval_metadata: 
            eval_results_correct_retrieval.append(None)
            continue
        if check_retrieval_correctness(
            qrels_query = qrels[query_id],
            retrieval_metadata_query = retrieval_metadata[query_id],
            num_retrieval_contexts = max_num_contexts
        ) == "correct":
            eval_results_correct_retrieval.append(eval_results[i])
        else: eval_results_correct_retrieval.append(None)
    eval_results_retrieval_correct = show_results(eval_results_correct_retrieval, configurations, attributes, eval_metadata)
    eval_results_retrieval_correct.to_csv(os.path.join(os.environ["RESULT_DIR"], "retrieval_correct.csv"), index = False)


    eval_results_partcorrect_retrieval = []
    for i, line in enumerate(attributes):
        query_id = line["_id"]
        if query_id not in qrels or query_id not in retrieval_metadata: 
            eval_results_partcorrect_retrieval.append(None)
            continue
        if check_retrieval_correctness(
            qrels_query = qrels[query_id],
            retrieval_metadata_query = retrieval_metadata[query_id],
            num_retrieval_contexts = max_num_contexts
        ) == "partially-correct":
            eval_results_partcorrect_retrieval.append(eval_results[i])
        else: eval_results_partcorrect_retrieval.append(None)
    eval_results_retrieval_partcorrect = show_results(eval_results_partcorrect_retrieval, configurations, attributes, eval_metadata)
    eval_results_retrieval_partcorrect.to_csv(os.path.join(os.environ["RESULT_DIR"], "retrieval_partcorrect.csv"), index = False)


    eval_results_incorrect_retrieval = []
    for i, line in enumerate(attributes):
        query_id = line["_id"]
        if query_id not in qrels or query_id not in retrieval_metadata: 
            eval_results_incorrect_retrieval.append(None)
            continue
        if check_retrieval_correctness(
            qrels_query = qrels[query_id],
            retrieval_metadata_query = retrieval_metadata[query_id],
            num_retrieval_contexts = max_num_contexts
        ) == "wrong":
            eval_results_incorrect_retrieval.append(eval_results[i])
        else: eval_results_incorrect_retrieval.append(None)
    eval_results_retrieval_incorrect = show_results(eval_results_incorrect_retrieval, configurations, attributes, eval_metadata)
    eval_results_retrieval_incorrect.to_csv(os.path.join(os.environ["RESULT_DIR"], "retrieval_incorrect.csv"), index = False)

    prediction_view = []
    for i, line in enumerate(attributes):
        query_id = line["_id"]
        if query_id not in qrels or query_id not in retrieval_metadata: continue

        retrieval_correctness = check_retrieval_correctness(
            qrels_query = qrels[query_id],
            retrieval_metadata_query = retrieval_metadata[query_id],
            num_retrieval_contexts = max_num_contexts
        )

        to_append = [query_id, f"RETRIEVAL: {retrieval_correctness}", queries[i]["text"], eval_metadata["predictions"][query_id], answers[i]["short"], answers[i]["text"], eval_results_.get(query_id), line]
        prediction_view.append(to_append)

    with open(f"prediction_view_{_metadata_folder}.json", "w") as f:
        json.dump(prediction_view, f, indent = 4)


if __name__ == "__main__":
    main()