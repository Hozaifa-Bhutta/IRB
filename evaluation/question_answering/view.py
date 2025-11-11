import json, hydra, os, itertools
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from evaluation.question_answering.utils.allowed_datasets import ALLOWED_DATASETS
from evaluation.question_answering.eval import metadata_folder_name_creation, data_relative_path
from steps.utils.generic import read_json_or_jsonl
from typing import List, Dict


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

def get_average_performance(eval_res_list: List[List[float]]):
    correct, incorrect, not_attempted = [], [], []
    accuracy = []

    for corr, incorr, not_att in eval_res_list:
        correct.append(corr)
        incorrect.append(incorr)
        not_attempted.append(not_att)

        if corr == 1:
            accuracy.append(1)
        else: accuracy.append(0)
    
    formatted_output = f"CORRECT: {round(np.mean(correct), 3)} INCORRECT: {round(np.mean(incorrect), 3)} NOT_ATTEMPTED: {round(np.mean(not_attempted), 3)} ACCURACY: {round(np.mean(accuracy), 3)}"
    return formatted_output


def filter_by_num_keypoints(att: Dict, choice: str = "single"):
    assert choice in ["single", "multi"]
    nkp = "single" if att.get("num_keypoints") == 1 else "multi"

    return nkp == choice


def filter_by_language(att: Dict, choice: str = "english_only"):
    assert choice in ["english_only", "multilingual"]

    langs = (att.get("evidence_attr").get("langs"))
    langs = list(itertools.chain.from_iterable(langs))

    _type = None
    if any([l != "en" for l in langs]):
        _type = "multilingual"
    else: _type = "english_only"

    return _type == choice


def filter_by_freshness(att: Dict, choice: int = 2024):
    create_timestamp = att.get("wiki_create_timestamp")
    published_dates = att.get("evidence_attr", {}).get("published_dates")

    create_timestamp = int(create_timestamp[:4])
    published_dates = [int(item[:4]) for item in list(itertools.chain.from_iterable(published_dates))]

    all_years = published_dates + [create_timestamp]

    min_year = min(all_years)

    return min_year == choice


def filter_by_topic(att: Dict, choice: str = "History_and_Society"):
    topics = att.get("topics")

    return any([choice in top for top in topics])


def filter_by_numhop(att: Dict, choice: int = "single"):
    assert choice in ["single", "multi"]
    nh = "single" if att.get("num_hops") == 1 else "multi"

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


def show_results(eval_results, configurations, attributes):
    print("General performance:", get_average_performance([item for item in eval_results if item]))
    for config_dict in configurations:
        temp = []
        for att, eval_res in zip(attributes, eval_results):
            if eval_res is None: continue
            if general_filter_func(att, config_dict): temp.append(eval_res)

        formatted_output = get_average_performance(temp)
        print(config_dict, f"Support: {len(temp)}", formatted_output)

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
        for line in attributes:
            query_id = line["_id"]
            eval_res = eval_results_.get(query_id, {})

            if eval_res:
                corr, incorr, not_att = float(eval_res.get("CORRECT", 0)), float(eval_res.get("INCORRECT", 0)), float(eval_res.get("NOT_ATTEMPTED", 0))
                eval_results.append([corr, incorr, not_att])
            else: eval_results.append(None)

    print("===Overall===")
    show_results(eval_results, configurations, attributes)
    print("=============")

    print("===Samples whose retrieval results are CORRECT===")
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
    show_results(eval_results_correct_retrieval, configurations, attributes)

    print("=============")

    print("===Samples whose retrieval results are PARTIALLY-CORRECT===")
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
    show_results(eval_results_partcorrect_retrieval, configurations, attributes)
    print("=============")

    print("===Samples whose retrieval results are INCORRECT===")
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
    show_results(eval_results_incorrect_retrieval, configurations, attributes)

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