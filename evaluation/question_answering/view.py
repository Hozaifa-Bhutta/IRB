import json, hydra, os, itertools
import numpy as np
from omegaconf import DictConfig
from utils.allowed_datasets import ALLOWED_DATASETS
from typing import List, Dict


def data_relative_path(dataset_name, dataset_date = None, subset = None):
    if dataset_date and subset:
        return os.path.join("benchmarks", dataset_date, dataset_name, subset)
    elif dataset_date and not subset:
        return os.path.join("benchmarks", dataset_date, dataset_name)
    else:
        raise NotImplemented
    

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


def general_filter_func(att: Dict, choice_dict: Dict[str, str]):
    # the keys are 'language', 'freshness', 'topic', 'keypoints'

    filter_mapper = {
        "language": filter_by_language,
        "freshness": filter_by_freshness,
        "topic": filter_by_topic,
        "keypoints": filter_by_num_keypoints
    }

    return all([filter_mapper[k](att, v) for k, v in choice_dict.items()])



@hydra.main(version_base=None, config_path="../../conf/evaluation", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig):
    retrieval_model = cfg.general.retrieval_model
    work_dir = cfg.general.work_dir
    dataset = cfg.general.dataset
    dataset_date = cfg.general.dataset_date
    subset = cfg.qa.subset
    retrieval_metadata_folder = cfg.general.retrieval_metadata_path
    outfolder = cfg.qa.outfolder
    use_retrieval_contexts = cfg.qa.use_retrieval_contexts
    use_chunk = cfg.qa.use_chunk

    configurations = cfg.view.configurations

    # open attributes, queries, and evaluation_metadata file


    dataset_name_2_relative_path = {
        dn: data_relative_path(dn, dataset_date, subset) \
            for dn in ALLOWED_DATASETS
    }

    queries_path = os.path.join(
        work_dir, 
        dataset_name_2_relative_path[dataset],
        "queries.jsonl")
    
    attributes_path = os.path.join(
        work_dir, 
        dataset_name_2_relative_path[dataset],
        "attributes.jsonl")
    
    evaluation_metadata_file = os.path.join(outfolder, f"{dataset}__{retrieval_model}__rc{int(use_retrieval_contexts)}__chunk{use_chunk}.evaluation_metadata.txt")


    with open(queries_path) as f:
        queries = [json.loads(line) for line in f]

    with open(attributes_path) as f:
        attributes = [json.loads(line) for line in f]

    with open(evaluation_metadata_file) as f:
        evaluation_metadata = []
        for line in f:
            corr, incorr, not_att = [float(item) for item in line.split(",")]
            evaluation_metadata.append([corr, incorr, not_att])


    for config_dict in configurations:
        temp = []
        for att, eval_res in zip(attributes, evaluation_metadata):
            if general_filter_func(att, config_dict): temp.append(eval_res)

        formatted_output = get_average_performance(temp)
        print(config_dict, f"Support: {len(temp)}", formatted_output)

if __name__ == "__main__":
    main()