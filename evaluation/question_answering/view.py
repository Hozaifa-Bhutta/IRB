import json, hydra, os, itertools
import numpy as np
from omegaconf import DictConfig
from utils.allowed_datasets import ALLOWED_DATASETS
from typing import List


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

    

    # next, view evaluation results based on different attributes
    # performance by number of keypoints
    _subset = {}
    for att, eval_res in zip(attributes, evaluation_metadata):
        if att.get("num_keypoints") not in _subset: _subset[att.get("num_keypoints")] = []
        _subset[att.get("num_keypoints")].append(eval_res)

    for nkp in sorted(_subset.keys()):
        eval_res_list = _subset[nkp]
        print(f"Num keypoints: {nkp}. Performance:", get_average_performance(eval_res_list))

    # performance on multi-lingual vs only english
    _subset = {"multilingual": [], "english_only": []}
    for att, eval_res in zip(attributes, evaluation_metadata):
        langs = (att.get("evidence_attr").get("langs"))
        langs = list(itertools.chain.from_iterable(langs))

        if any([l != "en" for l in langs]):
            _subset["multilingual"].append(eval_res)
        else:
            _subset["english_only"].append(eval_res)

    for lang_mode in sorted(_subset.keys()):
        eval_res_list = _subset[lang_mode]
        print(f"Language mode: {lang_mode}. Performance:", get_average_performance(eval_res_list))


if __name__ == "__main__":
    main()