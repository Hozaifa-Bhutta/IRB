import json, os, hydra, pytrec_eval
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from evaluation.retrieval.utils.allowed_datasets import ALLOWED_DATASETS
from steps.utils.generic import read_json_or_jsonl
from evaluation.question_answering.view import general_filter_func, CACHE, filter_by_hardness
from evaluation.retrieval.eval_dense import read_qrels, convert_to_pytrec_eval_format, evaluate


def get_average_performance(eval_results, max_num_contexts, split):
    ndcg_all = []
    recall_all = []
    for line in eval_results:
        assert len(line) == 1
        query_id = list(line.keys())[0]

        ndcg_val = line[query_id][f"ndcg_cut_{max_num_contexts}"]
        recall_val = line[query_id][f"recall_{max_num_contexts}"]
        ndcg_all.append(ndcg_val)
        recall_all.append(recall_val)
    
    return {
        "split": split,
        "support": len(eval_results),
        "ndcg": np.mean(ndcg_all),
        "recall": np.mean(recall_all)
    }

def show_results(eval_results, configurations, attributes, max_num_contexts):
    general_performance = get_average_performance([item for item in eval_results if item], max_num_contexts, split = "general")
    general_performance_hard = get_average_performance([item for item, att in zip(eval_results, attributes) if item and filter_by_hardness(att, True)], 
                                                       max_num_contexts, split = "general_hard")

    all_performances = []
    for config_dict in configurations:
        temp = []
        for att, eval_res in zip(attributes, eval_results):
            if eval_res is None: continue
            if general_filter_func(att, config_dict): 
                temp.append(eval_res)

        config_performance = get_average_performance(temp, max_num_contexts, split = str(config_dict))
        all_performances.append(config_performance)

    all_performances.append(general_performance)
    all_performances.append(general_performance_hard)

    df = pd.DataFrame(columns=["split", "support", "ndcg", "recall"], data = all_performances)

    return df


@hydra.main(version_base=None, config_path="../../conf/evaluation/", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig):
    retrieval_model = cfg.general.retrieval_model
    index_folder = cfg.general.index_folder
    work_dir = cfg.general.work_dir
    dataset = cfg.general.dataset
    dataset_date = cfg.general.dataset_date
    retrieval_metadata_path = cfg.general.retrieval_metadata_path

    configurations = cfg.view.configurations
    
    max_num_contexts = cfg.qa.max_num_contexts
    outfolder = cfg.qa.outfolder

    adv_collect_llm_model_name = cfg.adv_collection.llm_model_name
    try:
        with open(os.path.join(outfolder, f"adv_collected_query_id_{adv_collect_llm_model_name}.json")) as f:
            CACHE["hard_query_ids"] = set(json.load(f))
    except Exception: CACHE["hard_query_ids"] = set([])


    dataset_name_2_relative_path = {
        dn: os.path.join("benchmarks", dataset_date, dn) if dataset_date is not None else os.path.join("data", dn) \
            for dn in ALLOWED_DATASETS
    }

    attributes_path = os.path.join(
        work_dir, 
        dataset_name_2_relative_path[dataset],
        "attributes.jsonl")
    
    qrels_path = os.path.join(
        work_dir, 
        dataset_name_2_relative_path[dataset],
        "qrels",
        "test.tsv")
    
    attributes = read_json_or_jsonl(attributes_path)
    qrels = read_qrels(qrels_path)

    for mode in ["", "__reranked"]:
        try:
            retrieval_metadata = read_json_or_jsonl(os.path.join(retrieval_metadata_path, f"{dataset}__{retrieval_model}{mode}.json"))
        except Exception: continue

        query_retrieval_results = convert_to_pytrec_eval_format(
                [att["_id"] for att in attributes],
                [retrieval_metadata["full"][att["_id"]][:] for att in attributes], 
                type = "prediction")
        # print(evaluate(qrels, results = query_retrieval_results, k_values=[5, 10]))


        eval_results = []
        for att in attributes:
            query_id = att["_id"]

            if query_id not in qrels:
                eval_results.append(None)
                continue
            
            query_retrieval_results = convert_to_pytrec_eval_format(
                [query_id],
                [retrieval_metadata["full"][query_id][:]], 
                type = "prediction")

            query_qrels = {query_id: qrels[query_id]}

            evaluator =  pytrec_eval.RelevanceEvaluator(query_qrels, {f'ndcg_cut.{max_num_contexts}', f"recall.{max_num_contexts}"})
            eval_results.append(evaluator.evaluate(query_retrieval_results))

        eval_results_df = show_results(eval_results, configurations, attributes, max_num_contexts)
        eval_results_df.to_csv(os.path.join(os.environ["RESULT_DIR"], f"retriever{mode}.csv"), index=False)

if __name__ == "__main__":
    main()