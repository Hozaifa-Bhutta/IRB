import json, os, hydra
from omegaconf import DictConfig
from evaluation.question_answering.eval import metadata_folder_name_creation

@hydra.main(version_base=None, config_path="../../conf/evaluation", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig):
    llm_model_name = cfg.adv_collection.llm_model_name # llm_model_name to collect against
    retrieval_model = cfg.adv_collection.retrieval_model
    outfolder = cfg.qa.outfolder
    dataset = cfg.general.dataset

    _metadata_folder_with_retrieval = metadata_folder_name_creation(
        dataset = dataset, llm_model_name = llm_model_name, retrieval_model = retrieval_model,
        use_retrieval_contexts = 1, use_chunk = 0
    )
    _metadata_folder_without_retrieval = metadata_folder_name_creation(
        dataset = dataset, llm_model_name = llm_model_name, retrieval_model = retrieval_model,
        use_retrieval_contexts = 0, use_chunk = 0
    )
    eval_result_with_retrieval_outfile = os.path.join(outfolder, _metadata_folder_with_retrieval, "eval_result.json")
    eval_result_without_retrieval_outfile = os.path.join(outfolder, _metadata_folder_without_retrieval, "eval_result.json")

    print(f"Eval result file: \n{eval_result_with_retrieval_outfile} (with retrieval)\n{eval_result_without_retrieval_outfile} (without)")

    assert os.path.exists(eval_result_with_retrieval_outfile) and \
    os.path.exists(eval_result_without_retrieval_outfile)

    with open(eval_result_with_retrieval_outfile) as f:
        eval_result_with_retrieval = json.load(f)

    with open(eval_result_without_retrieval_outfile) as f:
        eval_result_without_retrieval = json.load(f)

    hard_query_id_with_retrieval = set([query_id for query_id in eval_result_with_retrieval if not eval_result_with_retrieval[query_id].get("CORRECT", 0.0) >= 0.99])
    hard_query_id_without_retrieval = set([query_id for query_id in eval_result_without_retrieval if not eval_result_without_retrieval[query_id].get("CORRECT", 0.0) >= 0.99])
    print(f"There are {len(hard_query_id_with_retrieval)} hard queries (with retrieval)!")
    print(f"There are {len(hard_query_id_without_retrieval)} hard queries (without retrieval)!")


    hard_query_id = hard_query_id_with_retrieval.intersection(hard_query_id_without_retrieval)

    # include 1-hop version of 2-hop questions that are retained
    to_update = set()
    for query_id in hard_query_id:
        if "~" not in query_id and query_id.endswith("--2"):
            one_hop_version_query_id = query_id[:-3] + "--1"
            to_update.add(one_hop_version_query_id)
    hard_query_id.update(to_update)
    hard_query_id = list(hard_query_id)

    print(f"There are {len(hard_query_id)} hard queries!")

    outfile = os.path.join(outfolder, f"adv_collected_query_id_{llm_model_name}.json")

    with open(outfile, "w") as f:
        json.dump(hard_query_id, f)

    


if __name__ == "__main__":
    main()