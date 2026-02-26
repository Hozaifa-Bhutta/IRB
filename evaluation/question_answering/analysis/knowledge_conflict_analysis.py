import json, os, hydra
from omegaconf import DictConfig
from evaluation.question_answering.eval import metadata_folder_name_creation
from steps.utils.generic import read_json_or_jsonl
from evaluation.question_answering.view import check_retrieval_correctness

@hydra.main(version_base=None, config_path="../../../conf/evaluation", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig):
    retrieval_model = cfg.general.retrieval_model
    retrieval_metadata_folder = cfg.general.retrieval_metadata_path
    work_dir = cfg.general.work_dir
    dataset = cfg.general.dataset
    dataset_date = cfg.general.dataset_date
    wikidump_date = cfg.general.wikidump_date

    max_num_contexts = cfg.qa.max_num_contexts
    outfolder = cfg.qa.outfolder
    eval_only = cfg.qa.eval_only
    use_chunk = cfg.qa.use_chunk
    qa_llm_model_name = cfg.qa.qa_llm_model_name
    eval_llm_model_name = cfg.qa.eval_llm_model_name
    eval_llm2_model_name = cfg.qa.eval_llm2_model_name


    metadata_folder_name_with_retrieval = metadata_folder_name_creation(
        dataset = dataset, llm_model_name = qa_llm_model_name, retrieval_model = retrieval_model,
        use_retrieval_contexts = 1, use_chunk = use_chunk
    )
    metadata_folder_name_without_retrieval = metadata_folder_name_creation(
        dataset = dataset, llm_model_name = qa_llm_model_name, retrieval_model = retrieval_model,
        use_retrieval_contexts = 0, use_chunk = use_chunk
    )

    eval_metadata_with_retrieval = read_json_or_jsonl(os.path.join(outfolder, metadata_folder_name_with_retrieval, "eval_metadata.json"))
    eval_metadata_without_retrieval = read_json_or_jsonl(os.path.join(outfolder, metadata_folder_name_without_retrieval, "eval_metadata.json"))

    eval_result_with_retrieval = read_json_or_jsonl(os.path.join(outfolder, metadata_folder_name_with_retrieval, "eval_result.json"))
    eval_result_without_retrieval = read_json_or_jsonl(os.path.join(outfolder, metadata_folder_name_without_retrieval, "eval_result.json"))


    with_retrieval_predictions = eval_metadata_with_retrieval["predictions"]
    without_retrieval_predictions = eval_metadata_without_retrieval["predictions"]

    for query_id in with_retrieval_predictions:
        wr_pred = with_retrieval_predictions[query_id]
        wor_pred = without_retrieval_predictions[query_id]

        wr_res = eval_result_with_retrieval[query_id]
        wor_res = eval_result_without_retrieval[query_id]

        if check_retrieval_correctness()