import os, hydra, json, sys, time, openai
sys.path.append("../../steps/utils")
sys.path.append("./steps/utils")
from omegaconf import DictConfig
from openai_utils import init_client, OPENAI_CLIENT
from utils.qa_prompt import QA_SYSTEM_PROMPT, QA_USER_PROMPT, QA_SYSTEM_PROMPT_WITHOUT_CONTEXT, QA_USER_PROMPT_WITHOUT_CONTEXT
from utils.qa_eval_metrics import run_bertscore_evaluation, run_minicheck_based_evaluation, run_llm_based_evaluation, run_llm_based_evaluation_keypoints
from tqdm import tqdm


def create_enumerated_list(texts):
    if not texts: return ""
    return '\n\n\n'.join(f"Context #{i+1}. {text}" for i, text in enumerate(texts))

def generate_answer(query, 
                    contexts, 
                    use_retrieval_contexts = True,
                    max_num_contexts = 20):
    time.sleep(2)
    qa_user_prompt = QA_USER_PROMPT if use_retrieval_contexts else QA_USER_PROMPT_WITHOUT_CONTEXT
    qa_system_prompt = QA_SYSTEM_PROMPT if use_retrieval_contexts else QA_SYSTEM_PROMPT_WITHOUT_CONTEXT

    str_context = create_enumerated_list([line["contents"] for line in contexts[:max_num_contexts]]) if use_retrieval_contexts else ""

    user_prompt = qa_user_prompt.replace("[ADD CONTEXT HERE]", str_context)
    user_prompt = user_prompt.replace("[ADD QUESTION HERE]", query)

    try:
        resp = OPENAI_CLIENT["client"].chat.completions.create(
            model=OPENAI_CLIENT["model"],
            messages=[
                {
                    "role": "system",
                    "content": qa_system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.1,
            max_tokens = 100,
        )

        result = resp.choices[0].message.content.strip()

        return result
    except openai.RateLimitError as e:
        return "Could not answer this question due to an error"

@hydra.main(version_base=None, config_path="../../conf/evaluation", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig):
    retrieval_model = cfg.general.retrieval_model
    retrieval_metadata_folder = cfg.general.retrieval_metadata_path
    work_dir = cfg.general.work_dir
    dataset = cfg.general.dataset
    max_num_contexts = cfg.qa.max_num_contexts
    outfolder = cfg.qa.outfolder
    use_retrieval_contexts = cfg.qa.use_retrieval_contexts
    minicheck_ckpt_path = cfg.qa.minicheck_ckpt_path
    eval_only = cfg.qa.eval_only
    openai_model_name = cfg.qa.openai_model_name

    local_llm_port = cfg.qa.local_llm_port
    local_llm_model = cfg.qa.local_llm_model

    eval_local_llm_port = cfg.qa.eval_local_llm_port
    eval_local_llm_model = cfg.qa.eval_local_llm_model

    openai_api_key = os.getenv("OPENAI_API_KEY")

    dataset_name_2_relative_path = {
        "irb": "data/irb",
        "irb_1_citations": "data/irb/1_citations",
        "irb_2_citations": "data/irb/2_citations",
        "irb_3_citations": "data/irb/3_citations",
        "irb_2005_2010":"data/irb/2005_2010",
        "irb_2010_2015":"data/irb/2010_2015",
        "irb_2015_2020":"data/irb/2015_2020",
        "irb_2020_2025":"data/irb/2020_2025",
        "irb_2025_2030":"data/irb/2025_2030",

        "irb_new": "data/irb_new",
        "irb_new_1_citations": "data/irb_new/1_citations",
        "irb_new_2_citations": "data/irb_new/2_citations",
        "irb_new_3_citations": "data/irb_new/3_citations",

        "irb_new_2025_2030": "data/irb_new/2025_2030"
    }

    outfile_pred = os.path.join(outfolder, f"{dataset}__{retrieval_model}__rc{int(use_retrieval_contexts)}.hyps.txt")
    outfile_gt = os.path.join(outfolder, f"{dataset}__{retrieval_model}__rc{int(use_retrieval_contexts)}.refs.txt")


    experiment_name = None
    if dataset in ["irb_1_citations", "irb_2_citations", "irb_3_citations", "irb_2005_2010", "irb_2010_2015", "irb_2015_2020", "irb_2020_2025", "irb_2025_2030"]:
        experiment_name = f"irb__{retrieval_model}"
    elif dataset in ["irb_new_1_citations", "irb_new_2_citations", "irb_new_3_citations"]:
        experiment_name = f"irb_new__{retrieval_model}"
    else:
        experiment_name = f"{dataset}__{retrieval_model}"

    retrieval_metadata_path = os.path.join(retrieval_metadata_folder, f"{experiment_name}.json")

    queries_path = os.path.join(
        work_dir, 
        dataset_name_2_relative_path[dataset],
        "queries.jsonl")
    
    groundtruth_answers_path = os.path.join(
        work_dir,
        dataset_name_2_relative_path[dataset],
        "answers.jsonl"
    )

    # load queries
    with open(queries_path) as f:
        queries = [json.loads(line) for line in f]
    
    with open(groundtruth_answers_path) as f:
        groundtruth_answers = [json.loads(line) for line in f]

    if eval_only is False:

        init_client(openai_api_key, 
                    openai_model_name = openai_model_name,
                    local = local_llm_port is not None, 
                    port = local_llm_port, 
                    model_name = local_llm_model)


        if use_retrieval_contexts:
            assert os.path.exists(retrieval_metadata_path)

            with open(retrieval_metadata_path) as f:
                retrieval_metadata = json.load(f)

            queries = [line for line in queries if line["_id"] in retrieval_metadata]
            groundtruth_answers = [line for line in groundtruth_answers if line["_id"] in retrieval_metadata]

        else: retrieval_metadata = {}

        
            

        groundtruths_preds = []
        for query, gt_answer in tqdm(zip(queries, groundtruth_answers), desc = "Generating answers"):
            query_text = query.get("text")
            query_id = query.get("_id")

            assert gt_answer.get("_id") == query_id

            gt_answer_text = gt_answer.get("text")
            gt_answer_text = gt_answer_text if isinstance(gt_answer_text, str) else "--__--".join(gt_answer_text)

            contexts = retrieval_metadata.get(query_id)

            answer = generate_answer(
                query = query_text,
                contexts = contexts,
                use_retrieval_contexts = use_retrieval_contexts,
                max_num_contexts = max_num_contexts
            )

            to_append = [gt_answer_text, answer]

            groundtruths_preds.append(to_append)
            if len(groundtruths_preds) == 100: break

        with open(outfile_pred, "w") as f:
            for line in groundtruths_preds:
                pred = line[1]
                f.write(pred.replace("\n", " "))
                f.write("\n")

        with open(outfile_gt, "w") as f:
            for line in groundtruths_preds:
                gt = line[0]
                f.write(gt.replace("\n", " "))
                f.write("\n")

    assert os.path.exists(outfile_gt) and os.path.exists(outfile_pred)
    # print("====BERTSCORE====")
    # run_bertscore_evaluation(outfile_pred, outfile_gt)
    # print("====MINICHECK====")
    evaluation_metadata_file = os.path.join(outfolder, f"{dataset}__{retrieval_model}__rc{int(use_retrieval_contexts)}.evaluation_metadata.txt")
    # run_minicheck_based_evaluation(outfile_pred, outfile_gt, minicheck_ckpt_path, evaluation_metadata_file)


    # re-init client
    init_client(
        "no-key", 
        local = eval_local_llm_port is not None, 
        port = eval_local_llm_port, 
        model_name = eval_local_llm_model
    )
    run_llm_based_evaluation_keypoints(
        queries = [q.get("text") for q in queries][:100],
        outfile_pred=outfile_pred,
        outfile_gt = outfile_gt,
        OPENAI_CLIENT = OPENAI_CLIENT,
        evaluation_metadata_file=evaluation_metadata_file
    )


if __name__ == "__main__":
    main()