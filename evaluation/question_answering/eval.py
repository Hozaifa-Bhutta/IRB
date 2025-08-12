import os, hydra, json, sys
sys.path.append("../../steps/utils")
sys.path.append("./steps/utils")
from omegaconf import DictConfig
from openai_utils import init_client, OPENAI_CLIENT
from utils.qa_prompt import QA_SYSTEM_PROMPT, QA_USER_PROMPT
from tqdm import tqdm


def create_enumerated_list(texts):
    return '\n'.join(f"Context #{i+1}. {text}" for i, text in enumerate(texts))

def generate_answer(query, 
                    contexts, 
                    qa_system_prompt = QA_SYSTEM_PROMPT, 
                    qa_user_prompt = QA_USER_PROMPT):
    str_context = create_enumerated_list([line["contents"] for line in contexts])

    user_prompt = qa_user_prompt.replace("[ADD CONTEXT HERE]", str_context)
    user_prompt = qa_user_prompt.replace("[ADD QUESTION HERE]", query)
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

@hydra.main(version_base=None, config_path="../../conf/evaluation", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig):
    retrieval_model = cfg.general.retrieval_model
    retrieval_metadata_folder = cfg.general.retrieval_metadata_path
    work_dir = cfg.general.work_dir
    dataset = cfg.general.dataset
    outfolder = cfg.qa.outfolder

    local_llm_port = cfg.qa.local_llm_port
    local_llm_model = cfg.qa.local_llm_model

    openai_api_key = os.getenv("OPENAI_API_KEY")

    dataset_name_2_relative_path = {
        "irb": "data/irb",
        "irb_1_citations": "data/irb/1_citations",
        "irb_2_citations": "data/irb/2_citations",
        "irb_3_citations": "data/irb/3_citations",
    }

    init_client(openai_api_key, 
                local = local_llm_port is not None, 
                port = local_llm_port, 
                model_name = local_llm_model)


    if dataset in ["irb_1_citations", "irb_2_citations", "irb_3_citations"]:
        retrieval_metadata_path = os.path.join(retrieval_metadata_folder, f"irb__{retrieval_model}.json")
    else:
        retrieval_metadata_path = os.path.join(retrieval_metadata_folder, f"{dataset}__{retrieval_model}.json")

    with open(retrieval_metadata_path) as f:
        retrieval_metadata = json.load(f)

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
        queries = [line for line in queries if line["_id"] in retrieval_metadata]
    
    with open(groundtruth_answers_path) as f:
        groundtruth_answers = [json.loads(line) for line in f]
        groundtruth_answers = [line for line in groundtruth_answers if line["_id"] in retrieval_metadata]
        

    groundtruths_preds = []
    for query, gt_answer in tqdm(zip(queries, groundtruth_answers), desc = "Generating answers"):
        query_text = query.get("text")
        query_id = query.get("_id")

        assert gt_answer.get("_id") == query_id

        gt_answer_text = gt_answer.get("text")

        contexts = retrieval_metadata.get(query_id)

        answer = generate_answer(
            query = query_text,
            contexts = contexts
        )

        to_append = [gt_answer_text, answer]

        groundtruths_preds.append(to_append)
        if len(groundtruths_preds) == 100: break

    outfile_pred = os.path.join(outfolder, f"hyps.txt")
    with open(outfile_pred, "w") as f:
        for line in groundtruths_preds:
            pred = line[1]
            f.write(pred.replace("\n", " "))
            f.write("\n")

    outfile_gt = os.path.join(outfolder, f"refs.txt")
    with open(outfile_gt, "w") as f:
        for line in groundtruths_preds:
            gt = line[0]
            f.write(gt.replace("\n", " "))
            f.write("\n")


if __name__ == "__main__":
    main()