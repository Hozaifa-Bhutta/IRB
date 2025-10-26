import os, hydra, json, sys, time, openai
sys.path.append("../../steps/utils")
sys.path.append("./steps/utils")
from omegaconf import DictConfig
from openai_utils import init_client, OPENAI_CLIENT
from utils.qa_prompt import QA_SYSTEM_PROMPT, QA_USER_PROMPT, QA_SYSTEM_PROMPT_WITHOUT_CONTEXT, QA_USER_PROMPT_WITHOUT_CONTEXT
from utils.qa_eval_metrics import run_bertscore_evaluation, run_minicheck_based_evaluation, run_llm_based_evaluation, run_llm_based_evaluation_keypoints
from utils.allowed_datasets import ALLOWED_DATASETS
from tqdm import tqdm

# maximum number of samples to run evaluation
NUM_SAMPLE = 1000**2


def create_enumerated_list(texts):
    if not texts: return ""
    truncated_texts = [" ".join(text.split()[:5000]) for text in texts]
    return '\n\n\n'.join(f"Context #{i+1}. {text}" for i, text in enumerate(truncated_texts))

def generate_answer(query, 
                    contexts, 
                    use_retrieval_contexts = True,
                    max_num_contexts = 20):
    # time.sleep(0.5)
    qa_user_prompt = QA_USER_PROMPT if use_retrieval_contexts else QA_USER_PROMPT_WITHOUT_CONTEXT
    qa_system_prompt = QA_SYSTEM_PROMPT if use_retrieval_contexts else QA_SYSTEM_PROMPT_WITHOUT_CONTEXT

    str_context = create_enumerated_list([line["contents"] for line in contexts[:max_num_contexts]]) if use_retrieval_contexts else ""

    user_prompt = qa_user_prompt.replace("[ADD CONTEXT HERE]", str_context)
    user_prompt = user_prompt.replace("[ADD QUESTION HERE]", query)

    print(user_prompt)

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
            max_tokens = 2048,
        )

        result = resp.choices[0].message.content.strip()

        return result
    except openai.RateLimitError as e:
        return "Could not answer this question due to an error"
    

def data_relative_path(dataset_name, dataset_date = None, subset = None):
    if dataset_date and subset:
        return os.path.join("benchmarks", dataset_date, dataset_name, subset)
    elif dataset_date and not subset:
        return os.path.join("benchmarks", dataset_date, dataset_name)
    else:
        raise NotImplemented

@hydra.main(version_base=None, config_path="../../conf/evaluation", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig):
    retrieval_model = cfg.general.retrieval_model
    retrieval_metadata_folder = cfg.general.retrieval_metadata_path
    work_dir = cfg.general.work_dir
    dataset = cfg.general.dataset
    dataset_date = cfg.general.dataset_date
    max_num_contexts = cfg.qa.max_num_contexts
    outfolder = cfg.qa.outfolder
    use_retrieval_contexts = cfg.qa.use_retrieval_contexts
    minicheck_ckpt_path = cfg.qa.minicheck_ckpt_path
    eval_only = cfg.qa.eval_only
    openai_model_name = cfg.qa.openai_model_name
    subset = cfg.qa.subset
    use_chunk = cfg.qa.use_chunk

    local_llm_port = cfg.qa.local_llm_port
    local_llm_model = cfg.qa.local_llm_model

    eval_local_llm_port = cfg.qa.eval_local_llm_port
    eval_local_llm_model = cfg.qa.eval_local_llm_model
    eval_openai_model_name = cfg.qa.eval_openai_model_name

    openai_api_key = os.getenv("OPENAI_API_KEY")

    dataset_name_2_relative_path = {
        dn: data_relative_path(dn, dataset_date, subset) \
            for dn in ALLOWED_DATASETS
    }

    if subset:
        outfile_pred = os.path.join(outfolder, f"{dataset}__{subset}__{retrieval_model}__rc{int(use_retrieval_contexts)}__chunk{use_chunk}.hyps.txt")
        outfile_gt = os.path.join(outfolder, f"{dataset}__{subset}__{retrieval_model}__rc{int(use_retrieval_contexts)}__chunk{use_chunk}.refs.txt")
    else:
        outfile_pred = os.path.join(outfolder, f"{dataset}__{retrieval_model}__rc{int(use_retrieval_contexts)}__chunk{use_chunk}.hyps.txt")
        outfile_gt = os.path.join(outfolder, f"{dataset}__{retrieval_model}__rc{int(use_retrieval_contexts)}__chunk{use_chunk}.refs.txt")


    experiment_name = f"{dataset}__{retrieval_model}"

    retrieval_metadata_path = os.path.join(retrieval_metadata_folder, f"{experiment_name}.json")

    queries_path = os.path.join(
        work_dir, 
        dataset_name_2_relative_path[dataset],
        "queries.jsonl")
    
    corpus_path= os.path.join(
        work_dir, 
        data_relative_path(dataset, dataset_date),
        "corpus.jsonl"
    )
    
    groundtruth_answers_path = os.path.join(
        work_dir,
        dataset_name_2_relative_path[dataset],
        "answers.jsonl"
    )

    # load queries
    with open(queries_path) as f:
        queries = [json.loads(line) for line in f]

    # load corpus
    with open(corpus_path) as f:
        docid2fulltext = {}
        for line in f:
            jline = json.loads(line)
            docid = jline["_id"]
            text = jline["text"]
            docid2fulltext[docid] = text

    
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

            if use_chunk:
                retrieval_metadata = retrieval_metadata["chunk"]
            else:
                temp = {}
                for query_id, contexts in retrieval_metadata["full"].items():
                    temp[query_id] = []
                    for line in contexts:
                        docid = line["docid"]
                        content = docid2fulltext[docid]

                        to_append = {"id": docid, "contents": content}
                        temp[query_id].append(to_append)

                retrieval_metadata = temp



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
            if len(groundtruths_preds) == NUM_SAMPLE: break

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

    else:
        groundtruths = []
        for gt_answer in groundtruth_answers:
            gt_answer_text = gt_answer.get("text")
            gt_answer_text = gt_answer_text if isinstance(gt_answer_text, str) else "--__--".join(gt_answer_text)

            groundtruths.append(gt_answer_text)

            if len(groundtruths) == NUM_SAMPLE: break
        
        with open(outfile_gt, "w") as f:
            for gt in groundtruths:
                f.write(gt.replace("\n", " "))
                f.write("\n")

        
    assert os.path.exists(outfile_gt) and os.path.exists(outfile_pred)
    # print("====BERTSCORE====")
    # run_bertscore_evaluation(outfile_pred, outfile_gt)
    # print("====MINICHECK====")
    evaluation_metadata_file = os.path.join(outfolder, f"{dataset}__{retrieval_model}__rc{int(use_retrieval_contexts)}__chunk{use_chunk}.evaluation_metadata.txt")
    # run_minicheck_based_evaluation(outfile_pred, outfile_gt, minicheck_ckpt_path, evaluation_metadata_file)


    # re-init client
    init_client(
        openai_api_key, 
        openai_model_name = eval_openai_model_name,
        local = eval_local_llm_port is not None, 
        port = eval_local_llm_port, 
        model_name = eval_local_llm_model
    )
    run_llm_based_evaluation_keypoints(
        queries = [q.get("text") for q in queries][:NUM_SAMPLE],
        outfile_pred=outfile_pred,
        outfile_gt = outfile_gt,
        OPENAI_CLIENT = OPENAI_CLIENT,
        evaluation_metadata_file=evaluation_metadata_file
    )


if __name__ == "__main__":
    main()