import os, hydra, json, sys, time, openai
from omegaconf import DictConfig
from llm_apis import init_llm
from evaluation.question_answering.utils.qa_prompt import QA_SYSTEM_PROMPT, QA_USER_PROMPT, QA_SYSTEM_PROMPT_WITHOUT_CONTEXT, QA_USER_PROMPT_WITHOUT_CONTEXT
from evaluation.question_answering.utils.qa_eval_metrics import run_llm_based_evaluation_keypoints
from evaluation.question_answering.utils.allowed_datasets import ALLOWED_DATASETS
from tqdm import tqdm

# maximum number of samples to run evaluation
NUM_SAMPLE = 1000 #1000**2


def create_enumerated_list(contexts):
    if not contexts: return ""
    texts = [cont["contents"] for cont in contexts]
    published_dates = [cont["published_date"] for cont in contexts]
    urls = [cont["id"] for cont in contexts]

    retrieved_contexts = []
    for url, text, published_date in zip(urls, texts, published_dates):
        truncated_text = " ".join(text.split()[:5000])
        retrieved_contexts.append(
            f"```\nURL: {url}\nPublished Date: {published_date}\n{truncated_text}\n```"
        )
    return '\n\n\n'.join(f"Context {i+1}:\n{text}" for i, text in enumerate(retrieved_contexts))

def generate_answer(query, 
                    contexts, 
                    wikidump_date,
                    use_retrieval_contexts = True,
                    max_num_contexts = 20,
                    LLM = None):
    # time.sleep(0.5)
    qa_user_prompt = QA_USER_PROMPT if use_retrieval_contexts else QA_USER_PROMPT_WITHOUT_CONTEXT
    qa_system_prompt = QA_SYSTEM_PROMPT if use_retrieval_contexts else QA_SYSTEM_PROMPT_WITHOUT_CONTEXT

    str_context = create_enumerated_list(contexts[:max_num_contexts]) if use_retrieval_contexts else ""

    user_prompt = qa_user_prompt[:].replace("[ADD CONTEXT HERE]", str_context)\
        .replace("[ADD QUESTION HERE]", query)\
        .replace("[ADD_QUESTION_DATE]", wikidump_date)

    # print(user_prompt)

    try:
        response = LLM.client.responses.create(
            model = LLM.model_name,
            instructions = qa_system_prompt,
            input = user_prompt,
            max_output_tokens = 2048,
            tool_choice = "none"
        )

        result = {
            "text": response.output_text.strip(),
            "raw": response.model_dump()
        }

        return result
    except openai.RateLimitError as e:
        return "Could not answer this question due to an error"
    

def data_relative_path(dataset_name, dataset_date = None):
    if dataset_date:
        return os.path.join("benchmarks", dataset_date, dataset_name)
    else:
        raise NotImplementedError
    

def metadata_folder_name_creation(dataset: str, llm_model_name: str, retrieval_model: str, use_retrieval_contexts: bool, use_chunk: bool):
    temp = [dataset, llm_model_name] + \
            ([retrieval_model, f"chunk{use_chunk}"] if use_retrieval_contexts else [])
    print(temp)
    return "__".join(temp)

@hydra.main(version_base=None, config_path="../../conf/evaluation", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig):
    retrieval_model = cfg.general.retrieval_model
    retrieval_metadata_folder = cfg.general.retrieval_metadata_path
    work_dir = cfg.general.work_dir
    dataset = cfg.general.dataset
    dataset_date = cfg.general.dataset_date
    wikidump_date = cfg.general.wikidump_date

    max_num_contexts = cfg.qa.max_num_contexts
    outfolder = cfg.qa.outfolder
    use_retrieval_contexts = cfg.qa.use_retrieval_contexts
    eval_only = cfg.qa.eval_only
    use_chunk = cfg.qa.use_chunk
    qa_llm_model_name = cfg.qa.qa_llm_model_name
    eval_llm_model_name = cfg.qa.eval_llm_model_name

    dataset_name_2_relative_path = {
        dn: data_relative_path(dn, dataset_date) \
            for dn in ALLOWED_DATASETS
    }

    _metadata_folder = metadata_folder_name_creation(
        dataset = dataset, llm_model_name = qa_llm_model_name, retrieval_model = retrieval_model,
        use_retrieval_contexts = use_retrieval_contexts, use_chunk = use_chunk
    )
    os.makedirs(os.path.join(outfolder, _metadata_folder), exist_ok=True)
    eval_metadata_outfile = os.path.join(outfolder, _metadata_folder, "eval_metadata.json")
    eval_result_outfile = os.path.join(outfolder, _metadata_folder, "eval_result.json")

    print(f"Model predictions will be written to '{eval_metadata_outfile}'")
    print(f"Evaluation results will be written to '{eval_result_outfile}'")

    retrieval_metadata_path = os.path.join(retrieval_metadata_folder, f"{dataset}__{retrieval_model}.json")

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
        docid2info = {}
        for line in f:
            jline = json.loads(line)
            docid = jline["_id"]
            text = jline["text"]
            published_date = jline["published_date"]
            docid2info[docid] = {"text": text, "published_date": published_date}

    
    with open(groundtruth_answers_path) as f:
        groundtruth_answers = [json.loads(line) for line in f]

    print(f"""QUERIES: {len(queries)}\nANSWERS: {len(groundtruth_answers)}\nCORPUS: {len(docid2info)}""")

    if eval_only is False:
        LLM = init_llm(qa_llm_model_name)


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
                        content = docid2info[docid]["text"]
                        published_date = docid2info[docid]["published_date"]

                        to_append = {"id": docid, "contents": content, "published_date": published_date}
                        temp[query_id].append(to_append)

                retrieval_metadata = temp

            queries = [line for line in queries if line["_id"] in retrieval_metadata]
            groundtruth_answers = [line for line in groundtruth_answers if line["_id"] in retrieval_metadata]

        else: retrieval_metadata = {}

        eval_metadata = {
            "config": {
                "LLM_model_name": qa_llm_model_name,
                "dataset": dataset,
                "retrieval_model": retrieval_model if use_retrieval_contexts else "None",
                "use_chunk": use_chunk
            },
            "predictions": {},
            "groundtruths": {},
            "queries": {},
            "raw_response": {}
        }
        for query, gt_answer in tqdm(zip(queries, groundtruth_answers), desc = "Generating answers", total = len(queries)):
            query_text = query.get("text")
            query_id = query.get("_id")

            assert gt_answer.get("_id") == query_id

            gt_answer_text = gt_answer.get("text")
            gt_answer_text = gt_answer_text if isinstance(gt_answer_text, str) else "--__--".join(gt_answer_text)

            contexts = retrieval_metadata.get(query_id)

            try:
                answer = generate_answer(
                    query = query_text,
                    contexts = contexts,
                    wikidump_date = wikidump_date,
                    use_retrieval_contexts = use_retrieval_contexts,
                    max_num_contexts = max_num_contexts,
                    LLM = LLM
                )
            except Exception as e:
                print(e)
                answer = {"text": "No answer generated", "raw": {}}

            to_append = [gt_answer_text, answer]

            eval_metadata["predictions"][query_id] = answer["text"]
            eval_metadata["groundtruths"][query_id] = gt_answer_text
            eval_metadata["queries"][query_id] = query_text
            eval_metadata["raw_response"][query_id] = answer["raw"]

            if len(eval_metadata["groundtruths"]) == NUM_SAMPLE: break

        with open(eval_metadata_outfile, "w") as f:
            json.dump(eval_metadata, f, indent = 4)

        
    assert os.path.exists(eval_metadata_outfile)

    # re-init client
    LLM = init_llm(eval_llm_model_name)
    LLM2 = init_llm("gemini-2.5-flash-for-eval")
    run_llm_based_evaluation_keypoints(
        eval_metadata_outfile = eval_metadata_outfile,
        groundtruth_answers = groundtruth_answers,
        LLM = LLM,
        LLM2 = LLM2,
        eval_result_outfile = eval_result_outfile
    )


if __name__ == "__main__":
    main()