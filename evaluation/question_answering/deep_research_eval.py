import os, hydra, json, openai, re
from omegaconf import DictConfig
from llm_apis import init_llm
from evaluation.question_answering.utils.qa_eval_metrics import run_llm_based_evaluation_keypoints
from evaluation.question_answering.utils.allowed_datasets import ALLOWED_DATASETS
from evaluation.question_answering.eval import data_relative_path, metadata_folder_name_creation

from evaluation.question_answering.deep_research.search_agent.openai_client import run_conversation_with_tools, SearchToolHandler
from evaluation.question_answering.deep_research.search_engine import init_ds_searcher

from tqdm import tqdm

REASONING_MODELS = ["gpt-5", "gpt-5-mini", "gpt-5-nano", "deepseek-r1", "gpt-oss-120b"] # reasoning models 


def generate_answer(query, 
                    qa_llm_model_name,
                    openai_client,
                    tool_handler,
                    wikidump_date,
                    max_iterations,
                    max_search_calls):
    
    def parse_answer(messages):
        try:
            text = messages[-1]["content"][0]["text"]
        except Exception as e:
            return "I don't know"
        
        pattern = r"Explanation:\s*(?P<explanation>.*?)\s*Exact Answer:\s*(?P<exact_answer>.*?)\s*Confidence:\s*(?P<confidence>.*)"
    
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            return {
                "explanation": match.group("explanation").strip(),
                "exact_answer": match.group("exact_answer").strip(),
                "confidence": match.group("confidence").strip()
            }
        
        return {}
        
        
    
    prompt = """You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tool provided. Please perform reasoning and use the tool step by step, in an interleaved manner. You may use the search tool at most {MaxSearchCalls} times.
The question, most of the time, will have a single answer. However, this is not guaranteed. In that case, it is your job to find all answers. You will be judged as correct if among your answers the groundtruth can be found.

The question is asked on: {QuestionDate}
Question: {Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences.}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}

Please follow these guidelines when formulating your Exact Answer:
1. If the question contains a false premise or assumption, Exact Answer should be "False premise question".
2. If you are uncertain or don’t know the answer, Exact Answer should be "I don’t know".
3. Note that the retrieved context can be in other languages than English. However, Exact Answer must be in English."""
    
    initial_request = {
        "model": qa_llm_model_name,
        "max_output_tokens": 2048,
        "input": [
            {"role": "user", "content": prompt.format(Question = query, 
                                                      QuestionDate = wikidump_date,
                                                      MaxSearchCalls = max_search_calls)}
        ],
        "tools": tool_handler.get_tool_definitions(),
        "truncation": "auto",
    }

    if qa_llm_model_name in REASONING_MODELS:
        initial_request["reasoning"] = {"effort": "medium"}
    
    messages, tool_usage, status = run_conversation_with_tools(
        client = openai_client,
        initial_request=initial_request,
        tool_handler=tool_handler,
        max_iterations=max_iterations,
        max_search_calls = max_search_calls,
        verbose = True
    )

    answer = parse_answer(messages)
    
    res = {
        "text":  answer["exact_answer"],
        "explanation": answer["explanation"],
        "confidence": answer["confidence"],
        "messages": messages,
        "raw": {},
        "tool_usage": tool_usage,
        "status": status
    }

    return res


@hydra.main(version_base=None, config_path="../../conf/evaluation", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig):
    retrieval_model = cfg.general.retrieval_model
    work_dir = cfg.general.work_dir
    dataset = cfg.general.dataset
    dataset_date = cfg.general.dataset_date
    wikidump_date = cfg.general.wikidump_date
    index_folder = cfg.general.index_folder

    outfolder = cfg.qa.outfolder
    use_retrieval_contexts = cfg.qa.use_retrieval_contexts
    eval_only = cfg.qa.eval_only
    use_chunk = cfg.qa.use_chunk
    max_iterations = cfg.qa.deep_research.max_iterations
    max_search_calls = cfg.qa.deep_research.max_search_calls
    qa_llm_model_name = cfg.qa.qa_llm_model_name
    eval_llm_model_name = cfg.qa.eval_llm_model_name
    eval_llm2_model_name = cfg.qa.eval_llm2_model_name


    dataset_name_2_relative_path = {
        dn: data_relative_path(dn, dataset_date) \
            for dn in ALLOWED_DATASETS
    }

    _metadata_folder = metadata_folder_name_creation(
        dataset = dataset, llm_model_name = qa_llm_model_name, retrieval_model = retrieval_model,
        use_retrieval_contexts = use_retrieval_contexts, use_chunk = use_chunk
    )
    _metadata_folder = f"deepres_{_metadata_folder}"

    os.makedirs(os.path.join(outfolder, _metadata_folder), exist_ok=True)
    eval_metadata_outfile = os.path.join(outfolder, _metadata_folder, "eval_metadata.json")
    eval_result_outfile = os.path.join(outfolder, _metadata_folder, "eval_result.json")

    print(f"Model predictions will be written to '{eval_metadata_outfile}'")
    print(f"Evaluation results will be written to '{eval_result_outfile}'")


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
        searcher = init_ds_searcher(model_name = retrieval_model, index_folder = index_folder)
        tool_hander = SearchToolHandler(searcher = searcher, include_get_document = False)
        openai_client = openai.OpenAI()

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

            try:
                answer = generate_answer(
                    query = query_text,
                    qa_llm_model_name = qa_llm_model_name,
                    openai_client = openai_client,
                    tool_handler = tool_hander,
                    wikidump_date = wikidump_date,
                    max_iterations = max_iterations,
                    max_search_calls = max_search_calls
                )
            except Exception as e:
                print(e)
                answer = {"text": "No answer generated", "raw": {}}

            eval_metadata["predictions"][query_id] = answer["text"]
            eval_metadata["groundtruths"][query_id] = gt_answer_text
            eval_metadata["queries"][query_id] = query_text
            eval_metadata["raw_response"][query_id] = answer

        with open(eval_metadata_outfile, "w") as f:
            json.dump(eval_metadata, f, indent = 4)

        
    assert os.path.exists(eval_metadata_outfile)

    # re-init client
    LLM = init_llm(eval_llm_model_name)
    LLM2 = init_llm(eval_llm2_model_name) if eval_llm2_model_name else None
    run_llm_based_evaluation_keypoints(
        eval_metadata_outfile = eval_metadata_outfile,
        groundtruth_answers = groundtruth_answers,
        LLM = LLM,
        LLM2 = LLM2,
        eval_result_outfile = eval_result_outfile
    )

if __name__ == "__main__":
    main()