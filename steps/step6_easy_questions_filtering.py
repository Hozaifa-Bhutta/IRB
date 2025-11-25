# this step is for making the dataset more difficult. In this step, we will be filtering questions that
# 1) can be answered (by the model with which we generate the dataset) without any retrieval
# 2) multi-hop questions that can be answered with the retrieval context of the first hop (these questions are not really multi-hop)

import os, hydra, itertools, random, shutil
import pandas as pd
from omegaconf import DictConfig
from steps.utils.generic import read_json_or_jsonl, write_to_jsonl, write_to_json, maybe_create_folder
from typing import List, Dict
from collections import Counter
from tqdm import tqdm
from evaluation.question_answering.eval import generate_answer
from evaluation.question_answering.utils.prompts import LLM_BASED_EVALUATION_PROMPT
from llm_apis import BaseLLMAPI, init_llm


def read_qrels(path):
    df = pd.read_csv(path, sep='\t')
    qrels = {}
    for line in df.to_dict(orient = "records"):
        query_id = line["query-id"]
        doc_id = line["corpus-id"]
        score = line["score"]

        if query_id not in qrels: qrels[query_id] = {}
        qrels[query_id][doc_id] = score

    return qrels


def evaluate_answers(queries, generated_answers, groundtruth_answers, LLM: BaseLLMAPI):
    assert len(generated_answers) == len(groundtruth_answers) == len(queries)

    eval_result = {}
    for i in tqdm(range(len(generated_answers)), desc = "Evaluating answers"):
        if generated_answers[i] is None: 
            continue
        
        query_id = queries[i]["_id"]
        query = queries[i]["text"]
        prediction = generated_answers[i]["text"]
        keypoints = groundtruth_answers[i]["text"] # molecular facts
        gt_short = groundtruth_answers[i]["short"] # gold answer
        print(query, gt_short, prediction, keypoints)

        eval_user_prompt = LLM_BASED_EVALUATION_PROMPT["user"][:].replace("[QUESTION]", query)\
                                                .replace("[SHORT]", gt_short)\
                                                .replace("[GENERATED_ANSWER]", prediction)\
                                                .replace("[ADD_KEYPOINTS_HERE]", "\n".join(keypoints))
        json_result = []
        try:
            for llm in [LLM]:
                if llm is None: continue
                result = llm.generate(
                    system_prompt = LLM_BASED_EVALUATION_PROMPT["system"],
                    user_prompt = eval_user_prompt,
                    max_output_tokens = 16,
                    temperature = 0.2
                )
                if "A" in result: json_result.append("CORRECT")
                elif "B" in result: json_result.append("INCORRECT")
                elif "C" in result: json_result.append("NOT_ATTEMPTED")
                else: json_result.append("INCORRECT")
        except Exception as e:
            print(e)
            continue

            
        
        result_counter = Counter(json_result)
        total = sum(result_counter.values())

        eval_result[query_id] = {k: v / total for k, v in result_counter.items()}

    return eval_result



def answer_queries_without_retrieval(queries, wikidump_date, LLM):
    all_answers = []
    for line in tqdm(queries, desc = "Answering queries without retrieval"):
        query_id = line["_id"]
        if int(query_id.split("--")[-1]) > 1: 
            all_answers.append(None)
            continue

        query_text = line["text"]
        try:
            answer = generate_answer(
                query = query_text, 
                contexts = [],
                wikidump_date = wikidump_date, 
                use_retrieval_contexts = False,
                LLM = LLM
            )
        except Exception as e:
            print(e)
            answer = None
        all_answers.append(answer)

    return all_answers

def answer_multihop_queries_with_firsthop_evidence(queries, wikidump_date, corpus, qrels, LLM):
    doc_id_2_document_content = {line["_id"]: line["text"] for line in corpus}
    all_answers = []
    for line in tqdm(queries, desc = "Answer queries with first-hop evidence"):
        query_id = line["_id"]
        if int(query_id.split("--")[-1]) == 1: 
            all_answers.append(None)
            continue

        query_text = line["text"]

        query_id_single_hop = query_id[:-3] + "--1"
        single_hop_supporting_documents = [{"contents": doc_id_2_document_content[doc_id]} for doc_id in qrels.get(query_id_single_hop, [])]
        if not single_hop_supporting_documents: 
            all_answers.append(None)
            continue

        try:
            answer = generate_answer(
                query = query_text,
                contexts = single_hop_supporting_documents,
                wikidump_date = wikidump_date,
                use_retrieval_contexts = True,
                max_num_contexts = len(single_hop_supporting_documents),
                LLM = LLM
            )
        except Exception as e:
            print(e)
            answer = None
        all_answers.append(answer)

    return all_answers


@hydra.main(version_base=None, config_path="../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig):
    full_benchmark_folder = cfg.step5.output_folder
    output_folder = cfg.step6.output_folder
    wikidump_date = cfg.general.wikidump_date
    llm_model_name = cfg.general.llm_model_name

    LLM = init_llm(llm_model_name)
    LLM2 = init_llm("gemini-2.5-flash-for-eval")


    output_queries_file = os.path.join(output_folder, "queries.jsonl")
    output_answers_file = os.path.join(output_folder, "answers.jsonl")
    output_attributes_file = os.path.join(output_folder, "attributes.jsonl")


    full_bench_queries = read_json_or_jsonl(os.path.join(full_benchmark_folder, "queries.jsonl"))[:]
    full_bench_answers = read_json_or_jsonl(os.path.join(full_benchmark_folder, "answers.jsonl"))[:]
    full_bench_attributes = read_json_or_jsonl(os.path.join(full_benchmark_folder, "attributes.jsonl"))[:]
    full_bench_corpus = read_json_or_jsonl(os.path.join(full_benchmark_folder, "corpus.jsonl"))
    qrels = read_qrels(os.path.join(full_benchmark_folder, "qrels", "test.tsv"))


    # # generate answers for single-hop questions without supporting documents to see if the model correctly answer it
    # answers_without_retrieval = answer_queries_without_retrieval(
    #     queries = full_bench_queries,
    #     wikidump_date = wikidump_date,
    #     LLM = LLM
    # )
    # evaluation_answers_without_retrieval = evaluate_answers(
    #     queries = full_bench_queries,
    #     generated_answers = answers_without_retrieval,
    #     groundtruth_answers = full_bench_answers,
    #     LLM = LLM2
    # )
    # # get the ids of the queries for which the model answer correctly without supporting documents
    # easy_query_ids = {query_id for query_id in evaluation_answers_without_retrieval if "CORRECT" in evaluation_answers_without_retrieval[query_id]}
    easy_query_ids = set()

    # next, generate answers for multi-hop question using only the evidence of the first hop
    answers_with_first_hop_evidence = answer_multihop_queries_with_firsthop_evidence(
        queries = full_bench_queries,
        wikidump_date = wikidump_date,
        corpus = full_bench_corpus, 
        qrels = qrels,
        LLM = LLM
    )
    evaluation_answers_with_first_hop_evidence = evaluate_answers(
        queries = full_bench_queries,
        generated_answers = answers_with_first_hop_evidence,
        groundtruth_answers = full_bench_answers,
        LLM = LLM2
    )
    easy_query_ids.update({query_id for query_id in evaluation_answers_with_first_hop_evidence if "CORRECT" in evaluation_answers_with_first_hop_evidence[query_id]})

    output_queries = [line for line in full_bench_queries if line["_id"] not in easy_query_ids]
    output_answers = [line for line in full_bench_answers if line["_id"] not in easy_query_ids]
    output_attributes = [line for line in full_bench_attributes if line["_id"] not in easy_query_ids]

    write_to_jsonl(output_queries, output_queries_file)
    write_to_jsonl(output_answers, output_answers_file)
    write_to_jsonl(output_attributes, output_attributes_file)

    full_bench_corpus_file = os.path.join(full_benchmark_folder, "corpus.jsonl")
    output_corpus_file = os.path.join(output_folder, "corpus.jsonl")
    shutil.copy(full_bench_corpus_file, output_corpus_file)

    full_bench_qrels_file = os.path.join(full_benchmark_folder, "qrels")
    output_qrels_file = os.path.join(output_folder, "qrels")
    shutil.copytree(full_bench_qrels_file, output_qrels_file)


if __name__ == "__main__":
    main()