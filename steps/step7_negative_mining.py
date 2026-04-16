


import os, hydra, shutil, requests, asyncio, math
import pandas as pd
from omegaconf import DictConfig
from copy import deepcopy
from tqdm import tqdm
from urllib.parse import urlparse
from typing import List, Dict, Optional, Any

from llm_apis import init_llm, BaseLLMAPI
from steps.utils.generic import read_json_or_jsonl, write_to_jsonl, maybe_create_folder
from steps.utils.bad_domains import BAD_DOMAINS
from steps.utils.hard_neg import brave_api_search, create_ai_content_farm_site
from steps.step2_1_url_content_crawling import check_urls_in_parallel, get_content_from_resp


def get_stale_corpus(single_hop_queries: List[str],
                        num_hard_negs: int, start_from: str):
    all_urls = set([])
    url2date = {}
    for query in tqdm(single_hop_queries, desc = "Brave search"):
        search_results = brave_api_search(query = query, 
                                          num_search_results = num_hard_negs, 
                                          evidence_start_from = start_from)

        for sr in search_results:
            url = sr.get("url")
            published_date = sr.get("page_age")

            domain = urlparse(url).netloc.lower()
            social_domains = ["facebook.com", "twitter.com", "x.com", "github.com", "google.com", "amazon.com", "youtube.com"]
            if domain in BAD_DOMAINS or any(s in domain for s in social_domains): continue

            all_urls.add(url)
            url2date[url] = published_date

    all_urls = [url for url in all_urls if url2date.get(url) is not None]
    responses = asyncio.run(check_urls_in_parallel(all_urls, url2date, None, max_concurrents=16))

    augmented_corpus = []
    for resp, url in zip(responses, all_urls):
        obj = get_content_from_resp(resp, url)

        content = obj.get("url_content")
        content = content if content else obj.get("error")
        if obj.get("accessible") is True:
            published_date = obj.get("published_date")
            lang = obj.get("lang")

            augmented_corpus.append({"_id": url, "title": "", "text": content, "published_date": published_date, "lang": lang})

    return augmented_corpus


# def get_imposter_corpus(answers: List[str], answers_types: List[str], wikidump_date: str, gold_documents: List[List[Dict[str, Any]]]):
#     augmented_corpus = []

#     for answer, answer_type, documents in tqdm(zip(answers, answers_types, gold_documents), desc = "Creating imposter sites"):
#         to_extend = create_imposter_site(answer = answer, answer_type = answer_type, wikidump_date = wikidump_date, documents = documents)
#         if to_extend:
#             augmented_corpus.extend(to_extend)

#     return augmented_corpus


def get_ai_content_farm_corpus(questions: List[str], LLM: BaseLLMAPI):
    augmented_corpus = []
    for question in tqdm(questions, desc = "Generating AI content farm news"):
        to_append = create_ai_content_farm_site(question = question, LLM = LLM)
        augmented_corpus.append(to_append)
    
    return augmented_corpus


@hydra.main(version_base=None, config_path="../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig):
    sampled_benchmark_folder = cfg.step6.output_folder
    output_folder = cfg.step7.output_folder
    num_hard_negs = cfg.step7.num_hard_negs
    start_from = cfg.general.start_from
    wikidump_date = cfg.general.wikidump_date
    llm_model_name = cfg.general.llm_model_name

    maybe_create_folder(output_folder)

    original_corpus_file = os.path.join(sampled_benchmark_folder, "corpus.jsonl")
    augmented_corpus_file = os.path.join(output_folder, "corpus.jsonl")

    original_corpus = read_json_or_jsonl(original_corpus_file)
    queries = read_json_or_jsonl(os.path.join(sampled_benchmark_folder, "queries.jsonl"))
    answers = read_json_or_jsonl(os.path.join(sampled_benchmark_folder, "answers.jsonl"))

    # read qrels
    df_qrels = pd.read_csv(os.path.join(sampled_benchmark_folder, "qrels", "test.tsv"), sep = '\t')
    qrels = {}
    for line in df_qrels.to_dict("records"):
        query_id = line["query-id"]
        doc_id = line["corpus-id"]
        
        if query_id not in qrels: qrels[query_id] = {}
        qrels[query_id][doc_id] = 1




    # get stale documents
    single_hop_queries = set([])
    for line in queries:
        component_single_hop_queries = line["component_single_hop_questions"]
        single_hop_queries.update(component_single_hop_queries)
    
    single_hop_queries = list(single_hop_queries)

    stale_corpus = get_stale_corpus(
        single_hop_queries = single_hop_queries,
        num_hard_negs = num_hard_negs,
        start_from = start_from
    )


    # # create imposter documents
    # answers_str = [item["short"] for item in answers]
    # answers_types = [item["type"] for item in answers]

    # id2doc = {line["_id"]: line for line in original_corpus}
    # gold_documents = []
    # for ans in answers:
    #     _id = ans["_id"]
    #     gold_doc_ids = list(qrels[_id].keys())
    #     to_append = [id2doc[gdi] for gdi in gold_doc_ids]
    #     gold_documents.append(to_append)


    # imposter_corpus = get_imposter_corpus(
    #     answers = answers_str,
    #     answers_types = answers_types,
    #     wikidump_date = wikidump_date,
    #     gold_documents = gold_documents
    # )

    LLM = init_llm(llm_model_name)
    content_farm_corpus = get_ai_content_farm_corpus(questions = single_hop_queries, LLM = LLM)

    corpus = [] 
    for line in original_corpus:
        to_append = deepcopy(line)
        to_append["note"] = "original"
        corpus.append(to_append)

    for line in stale_corpus:
        to_append = deepcopy(line)
        to_append["note"] = "stale"
        corpus.append(to_append)
    
    # for line in imposter_corpus:
    #     to_append = deepcopy(line)
    #     to_append["note"] = "imposter"
    #     corpus.append(to_append)

    for line in content_farm_corpus:
        to_append = deepcopy(line)
        to_append["note"] = "AI content"
        corpus.append(to_append)


    write_to_jsonl(corpus, augmented_corpus_file)

    for _type in ["queries", "answers", "attributes"]:
        src_file = os.path.join(sampled_benchmark_folder, f"{_type}.jsonl")
        dst_file = os.path.join(output_folder, f"{_type}.jsonl")
        shutil.copy(src_file, dst_file)

    sampled_bench_qrels_file = os.path.join(sampled_benchmark_folder, "qrels")
    sampled_qrels_file = os.path.join(output_folder, "qrels")
    shutil.copytree(sampled_bench_qrels_file, sampled_qrels_file)



if __name__ == "__main__":
    main()