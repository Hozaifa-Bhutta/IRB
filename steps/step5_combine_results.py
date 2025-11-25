# script to combine the results of the previous step into the final dataset
# the input include: output of steps 1, 2_1, 2_2, 3, 4

import os, hydra, random
import pandas as pd
from rapidfuzz import fuzz
from copy import deepcopy
from datetime import datetime
from omegaconf import DictConfig
from tqdm import tqdm
from typing import List
from steps.utils.generic import read_json_or_jsonl, write_to_jsonl, write_to_json, maybe_create_folder, SIMPLE_TEXT_SPLITTER
from steps.utils.token_counting import token_count_tiktoken

def process_qrels(qrels):
    qrels_lines = []
    for qid in qrels:
        for pid in qrels[qid]:
            to_append = {
                "query-id": qid,
                "corpus-id": pid,
                "score": qrels[qid][pid]
            }
            qrels_lines.append(to_append)

    qrels_lines = list(sorted(qrels_lines, key = lambda x: x["query-id"]))
    return pd.DataFrame(qrels_lines)


def wiki_topics_processing(topics: List[str]):
    # https://www.mediawiki.org/wiki/ORES/Articletopic

    res = set()
    for top in topics:
        res.add(top)

    return list(res)

def get_corpus(crawled_url_content_files_full_path: List[str]):
    corpus = []
    for cuc_file_path in crawled_url_content_files_full_path:
        try:
            cuc_data = read_json_or_jsonl(cuc_file_path)
        except Exception: continue

        url_content_mapper = cuc_data.get("url_content_mapper")
        if not url_content_mapper: continue

        for url in url_content_mapper:
            content = url_content_mapper.get(url).get("url_content")
            if url_content_mapper.get(url, {}).get("error") or not content: continue

            published_date = url_content_mapper.get(url).get("published_date")
            lang = url_content_mapper.get(url).get("lang")

            corpus.append({"_id": url, "title": "", "text": content, "published_date": published_date, "lang": lang})

    return corpus

def get_qrels(fact_groundedness_files_full_path: List[str],
              question_generation_files_full_path: List[str]):
    qrels = {}

    for fgf_file_path, qg_file_path in zip(fact_groundedness_files_full_path, question_generation_files_full_path):
        try:
            fgf_data = read_json_or_jsonl(fgf_file_path)
        except Exception: continue

        wiki_title = fgf_data.get("title")
        groundedness_check = fgf_data.get("groundedness_check")
        if not groundedness_check: continue

        # qrels for single-hop questions
        for fact_url_kp_id, check_label in groundedness_check.items():
            fact_id, url, kp_id = fact_url_kp_id.split("--__--")

            if not check_label: continue

            query_id = f"{wiki_title}--{fact_id}--1"

            if query_id not in qrels: qrels[query_id] = {}
            qrels[query_id][url] = 1

        # qrels for multi-hop questions require information about aux_fact_id
        try:
            qg_data = read_json_or_jsonl(qg_file_path)
        except Exception: continue

        fact_question_mapper = qg_data.get("fact_question_mapper")
        for fact_id, fact_queries in fact_question_mapper.items():
            for fact_query in fact_queries:
                aux_fact_id = fact_query["aux_fact_id"]
                num_hops = fact_query["num_hops"]
                if aux_fact_id is None or num_hops == 1: continue

                query_id = f"{wiki_title}--{fact_id}--{num_hops}"
                query_id_1hop = f"{wiki_title}--{fact_id}--1"
                aux_query_id = f"{wiki_title}--{aux_fact_id}--1"

                if query_id not in qrels: qrels[query_id] = deepcopy(qrels[query_id_1hop])
                qrels[query_id].update(deepcopy(qrels[aux_query_id]))

    return qrels



def get_queries_and_answers(question_generation_files_full_path: List[str],
                        decontextualized_facts_files_full_path: List[str],
                        fact_groundedness_files_full_path: List[str]):
    
    answers, queries = [], []
    for qg_file_path, dff_file_path, fgf_file_path in zip(question_generation_files_full_path,
                                                          decontextualized_facts_files_full_path, 
                                                          fact_groundedness_files_full_path):
        
        try:
            qg_data = read_json_or_jsonl(qg_file_path)
            dff_data = read_json_or_jsonl(dff_file_path)
            fgf_data = read_json_or_jsonl(fgf_file_path)
        except Exception: continue

        keypoints_mapper = dff_data.get("keypoints_mapper")
        groundedness_check = fgf_data.get("groundedness_check")
        fact_question_mapper = qg_data.get("fact_question_mapper")
        wiki_title = dff_data.get("title")

        if not any([keypoints_mapper, groundedness_check, fact_question_mapper]): continue

        keypoints_mapper = {int(k): v for k,v in keypoints_mapper.items()}
        fact_question_mapper = {int(k): v for k,v in fact_question_mapper.items()}

        good_keypoints = {}
        for fact_url_kp_id, check_label in groundedness_check.items():
            if not check_label: continue
            fact_id, url, kp_id = fact_url_kp_id.split("--__--")
            fact_id = int(fact_id)
            kp_id = int(kp_id)

            if fact_id not in good_keypoints: good_keypoints[fact_id] = set()
            good_keypoints[fact_id].add(kp_id)

        for fact_id, fact_queries in fact_question_mapper.items():
            keypoints = keypoints_mapper[fact_id]
            for fact_query in fact_queries:
                query_text = fact_query["question"]
                num_hops = fact_query["num_hops"]
                aux_fact_id = fact_query["aux_fact_id"]
                aux_keypoints = keypoints_mapper[aux_fact_id] if aux_fact_id is not None else []

                _gold_answer = [triplet["head_unmasked"] for triplet in fact_query["masked_kg"] if triplet["head"] ==  "<Unknown> #1"]

                if not _gold_answer: continue
                else: gold_answer = _gold_answer[0]
                
                query_id = f"{wiki_title}--{fact_id}--{num_hops}"

                if "<Unknown" in query_text \
                    or fuzz.partial_ratio(SIMPLE_TEXT_SPLITTER(gold_answer), SIMPLE_TEXT_SPLITTER(query_text)) >= 50: continue

                queries.append({"_id": query_id, "text": query_text})
                answer_keypoints = [kp for kp_index, kp in enumerate(keypoints) if kp_index in good_keypoints[fact_id]]
                if aux_fact_id is not None:
                    answer_keypoints += [kp for kp_index, kp in enumerate(aux_keypoints) if kp_index in good_keypoints[aux_fact_id]]

                answers.append({"_id": query_id, "text": answer_keypoints, "short": gold_answer})

    return queries, answers


def get_attributes(queries, answers, corpus, qrels,
                   extracted_facts_files_full_path: List[str]):
    wikititle2info = {}
    for eff_file_path in extracted_facts_files_full_path:
        try:
            eff_data = read_json_or_jsonl(eff_file_path)
        except Exception:
            continue

        if not eff_data.get("raw_facts"): continue

        wiki_title = eff_data.get("title")
        topics = wiki_topics_processing(eff_data.get("topics"))
        create_timestamp = eff_data.get("create_timestamp")
        wikititle2info[wiki_title] = {
            "topics": topics,
            "create_timestamp": create_timestamp
        }


    url2index = {line["_id"]: i for i, line in enumerate(corpus)}

    attributes = []
    for i in range(len(queries)):
        query_id = queries[i]["_id"]
        num_hops = int(queries[i]["_id"].split("--")[-1])

        query_wiki_title = query_id.split("--")[0]
        
        num_keypoints = len(answers[i]["text"])

        supporting_evidence_urls = list(qrels[query_id].keys())
        supporting_evidence = [corpus[url2index[url]] for url in supporting_evidence_urls]

        evidence_langs = [ev["lang"] for ev in supporting_evidence]
        evidence_published_dates = [ev["published_date"] for ev in supporting_evidence]

        to_append = {
                "_id": queries[i]["_id"],
                "num_keypoints": num_keypoints,
                "num_hops": num_hops,
                "topics": wikititle2info[query_wiki_title]["topics"],
                "wiki_create_timestamp": wikititle2info[query_wiki_title]["create_timestamp"],
                "evidence_attr": {
                    "langs": evidence_langs,
                    "published_dates": evidence_published_dates
                }
            }
        attributes.append(to_append)

    return attributes


def sanity_check(queries, answers, attributes, qrels):
    assert len(queries) == len(answers) == len(attributes)

    for i in range(len(queries)):
        assert queries[i]["_id"] == answers[i]["_id"] == attributes[i]["_id"]
        assert attributes[i]["num_keypoints"] >= attributes[i]["num_hops"]
        assert attributes[i]["num_keypoints"] == len(answers[i]["text"])

        evidence_attr = attributes[i]["evidence_attr"]
        langs = evidence_attr["langs"]
        published_dates = evidence_attr["published_dates"]

        # assert len(langs) == attributes[i]["num_keypoints"] == len(published_dates), queries[i]["_id"]

    assert all([line["_id"] in qrels for line in queries])




@hydra.main(version_base=None, config_path="../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig):
    extracted_facts_folder = cfg.step1.output_folder
    crawled_url_content_folder = cfg.step2_1.output_folder
    decontextualized_facts_folder = cfg.step2_2.output_folder
    fact_groundedness_folder = cfg.step3.output_folder
    question_generation_folder = cfg.step4.output_folder
    output_folder = cfg.step5.output_folder

    files = os.listdir(extracted_facts_folder)
    files = [file for file in files if file.endswith('.json')]
    extracted_facts_files_full_path = [os.path.join(extracted_facts_folder, file) for file in files]
    crawled_url_content_files_full_path = [os.path.join(crawled_url_content_folder, file) for file in files]
    decontextualized_facts_files_full_path = [os.path.join(decontextualized_facts_folder, file) for file in files]
    fact_groundedness_files_full_path = [os.path.join(fact_groundedness_folder, file) for file in files]
    question_generation_files_full_path = [os.path.join(question_generation_folder, file) for file in files]

    corpus = get_corpus(
        crawled_url_content_files_full_path = crawled_url_content_files_full_path
    )

    qrels = get_qrels(
        fact_groundedness_files_full_path = fact_groundedness_files_full_path,
        question_generation_files_full_path = question_generation_files_full_path
    )

    queries, answers = get_queries_and_answers(
        question_generation_files_full_path = question_generation_files_full_path,
        decontextualized_facts_files_full_path = decontextualized_facts_files_full_path,
        fact_groundedness_files_full_path = fact_groundedness_files_full_path
    )

    attributes = get_attributes(
        queries = queries,
        answers = answers,
        corpus = corpus,
        qrels = qrels,
        extracted_facts_files_full_path = extracted_facts_files_full_path
    )

        
    sanity_check(queries, answers, attributes, qrels)
    
    write_to_jsonl(corpus, os.path.join(output_folder, "corpus.jsonl"))
    write_to_jsonl(queries, os.path.join(output_folder, "queries.jsonl"))
    write_to_jsonl(answers, os.path.join(output_folder, "answers.jsonl"))
    write_to_jsonl(attributes, os.path.join(output_folder, "attributes.jsonl"))

    df_qrels = process_qrels(qrels)
    maybe_create_folder(os.path.join(output_folder, "qrels"))
    df_qrels.to_csv(os.path.join(output_folder, "qrels", "test.tsv"), sep='\t', index=False)
        

if __name__ == "__main__":
    main()