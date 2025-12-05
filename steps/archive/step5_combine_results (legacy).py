# script to combine the results of the previous step into the final dataset
# the input include: output of steps 1, 2_1, 2_2, 3, 4

import os, hydra, random
import pandas as pd
from datetime import datetime
from omegaconf import DictConfig
from tqdm import tqdm
from typing import List
from steps.utils.generic import read_json_or_jsonl, write_to_jsonl, write_to_json, maybe_create_folder
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

    return pd.DataFrame(qrels_lines)


def sample_qa(queries, answers, num_samples, seed=42):
    assert len(queries) == len(answers)
    if seed is not None:
        random.seed(seed)
    indices = random.sample(range(len(queries)), min(len(queries), num_samples))
    sampled_queries = [queries[i] for i in indices]
    sampled_answers = [answers[i] for i in indices]
    for q, a in zip(sampled_queries, sampled_answers):
        assert q["_id"] == a["_id"], (q["_id"], a["_id"])
    return sampled_queries, sampled_answers


def wiki_topics_processing(topics: List[str]):
    # https://www.mediawiki.org/wiki/ORES/Articletopic

    res = set()
    for top in topics:
        res.add(top)

    return list(res)




def sanity_check(queries, answers, attributes, qrels):
    assert len(queries) == len(answers) == len(attributes)

    for i in range(len(queries)):
        assert queries[i]["_id"] == answers[i]["_id"] == attributes[i]["_id"]
        assert attributes[i]["num_keypoints"] >= attributes[i]["num_hops"]
        assert attributes[i]["num_keypoints"] == len(answers[i]["text"])

        evidence_attr = attributes[i]["evidence_attr"]
        langs = evidence_attr["langs"]
        published_dates = evidence_attr["published_dates"]

        assert len(langs) == attributes[i]["num_keypoints"] == len(published_dates), queries[i]["_id"]

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

    all_queries = []
    all_corpus = []
    all_answers = []
    all_qrels = {}
    all_attributes = []
    for ef_file_path, cuc_file_path, dff_file_path, fgf_file_path, qg_file_path in tqdm(zip(extracted_facts_files_full_path, 
                                                                                                            crawled_url_content_files_full_path, 
                                                                                                            decontextualized_facts_files_full_path,
                                                                                                            fact_groundedness_files_full_path,
                                                                                                            question_generation_files_full_path), total = len(files)):
        try:
            ef_data = read_json_or_jsonl(ef_file_path)
            cuc_data = read_json_or_jsonl(cuc_file_path)
            dff_data = read_json_or_jsonl(dff_file_path)
            fgf_data = read_json_or_jsonl(fgf_file_path)
            qg_data = read_json_or_jsonl(qg_file_path)
        except FileNotFoundError: continue

        raw_facts = ef_data.get("raw_facts")
        url_content_mapper = cuc_data.get("url_content_mapper")
        keypoints_mapper = dff_data.get("keypoints_mapper")
        groundedness_check = fgf_data.get("groundedness_check")
        fact_question_mapper = qg_data.get("fact_question_mapper")

        wiki_title = ef_data.get("title")
        create_timestamp = ef_data.get("create_timestamp")
        timestamp = ef_data.get("timestamp")
        topics = wiki_topics_processing(ef_data.get("topics"))

        queries = []
        corpus = []
        answers = []
        qrels = {}

        if any([item is None for item in [raw_facts, url_content_mapper, keypoints_mapper, fact_question_mapper]]): continue

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

        # queries and answers
        query_id_2_aux_fact_id = {}
        for fact_id in keypoints_mapper:
            if int(fact_id) not in fact_question_mapper: continue
            keypoints = keypoints_mapper[fact_id]
            fact_queries = fact_question_mapper[fact_id]

            for line in fact_queries:
                query_text = line["question"]
                num_hops = line["num_hops"]
                aux_fact_id = line["aux_fact_id"]
                aux_keypoints = keypoints_mapper[aux_fact_id] if aux_fact_id is not None else []

                # if aux_fact_id is not None and aux_fact_id not in fact_question_mapper: continue

                query_id_2_aux_fact_id[f"{wiki_title}--{fact_id}--{num_hops}"] = aux_fact_id

                _first_masked_node = [triplet["head_unmasked"] for triplet in line["masked_kg"] if triplet["head"] ==  "<Unknown> #1"]
                if not _first_masked_node: continue
                else: first_masked_node = _first_masked_node[0]
                
                query_id = f"{wiki_title}--{fact_id}--{num_hops}"

                if "<Unknown" in query_text: continue

                queries.append({"_id": query_id, "text": query_text})
                answer_keypoints = [kp for kp_index, kp in enumerate(keypoints) if kp_index in good_keypoints[fact_id]]
                if aux_fact_id is not None:
                    answer_keypoints += [kp for kp_index, kp in enumerate(aux_keypoints) if kp_index in good_keypoints[aux_fact_id]]

                answers.append({"_id": query_id, "text": answer_keypoints, "short": first_masked_node})


        # filter queries and answers based on if the answer contain any keypoints (if not then remove the id)
        good_qids = set([])
        for line in answers:
            query_id = line["_id"]
            keypoints = line["text"]
            if keypoints: good_qids.add(query_id)

        queries = [line for line in queries if line["_id"] in good_qids]
        answers = [line for line in answers if line["_id"] in good_qids]

        queries_ids = [line["_id"] for line in queries]
        print(queries_ids)

        ####################################################################

        # corpus
        for url in url_content_mapper:
            content = url_content_mapper.get(url).get("url_content")
            if url_content_mapper.get(url, {}).get("error") or not content: continue

            published_date = url_content_mapper.get(url).get("published_date")

            corpus.append({"_id": url, "title": "", "text": content, "published_date": published_date})
        ####################################################################

        # qrels
        query_id_2_keypoints = {}
        for fact_url_kp_id, check_label in groundedness_check.items():
            fact_id, url, kp_id = fact_url_kp_id.split("--__--")
            query_id = f"{wiki_title}--{fact_id}"

            # if int(fact_id) not in fact_question_mapper: continue

            if query_id not in query_id_2_keypoints: query_id_2_keypoints[query_id] = {}

            if check_label: 
                url_lang = url_content_mapper.get(url, {}).get("lang")
                published_date = url_content_mapper.get(url, {}).get("published_date")
                content = url_content_mapper.get(url).get("url_content")

                if kp_id not in query_id_2_keypoints[query_id]:
                    query_id_2_keypoints[query_id][kp_id] = []

                query_id_2_keypoints[query_id][kp_id].append({"url": url, "lang": url_lang, "published_date": published_date})

        for fact_url_kp_id, check_label in groundedness_check.items():
            fact_id, url, kp_id = fact_url_kp_id.split("--__--")
            query_id = f"{wiki_title}--{fact_id}"

            # if int(fact_id) not in fact_question_mapper: continue

            for num_hop in range(3):
                query_id_with_num_hop = f"{query_id}--{num_hop}"

                # if query_id_with_num_hop not in queries_ids: continue

                if query_id_with_num_hop not in qrels: qrels[query_id_with_num_hop] = {}
                qrels[query_id_with_num_hop][url] = 1 #int(check_label) if url not in qrels[query_id] else max(int(check_label), int(qrels[query_id][url]))

        for query_id_with_num_hop in queries_ids:
            aux_fact_id = query_id_2_aux_fact_id.get(query_id_with_num_hop)
            aux_query_id = f"{wiki_title}--{aux_fact_id}--1"

            if not aux_fact_id: continue
            for url in qrels[aux_query_id]:
                qrels[query_id_with_num_hop][url] = 1
        ####################################################################
        
        # attributes
        attributes = []
        for i in range(len(queries)):
            query_id = "--".join(queries[i]["_id"].split("--")[:-1]) # query_id without num_hop
            num_hops = int(queries[i]["_id"].split("--")[-1])
            
            num_keypoints = len(answers[i]["text"])
            aux_fact_id = query_id_2_aux_fact_id[queries[i]["_id"]]
            aux_query_id = f"{wiki_title}--{aux_fact_id}"

            evidence_langs = []
            evidence_published_dates = []
            for qid in [query_id, aux_query_id]:
                for kp_id in query_id_2_keypoints.get(qid, {}):
                    evidence_langs_kp_id = []
                    evidence_published_dates_kp_id = []
                    for evidence in query_id_2_keypoints[qid][kp_id]:
                        lang = evidence.get("lang")
                        published_date = evidence.get("published_date")

                        if lang and published_date: 
                            evidence_langs_kp_id.append(lang)
                            evidence_published_dates_kp_id.append(published_date)

                    evidence_langs.append(evidence_langs_kp_id)
                    evidence_published_dates.append(evidence_published_dates_kp_id)
            if wiki_title == "Beit Sahour tax strike":
                print(query_id, aux_query_id, evidence_langs)

            to_append = {
                "_id": queries[i]["_id"],
                "num_keypoints": num_keypoints,
                "num_hops": num_hops,
                "topics": topics,
                "wiki_create_timestamp": create_timestamp,
                "evidence_attr": {
                    "langs": evidence_langs,
                    "published_dates": evidence_published_dates
                }
            }
            attributes.append(to_append)
        ####################################################################


        all_queries.extend(queries)
        all_corpus.extend(corpus)
        all_answers.extend(answers)
        all_attributes.extend(attributes)
        all_qrels.update(qrels)

        
    sanity_check(queries, answers, attributes, all_qrels)
    
    write_to_jsonl(all_corpus, os.path.join(output_folder, "corpus.jsonl"))
    write_to_jsonl(all_queries, os.path.join(output_folder, "queries.jsonl"))
    write_to_jsonl(all_answers, os.path.join(output_folder, "answers.jsonl"))
    write_to_jsonl(all_attributes, os.path.join(output_folder, "attributes.jsonl"))

    df_qrels = process_qrels(all_qrels)
    maybe_create_folder(os.path.join(output_folder, "qrels"))
    df_qrels.to_csv(os.path.join(output_folder, "qrels", "test.tsv"), sep='\t', index=False)
        

if __name__ == "__main__":
    main()