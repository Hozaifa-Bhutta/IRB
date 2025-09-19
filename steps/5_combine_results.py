# script to combine the results of the previous step into the final dataset
# the input include: output of steps 1, 2_1, 2_2, 3, 4

import os, hydra, random
import pandas as pd
from datetime import datetime
from omegaconf import DictConfig
from tqdm import tqdm
from utils.generic import read_json_or_jsonl, write_to_jsonl, write_to_json, maybe_create_folder



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


@hydra.main(version_base=None, config_path="../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig):
    extracted_facts_folder = cfg.step1.output_folder
    crawled_url_content_folder = cfg.step2_1.output_folder
    decontextualized_facts_folder = cfg.step2_2.output_folder
    fact_groundedness_folder = cfg.step3.output_folder
    question_generation_folder = cfg.step4.output_folder
    output_folder = cfg.step5.output_folder

    utilize_fact_groundedness_check = cfg.general.utilize_fact_groundedness_check

    files = os.listdir(extracted_facts_folder)
    files = [file for file in files if file.endswith('.json')]
    extracted_facts_files_full_path = [os.path.join(extracted_facts_folder, file) for file in files]
    crawled_url_content_files_full_path = [os.path.join(crawled_url_content_folder, file) for file in files]
    decontextualized_facts_files_full_path = [os.path.join(decontextualized_facts_folder, file) for file in files]
    fact_groundedness_files_full_path = [os.path.join(fact_groundedness_folder, file) for file in files]
    question_generation_files_full_path = [os.path.join(question_generation_folder, file) for file in files]

    queries = []
    corpus = []
    answers = []
    qrels = {}
    qrels_num_citations = {1:{}, 2:{}, 3:{}}
    qrels_by_time = {2030:{}, 2025: {}, 2020: {}, 2015: {}, 2010: {}}
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

        if any([item is None for item in [raw_facts, url_content_mapper, keypoints_mapper, fact_question_mapper]]): continue

        fact_ids_to_include = set(fact_question_mapper.keys()).intersection(set(keypoints_mapper.keys()))
        fact_ids_to_include = set([int(fact_id) for fact_id in fact_ids_to_include])

        # queries
        for fact_id, query in fact_question_mapper.items():
            fact_id = int(fact_id)
            if fact_id not in fact_ids_to_include: continue
            queries.append({"_id": f"{wiki_title}--{fact_id}", "text": query})


        # answer
        good_keypoints = {}
        for fact_url_kp_id, check_label in groundedness_check.items():
            if utilize_fact_groundedness_check and not check_label: continue
            fact_id, url, kp_id = fact_url_kp_id.split("--__--")
            fact_id = int(fact_id)
            kp_id = int(kp_id)

            if fact_id not in good_keypoints: good_keypoints[fact_id] = set()
            good_keypoints[fact_id].add(kp_id)

        for fact_id, keypoints in keypoints_mapper.items():
            fact_id = int(fact_id)
            query_id = f"{wiki_title}--{fact_id}"

            if fact_id not in fact_ids_to_include: continue

            answers.append({"_id": query_id, "text": [kp for kp_index, kp in enumerate(keypoints) if kp_index in good_keypoints[fact_id]]})

        # filter queries and answers based on if the answer contain any keypoints (if not then remove the id)
        good_qids = set([])
        for line in answers:
            query_id = line["_id"]
            keypoints = line["text"]
            if keypoints: good_qids.add(query_id)

        queries = [line for line in queries if line["_id"] in good_qids]
        answers = [line for line in answers if line["_id"] in good_qids]

        # corpus
        for fact in raw_facts:
            fact_id = fact["fact"]
            citation_urls = fact["citation_urls"]

            for url in citation_urls:
                content = url_content_mapper.get(url).get("url_content")
                if url_content_mapper.get(url, {}).get("error") or not content: continue

                published_date = url_content_mapper.get(url).get("published_date")
                published_year = datetime.strptime(published_date, "%Y-%m-%d").year if published_date else None

                corpus.append({"_id": url, "title": "", "text": content, "published_date": published_date})

        # qrels
        query_id_2_keypoints = {}
        for fact_url_kp_id, check_label in groundedness_check.items():
            fact_id, url, kp_id = fact_url_kp_id.split("--__--")
            query_id = f"{wiki_title}--{fact_id}"

            if int(fact_id) not in fact_ids_to_include: continue

            if query_id not in query_id_2_keypoints: query_id_2_keypoints[query_id] = set()
            if (utilize_fact_groundedness_check and check_label) or not utilize_fact_groundedness_check: query_id_2_keypoints[query_id].add(kp_id)

            if query_id not in qrels: qrels[query_id] = {}
            qrels[query_id][url] = int(check_label) if url not in qrels[query_id] else max(int(check_label), int(qrels[query_id][url]))
        
        for query_id, keypoints in query_id_2_keypoints.items():
            num_keypoints = len(keypoints)
            if num_keypoints not in qrels_num_citations: continue
            qrels_num_citations[num_keypoints][query_id] = qrels[query_id]
        

            

    
    write_to_jsonl(corpus, os.path.join(output_folder, "corpus.jsonl"))
    write_to_jsonl(queries, os.path.join(output_folder, "queries.jsonl"))
    write_to_jsonl(answers, os.path.join(output_folder, "answers.jsonl"))

    df_qrels = process_qrels(qrels)
    maybe_create_folder(os.path.join(output_folder, "qrels"))
    df_qrels.to_csv(os.path.join(output_folder, "qrels", "test.tsv"), sep='\t', index=False)

    # group queries and answers by specific number of documents

    for length in [1,2,3]:
        group_folder = os.path.join(output_folder, f"{length}_citations")
        group_qrels_folder = os.path.join(group_folder, "qrels")
        maybe_create_folder(group_folder)
        maybe_create_folder(group_qrels_folder)
        df_qrels = process_qrels(qrels_num_citations[length])

        df_qrels.to_csv(os.path.join(group_qrels_folder, f"test.tsv"), sep='\t', index=False)

        answers_grouped = [line for line in answers if line["_id"] in qrels_num_citations[length]]
        queries_grouped = [line for line in queries if line["_id"] in qrels_num_citations[length]]

        sampled_queries_grouped, sampled_answers_grouped = sample_qa(queries_grouped, answers_grouped, num_samples = 100, seed = 42)

        assert all([len(line["text"]) == length for line in answers_grouped])
        write_to_jsonl(sampled_answers_grouped, os.path.join(group_folder, f"answers.jsonl"))
        write_to_jsonl(sampled_queries_grouped, os.path.join(group_folder, f"queries.jsonl"))

    # group queries and answers by time

    # for year in [2030, 2025, 2020, 2015, 2010]:
    #     group_folder = os.path.join(output_folder, f"{year-5}_{year}")
    #     group_qrels_folder = os.path.join(group_folder, "qrels")
    #     maybe_create_folder(group_folder)
    #     maybe_create_folder(group_qrels_folder)
    #     df_qrels = process_qrels(qrels_by_time[year])

    #     df_qrels.to_csv(os.path.join(group_qrels_folder, f"test.tsv"), sep='\t', index=False)

    #     answers_grouped = [line for line in answers if line["_id"] in qrels_by_time[year]]
    #     queries_grouped = [line for line in queries if line["_id"] in qrels_by_time[year]]
    #     write_to_jsonl(answers_grouped, os.path.join(group_folder, f"answers.jsonl"))
    #     write_to_jsonl(queries_grouped, os.path.join(group_folder, f"queries.jsonl"))
        



if __name__ == "__main__":
    main()