# script to combine the results of the previous step into the final dataset
# the input include: output of steps 1, 2_1, 2_2, 3, 4

import os, hydra
import pandas as pd
from datetime import datetime
from omegaconf import DictConfig
from argparse import ArgumentParser
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
        modified_fact_mapper = dff_data.get("modified_fact_mapper")
        groundedness_check = fgf_data.get("groundedness_check")
        fact_question_mapper = qg_data.get("fact_question_mapper")

        wiki_title = ef_data.get("title")

        if any([item is None for item in [raw_facts, url_content_mapper, modified_fact_mapper, fact_question_mapper]]): continue

        good_facts = set([])
        for fact_id, query in fact_question_mapper.items():
            fact_id = int(fact_id)
            good_facts.add(fact_id)
            queries.append({"_id": f"{wiki_title}--{fact_id}", "text": query})

        for fact in raw_facts:
            fact_id = fact["fact"]
            citation_urls = fact["citation_urls"]
            positions = fact["pos"]
            group_ = set()
            earliest_published_year_ = set()
            if fact_id not in good_facts or not citation_urls: continue

            query_id = f"{wiki_title}--{fact_id}"
            if query_id not in qrels: qrels[query_id] = {}

            for url, pos in zip(citation_urls, positions):
                content = url_content_mapper.get(url).get("url_content")
                if url_content_mapper.get(url, {}).get("error") or not content: continue

                published_date = url_content_mapper.get(url).get("published_date")
                published_year = datetime.strptime(published_date, "%Y-%m-%d").year if published_date else None

                corpus.append({"_id": url, "title": "", "text": content, "published_date": published_date})

                qrels[query_id][url] = groundedness_check.get(f"{fact_id}--__--{url}", 0) + 1

                group_.add(pos)
                earliest_published_year_.add(published_year if published_year else 2006)
            
            group =len(group_)
            if group in [1,2,3]:
                qrels_num_citations[group][query_id] = qrels[query_id]

            earliest_published_year = min(earliest_published_year_) if earliest_published_year_ else 2006
            for year in [2030, 2025, 2020, 2015, 2010]:
                from_year = year - 5
                to_year = year
                if from_year <= earliest_published_year < to_year: 
                    qrels_by_time[year][query_id] = qrels[query_id]

        for fact_id, modified_fact in modified_fact_mapper.items():
            fact_id = int(fact_id)
            query_id = f"{wiki_title}--{fact_id}"

            if fact_id not in good_facts: continue

            answers.append({"_id": query_id, "text": modified_fact})
            

    
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
        write_to_jsonl(answers_grouped, os.path.join(group_folder, f"answers.jsonl"))
        write_to_jsonl(queries_grouped, os.path.join(group_folder, f"queries.jsonl"))

    # group queries and answers by time

    for year in [2030, 2025, 2020, 2015, 2010]:
        group_folder = os.path.join(output_folder, f"{year-5}_{year}")
        group_qrels_folder = os.path.join(group_folder, "qrels")
        maybe_create_folder(group_folder)
        maybe_create_folder(group_qrels_folder)
        df_qrels = process_qrels(qrels_by_time[year])

        df_qrels.to_csv(os.path.join(group_qrels_folder, f"test.tsv"), sep='\t', index=False)

        answers_grouped = [line for line in answers if line["_id"] in qrels_by_time[year]]
        queries_grouped = [line for line in queries if line["_id"] in qrels_by_time[year]]
        write_to_jsonl(answers_grouped, os.path.join(group_folder, f"answers.jsonl"))
        write_to_jsonl(queries_grouped, os.path.join(group_folder, f"queries.jsonl"))
        



if __name__ == "__main__":
    main()