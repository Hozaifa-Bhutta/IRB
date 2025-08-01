# script to combine the results of the previous step into the final dataset
# the input include: output of steps 1, 2_1, 2_2, 3, 4

import os, hydra
from omegaconf import DictConfig
from argparse import ArgumentParser
from tqdm import tqdm
from utils.generic import read_json_or_jsonl, write_to_jsonl, write_to_json


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
            if fact_id not in good_facts or not citation_urls: continue

            query_id = f"{wiki_title}--{fact_id}"
            if query_id not in qrels: qrels[query_id] = {}

            for url in citation_urls:
                content = url_content_mapper.get(url)["url_content"]
                if url_content_mapper.get(url, {}).get("error") or not content: continue

                corpus.append({"_id": url, "title": "", "text": content})

                qrels[query_id][url] = groundedness_check.get(f"{fact_id}--__--{url}", 0)

        for fact_id, modified_fact in modified_fact_mapper.items():
            fact_id = int(fact_id)
            query_id = f"{wiki_title}--{fact_id}"

            if fact_id not in good_facts: continue

            answers.append({"_id": query_id, "text": modified_fact})
            

    
    write_to_jsonl(corpus, os.path.join(output_folder, "corpus.jsonl"))
    write_to_jsonl(queries, os.path.join(output_folder, "queries.jsonl"))
    write_to_jsonl(answers, os.path.join(output_folder, "answers.jsonl"))
    write_to_json(qrels, os.path.join(output_folder, "qrels.json"))


if __name__ == "__main__":
    main()