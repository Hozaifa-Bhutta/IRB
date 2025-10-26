import json, hydra, os, string, nltk
import numpy as np
from collections import defaultdict
from omegaconf import DictConfig
from utils.openai_utils import init_client, OPENAI_CLIENT
from utils.kg_based_qg import KGBasedQGChecker
from utils.generic import read_json_or_jsonl, write_to_json, maybe_create_folder
from tqdm import tqdm
from typing import List, Dict

@hydra.main(version_base=None, config_path="../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig)-> None:
    decontextualized_facts_folder = cfg.step2_2.output_folder
    fact_groundedness_folder = cfg.step3.output_folder
    extracted_kg_folder = cfg.step4.output_folder + "_extracted_kg"
    generated_question_folder = cfg.step4.output_folder + "_generated_question"
    output_folder = cfg.step4.output_folder + "_question_answerability"

    local_llm_port = cfg.general.local_llm_port
    local_llm_model = cfg.general.local_llm_model
    openai_model_name = cfg.general.openai_model_name

    openai_api_key = os.getenv("OPENAI_API_KEY")

    maybe_create_folder(output_folder)

    init_client(openai_api_key, 
                openai_model_name = openai_model_name,
                local = local_llm_port is not None, 
                port = local_llm_port, 
                model_name = local_llm_model)

    kg_based_qg_checker = KGBasedQGChecker(
        minicheck_model_name = None,
        minicheck_cache_dir = None,
        openai_client = OPENAI_CLIENT
    )
    
    files = os.listdir(decontextualized_facts_folder)
    files = [file for file in files if file.endswith('.json')]
    decontextualized_facts_files_full_path = [os.path.join(decontextualized_facts_folder, file) for file in files]
    fact_groundedness_files_full_path = [os.path.join(fact_groundedness_folder, file) for file in files]
    extracted_kg_files_full_path = [os.path.join(extracted_kg_folder, file) for file in files]
    generated_question_files_full_path = [os.path.join(generated_question_folder, file) for file in files]
    output_files_full_path = [os.path.join(output_folder, file) for file in files]

    for dff_file_path, fgf_file_path, extracted_kg_path, generated_question_path, output_file_path in tqdm(zip(decontextualized_facts_files_full_path, 
                                                                                                        fact_groundedness_files_full_path, 
                                                                                                        extracted_kg_files_full_path,
                                                                                                        generated_question_files_full_path,
                                                                                                        output_files_full_path), total = len(files)):

        try:
            dff_data = read_json_or_jsonl(dff_file_path)
            fgf_data = read_json_or_jsonl(fgf_file_path)
            ekg_data = read_json_or_jsonl(extracted_kg_path)
            qg_data = read_json_or_jsonl(generated_question_path)
        except FileNotFoundError:
            continue


        keypoints_mapper = dff_data.get("keypoints_mapper")
        groundedness_check = fgf_data.get("groundedness_check")
        fact_kg_mapper = ekg_data.get("fact_kg_mapper")
        fact_question_mapper = qg_data.get("fact_question_mapper")

        if not keypoints_mapper or not groundedness_check: continue

        groundedness_check = {tuple(k.split("--__--")): v for k,v in groundedness_check.items()}
        good_keypoints = set([])
        for k, v in groundedness_check.items():

            if v:
                good_keypoints.add(f"{k[0]}--__--{k[-1]}")

        keypoints_mapper = {int(k): v for k,v in keypoints_mapper.items()}
        fact_kg_mapper = {int(k): v for k,v in fact_kg_mapper.items()}
        fact_question_mapper = {int(k): v for k,v in fact_question_mapper.items()}
        keypoints_mapper_filtered = {}

        for fact_id, keypoints in keypoints_mapper.items():
            to_update = []
            for kp_index, kp in enumerate(keypoints):
                if f"{fact_id}--__--{kp_index}" in good_keypoints: to_update.append(kp)
            
            if to_update:
                keypoints_mapper_filtered[fact_id] = to_update
        
        keypoints_mapper = keypoints_mapper_filtered


        filtered_fact_question_mapper = {}
        for fact_id, keypoints in keypoints_mapper.items():
            if fact_id not in fact_question_mapper: continue

            all_questions = fact_question_mapper.get(fact_id)
            all_questions_filtered = []
            for line in all_questions:
                question = line["question"]
                answerability = kg_based_qg_checker.check_question_answerability(question)
                if answerability:
                    all_questions_filtered.append(line)

            if all_questions_filtered:
                filtered_fact_question_mapper[fact_id] = all_questions_filtered

        if filtered_fact_question_mapper:
            to_save = {
                "title": dff_data.get("title"),
                "wiki_url": dff_data.get("wiki_url"),
                "topics": dff_data.get("topics"),
                "create_timestamp": dff_data.get("create_timestamp"),
                "timestamp": dff_data.get("timestamp"),
                "fact_question_mapper": filtered_fact_question_mapper
            }

            write_to_json(data = to_save, filename = output_file_path)

if __name__ == "__main__":
    main()