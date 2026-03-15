# this script is used to validate the completeness of generated knowledge graph
# the input include step2_2 and step3, and step4_1
# the output should look exactly the same at the format of step4_1
# {
#     "title": "",
#     "wiki_url": "",
#     "topics": ["...", "..."],
#     "create_timestamp": "",
#     "timestamp": "",
#     "fact_kg_mapper": {
#         "{fact_id}": {
#             "head": "",
#             "head_type": "",
#             "relation": "",
#             "tail": "",
#             "tail_type": "",
#             "head_coverage": [keypoint indexes where head is present],
#             "tail_coverage": [keypoint indexes where tail is present]
#         }
#     }
# }

import json, hydra, os, random
import numpy as np
from collections import defaultdict
from omegaconf import DictConfig
from llm_apis import init_llm
from tqdm import tqdm
from typing import List, Dict


from steps.utils.prompts import GRAPH_COMPLETENESS_CHECK_PROMPT
from steps.utils.kg_based_qg import KGBasedQGUtils, KGBasedQGChecker
from steps.utils.generic import read_json_or_jsonl, write_to_json, maybe_create_folder


@hydra.main(version_base=None, config_path="../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig)-> None:
    decontextualized_facts_folder = cfg.step2_2.output_folder
    fact_groundedness_folder = cfg.step3.output_folder
    extracted_kg_folder = cfg.step4.output_folder + "_extracted_kg"
    output_folder = cfg.step4.output_folder + "_extracted_kg_checked"
    kg_completeness_word_check_threshold = cfg.step4.kg_completeness_word_check_threshold

    llm_model_name = cfg.general.llm_model_name

    maybe_create_folder(output_folder)

    LLM = init_llm(llm_model_name)

    kg_based_qg_checker = KGBasedQGChecker(
        minicheck_model_name = None,
        minicheck_cache_dir = None,
        graph_completeness_check_prompt = GRAPH_COMPLETENESS_CHECK_PROMPT,
        LLM = LLM
    )
    
    files = os.listdir(decontextualized_facts_folder)
    files = [file for file in files if file.endswith('.json')]
    decontextualized_facts_files_full_path = [os.path.join(decontextualized_facts_folder, file) for file in files]
    fact_groundedness_files_full_path = [os.path.join(fact_groundedness_folder, file) for file in files]
    extracted_kg_files_full_path = [os.path.join(extracted_kg_folder, file) for file in files]
    output_files_full_path = [os.path.join(output_folder, file) for file in files]

    # must remove in real run
    for dff_file_path, fgf_file_path, extracted_kg_path, output_file_path in tqdm(zip(decontextualized_facts_files_full_path, 
                                                                                    fact_groundedness_files_full_path, 
                                                                                    extracted_kg_files_full_path,
                                                                                    output_files_full_path), total = len(files)):
        if os.path.exists(output_file_path): continue
        
        try:
            dff_data = read_json_or_jsonl(dff_file_path)
            fgf_data = read_json_or_jsonl(fgf_file_path)
            ekg_data = read_json_or_jsonl(extracted_kg_path)
        except FileNotFoundError:
            continue


        keypoints_mapper = dff_data.get("keypoints_mapper")
        groundedness_check = fgf_data.get("groundedness_check")
        fact_kg_mapper = ekg_data.get("fact_kg_mapper")

        if not keypoints_mapper or not groundedness_check: continue

        groundedness_check = {tuple(k.split("--__--")): v for k,v in groundedness_check.items()}
        good_keypoints = set([])
        for k, v in groundedness_check.items():

            if v:
                good_keypoints.add(f"{k[0]}--__--{k[-1]}")

        keypoints_mapper = {int(k): v for k,v in keypoints_mapper.items()}
        fact_kg_mapper = {int(k): v for k,v in fact_kg_mapper.items()}
        keypoints_mapper_filtered = {}

        for fact_id, keypoints in keypoints_mapper.items():
            to_update = []
            for kp_index, kp in enumerate(keypoints):
                if f"{fact_id}--__--{kp_index}" in good_keypoints: to_update.append(kp)
            
            if to_update:
                keypoints_mapper_filtered[fact_id] = to_update
        
        keypoints_mapper = keypoints_mapper_filtered


        fact_kg_mapper_checked = {}
        for fact_id, keypoints in keypoints_mapper.items():
            good_kg = False
            try:
                good_kg = kg_based_qg_checker.check_completeness_of_extracted_kg_v2(
                    knowledge_graph = fact_kg_mapper[fact_id],
                    keypoints = keypoints,
                    word_check_threshold = kg_completeness_word_check_threshold,
                    verbose = False
                )
            except Exception as e:
                continue

            if good_kg:
                fact_kg_mapper_checked[fact_id] = fact_kg_mapper[fact_id]
            else: 
                with open('bad_case_logging_gitig_.txt', "a") as logging_f:
                    logging_f.write(f"""{dff_data.get("title")} - {fact_id}\n\n""")

        if fact_kg_mapper_checked:
            to_save = {
                "title": dff_data.get("title"),
                "wiki_url": dff_data.get("wiki_url"),
                "topics": dff_data.get("topics"),
                "create_timestamp": dff_data.get("create_timestamp"),
                "timestamp": dff_data.get("timestamp"),
                "fact_kg_mapper": fact_kg_mapper_checked
            }

            write_to_json(data = to_save, filename = output_file_path)



if __name__ == "__main__":
    main()