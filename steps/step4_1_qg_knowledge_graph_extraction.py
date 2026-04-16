# this script is used to generate knowledge graph from fact (the first sub-step of step 4)
# the input include step2_2 and step3
# the output should look like 
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


import json, os, hydra
from collections import defaultdict
from llm_apis import init_llm, BaseLLMAPI
from typing import List
from omegaconf import DictConfig
from tqdm import tqdm

from steps.utils.prompts import GRAPH_BUILDER_PROMPT
from steps.utils.generic import read_json_or_jsonl, write_to_json, maybe_create_folder
from steps.utils.kg_based_qg import KGBasedQGUtils, KGBasedQGChecker

from concurrent.futures import ThreadPoolExecutor, as_completed



# @hydra.main(version_base=None, config_path="../conf/steps", config_name=os.getenv("CONFIG_NAME"))
# def main(cfg: DictConfig)-> None:
#     decontextualized_facts_folder = cfg.step2_2.output_folder
#     fact_groundedness_folder = cfg.step3.output_folder
#     output_folder = cfg.step4.output_folder + "_extracted_kg"
#     kg_completeness_word_check_threshold = cfg.step4.kg_completeness_word_check_threshold

#     llm_model_name = cfg.general.llm_model_name

#     maybe_create_folder(output_folder)

#     LLM = init_llm(llm_model_name)
    

#     kg_based_qg_utils = KGBasedQGUtils(
#         LLM = LLM,
#         question_generation_prompt = None, # in this step we will not be generating questions yet
#         graph_builder_prompt = GRAPH_BUILDER_PROMPT
#     )
#     kg_based_qg_checker = KGBasedQGChecker(
#         minicheck_model_name = None,
#         minicheck_cache_dir = None
#     )
    
#     files = os.listdir(decontextualized_facts_folder)
#     files = [file for file in files if file.endswith('.json')]
#     decontextualized_facts_files_full_path = [os.path.join(decontextualized_facts_folder, file) for file in files]
#     fact_groundedness_files_full_path = [os.path.join(fact_groundedness_folder, file) for file in files]
#     output_files_full_path = [os.path.join(output_folder, file) for file in files]


#     for dff_file_path, fgf_file_path, output_file_path in tqdm(zip(decontextualized_facts_files_full_path, 
#                                                       fact_groundedness_files_full_path, 
#                                                       output_files_full_path), total = len(files)):
#         if os.path.exists(output_file_path): continue
#         try:
#             dff_data = read_json_or_jsonl(dff_file_path)
#             fgf_data = read_json_or_jsonl(fgf_file_path)
#         except FileNotFoundError:
#             continue
        
#         keypoints_mapper = dff_data.get("keypoints_mapper")
#         groundedness_check = fgf_data.get("groundedness_check")


#         if not keypoints_mapper or not groundedness_check: continue

#         groundedness_check = {tuple(k.split("--__--")): v for k,v in groundedness_check.items()}
#         good_keypoints = set([])
#         for k, v in groundedness_check.items():
#             if v:
#                 good_keypoints.add(f"{k[0]}--__--{k[-1]}")

#         keypoints_mapper = {int(k): v for k,v in keypoints_mapper.items()}
#         keypoints_mapper_filtered = {}

#         for fact_id, keypoints in keypoints_mapper.items():
#             to_update = []
#             for kp_index, kp in enumerate(keypoints):
#                 if f"{fact_id}--__--{kp_index}" in good_keypoints: to_update.append(kp)
            
#             if to_update:
#                 keypoints_mapper_filtered[fact_id] = to_update
        
#         keypoints_mapper = keypoints_mapper_filtered


#         fact_kg_mapper = {}
#         for fact_id, keypoints in keypoints_mapper.items():
#             try:
#                 extracted_kg = kg_based_qg_utils.extract_kg_from_keypoints(keypoints = keypoints)
#             except Exception as e:
#                 extracted_kg = None
                
#             if extracted_kg: fact_kg_mapper[fact_id] = extracted_kg

#         if fact_kg_mapper:
#             to_save = {
#                 "title": dff_data.get("title"),
#                 "wiki_url": dff_data.get("wiki_url"),
#                 "topics": dff_data.get("topics"),
#                 "create_timestamp": dff_data.get("create_timestamp"),
#                 "timestamp": dff_data.get("timestamp"),
#                 "fact_kg_mapper": fact_kg_mapper
#             }

#             write_to_json(data = to_save, filename = output_file_path)



# if __name__ == "__main__":
#     main()



@hydra.main(version_base=None, config_path="../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig) -> None:
    decontextualized_facts_folder = cfg.step2_2.output_folder
    fact_groundedness_folder = cfg.step3.output_folder
    output_folder = cfg.step4.output_folder + "_extracted_kg"
    kg_completeness_word_check_threshold = cfg.step4.kg_completeness_word_check_threshold

    llm_model_name = cfg.general.llm_model_name

    max_workers = 16

    maybe_create_folder(output_folder)

    LLM = init_llm(llm_model_name)
    
    kg_based_qg_utils = KGBasedQGUtils(
        LLM = LLM,
        question_generation_prompt = None,
        graph_builder_prompt = GRAPH_BUILDER_PROMPT
    )
    
    files = os.listdir(decontextualized_facts_folder)
    files = [file for file in files if file.endswith('.json')]
    decontextualized_facts_files_full_path = [os.path.join(decontextualized_facts_folder, file) for file in files]
    fact_groundedness_files_full_path = [os.path.join(fact_groundedness_folder, file) for file in files]
    output_files_full_path = [os.path.join(output_folder, file) for file in files]

    def process_single_file(dff_path, fgf_path, out_path):
        if os.path.exists(out_path): 
            return
            
        try:
            dff_data = read_json_or_jsonl(dff_path)
            fgf_data = read_json_or_jsonl(fgf_path)
        except FileNotFoundError:
            return
        
        keypoints_mapper = dff_data.get("keypoints_mapper")
        groundedness_check = fgf_data.get("groundedness_check")

        if not keypoints_mapper or not groundedness_check: 
            return

        groundedness_check = {tuple(k.split("--__--")): v for k,v in groundedness_check.items()}
        good_keypoints = set([])
        for k, v in groundedness_check.items():
            if v:
                good_keypoints.add(f"{k[0]}--__--{k[-1]}")

        keypoints_mapper = {int(k): v for k,v in keypoints_mapper.items()}
        keypoints_mapper_filtered = {}

        for fact_id, keypoints in keypoints_mapper.items():
            to_update = []
            for kp_index, kp in enumerate(keypoints):
                if f"{fact_id}--__--{kp_index}" in good_keypoints: to_update.append(kp)
            
            if to_update:
                keypoints_mapper_filtered[fact_id] = to_update
        
        keypoints_mapper = keypoints_mapper_filtered

        fact_kg_mapper = {}
        for fact_id, keypoints in keypoints_mapper.items():
            try:
                extracted_kg = kg_based_qg_utils.extract_kg_from_keypoints(keypoints = keypoints)
            except Exception as e:
                extracted_kg = None
                
            if extracted_kg: fact_kg_mapper[fact_id] = extracted_kg

        if fact_kg_mapper:
            to_save = {
                "title": dff_data.get("title"),
                "wiki_url": dff_data.get("wiki_url"),
                "topics": dff_data.get("topics"),
                "create_timestamp": dff_data.get("create_timestamp"),
                "timestamp": dff_data.get("timestamp"),
                "fact_kg_mapper": fact_kg_mapper
            }

            write_to_json(data = to_save, filename = out_path)

    tasks = zip(decontextualized_facts_files_full_path, 
                fact_groundedness_files_full_path, 
                output_files_full_path)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_file, dff, fgf, out): out 
            for dff, fgf, out in tasks
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting KGs"):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing file {futures[future]}: {e}")


if __name__ == "__main__":
    main()