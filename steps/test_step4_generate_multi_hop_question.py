# this script is used for step-by-step question generation (the second sub-step of step 4)
# the input is step2_2 and step3
# the output should look like
# {
#     "title": "",
#     "wiki_url": "",
#     "topics": ["...", "..."],
#     "create_timestamp": "",
#     "timestamp": "",
#     "fact_question_mapper": {
#         "fact_id": [
#             {
#                 "question": "",
#                 "num_hops": integer,
#                 "aux_fact_id": "",
#                 "paraphrase": "",
#                 "false_premise": "",
#                 "masked_kg": [
#                     {
#                         "head": "",
#                         "head_unmasked": "",
#                         "head_type": "",
#                         "relation": "",
#                         "tail": "",
#                         "tail_unmasked": "",
#                         "tail_type": ""
#                     },
#                 ]
#             }
#         ]
#     }
# }

import json, hydra, os, random, re
import numpy as np
from collections import defaultdict
from omegaconf import DictConfig
from llm_apis import init_llm
from tqdm import tqdm
from typing import List, Dict


from steps.utils.prompts import QUESTION_GENERATION_PROMPT_FROM_KG_SINGLE_STEP
from steps.utils.kg_based_qg import KGBasedQGUtils, KGBasedQGChecker
from steps.utils.generic import read_json_or_jsonl, write_to_json, maybe_create_folder





@hydra.main(version_base=None, config_path="../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig)-> None:
    decontextualized_facts_folder = cfg.step2_2.output_folder
    fact_groundedness_folder = cfg.step3.output_folder
    single_hop_question_folder = cfg.step4.output_folder + "_answerable_generated_single_hop_question"
    output_folder = cfg.step4.output_folder + "_generated_question"
    MAX_HOPS = cfg.step4.max_hops

    llm_model_name = cfg.general.llm_model_name
    wikidump_date = cfg.general.wikidump_date

    maybe_create_folder(output_folder)

    LLM = init_llm(llm_model_name)

    
    kg_based_qg_utils = KGBasedQGUtils(
        LLM = LLM,
        question_generation_prompt = QUESTION_GENERATION_PROMPT_FROM_KG_SINGLE_STEP, 
        graph_builder_prompt = None
    )
    kg_based_qg_checker = KGBasedQGChecker(
        minicheck_model_name = None,
        minicheck_cache_dir = None,
        LLM = LLM
    )
    
    files = os.listdir(decontextualized_facts_folder)
    files = [file for file in files if file.endswith('.json')]
    decontextualized_facts_files_full_path = [os.path.join(decontextualized_facts_folder, file) for file in files]
    fact_groundedness_files_full_path = [os.path.join(fact_groundedness_folder, file) for file in files]
    single_hop_question_files_full_path = [os.path.join(single_hop_question_folder, file) for file in files]
    output_files_full_path = [os.path.join(output_folder, file) for file in files]

    global_single_hop_masked_kg_mapper = {}
    global_keypoints_mapper = {}
    pages_to_process = []

    print("Phase 1: Building global single-hop knowledge graph pool...")
    for dff_file_path, fgf_file_path, single_hop_question_path, output_file_path in tqdm(zip(
            decontextualized_facts_files_full_path, 
            fact_groundedness_files_full_path, 
            single_hop_question_files_full_path,
            output_files_full_path), total = len(files)):
        
        if os.path.exists(output_file_path): continue
        
        try:
            dff_data = read_json_or_jsonl(dff_file_path)
            fgf_data = read_json_or_jsonl(fgf_file_path)
            shq_data = read_json_or_jsonl(single_hop_question_path)
        except FileNotFoundError:
            continue

        wiki_title = dff_data.get("title")

        keypoints_mapper = dff_data.get("keypoints_mapper")
        groundedness_check = fgf_data.get("groundedness_check")
        fact_single_hop_question_mapper = shq_data.get("fact_question_mapper")

        if not keypoints_mapper or not groundedness_check: continue

        groundedness_check = {tuple(k.split("--__--")): v for k,v in groundedness_check.items()}
        good_keypoints = set([])
        for k, v in groundedness_check.items():
            if v: good_keypoints.add(f"{k[0]}--__--{k[-1]}")

        keypoints_mapper = {int(k): v for k,v in keypoints_mapper.items()}
        fact_single_hop_question_mapper = {int(k): v for k,v in fact_single_hop_question_mapper.items()}
        keypoints_mapper_filtered = {}

        for fact_id, keypoints in keypoints_mapper.items():
            to_update = []
            for kp_index, kp in enumerate(keypoints):
                if f"{fact_id}--__--{kp_index}" in good_keypoints: to_update.append(kp)
            if to_update: keypoints_mapper_filtered[fact_id] = to_update
        
        keypoints_mapper = keypoints_mapper_filtered

        # prepare masked knowledge graph for single-hop questions
        local_single_hop_masked_kg_mapper = {}
        for fact_id, keypoints in keypoints_mapper.items():
            # masked_knowledge_graph_for_single_hop = fact_single_hop_question_mapper.get(fact_id, [])
            masked_knowledge_graph_for_single_hop = [{
                **item,
                "keypoints_str": "\n".join(keypoints)
            } for item in fact_single_hop_question_mapper.get(fact_id, [])]
                
            if masked_knowledge_graph_for_single_hop: 
                local_single_hop_masked_kg_mapper[fact_id] = masked_knowledge_graph_for_single_hop
                

                global_fact_id = f"{wiki_title}_{fact_id}"
                global_single_hop_masked_kg_mapper[global_fact_id] = masked_knowledge_graph_for_single_hop
                global_keypoints_mapper[global_fact_id] = keypoints

        # Cache this page's info for Pass 2 so we don't have to re-read files
        if local_single_hop_masked_kg_mapper:
            pages_to_process.append({
                "wiki_title": wiki_title,
                "dff_data": dff_data,
                "output_file_path": output_file_path,
                "local_single_hop_masked_kg_mapper": local_single_hop_masked_kg_mapper
            })


    # =====================================================================
    # PASS 2: Cross-Page Multi-Hop Generation & Saving
    # =====================================================================
    print("Phase 2: Generating cross-page multi-hop KGs and saving...")

    for page in tqdm(pages_to_process, desc="Processing Output Pages"):
        
        wiki_title = page["wiki_title"]
        dff_data = page["dff_data"]
        output_file_path = page["output_file_path"]
        local_single_hop_masked_kg_mapper = page["local_single_hop_masked_kg_mapper"]
        
        fact_question_mapper = {}

        for fact_id, local_start_kgs in local_single_hop_masked_kg_mapper.items():
            global_start_id = f"{wiki_title}_{fact_id}"
            all_masked_knowledge_graphs = []

            # --- RECURSIVE HELPER FUNCTION (Global Version) ---
            def build_multi_hop_chain(current_kg, current_chain, used_global_ids, target_hop):
                if current_kg["num_hops"] == target_hop:
                    return current_chain

                # Look through the ENTIRE GLOBAL DATASET
                for next_global_id, next_keypoints in global_keypoints_mapper.items():
                    if next_global_id in used_global_ids: continue
                    if not global_single_hop_masked_kg_mapper.get(next_global_id): continue

                    for next_single_hop_kg in global_single_hop_masked_kg_mapper[next_global_id]:
                        try:
                            combined_kg = kg_based_qg_utils.masked_knowledge_graph_combining(
                                masked_kg_1=current_kg, 
                                masked_kg_2=next_single_hop_kg
                            )
                            
                            if combined_kg:
                                aux_facts = current_kg.get("aux_fact_ids", [])
                                combined_kg["aux_fact_ids"] = aux_facts + [next_global_id]
                                
                                result = build_multi_hop_chain(
                                    current_kg = combined_kg, 
                                    current_chain = current_chain + [combined_kg], 
                                    used_global_ids = used_global_ids + [next_global_id],
                                    target_hop = target_hop
                                )
                                if result: return result 
                                
                        except Exception as e:
                            print(f"Error combining graphs: {e}")
                            continue
                            
                return None
            # ---------------------------------

            for start_kg in local_start_kgs:
                start_kg["aux_fact_ids"] = [] 
                
                result_chain = None
                for target_hop in reversed(range(2, MAX_HOPS + 1)):
                    result_chain = build_multi_hop_chain(
                        current_kg = start_kg, 
                        current_chain = [start_kg], 
                        used_global_ids = [global_start_id],
                        target_hop = target_hop
                    )
                    if result_chain: break
                
                if result_chain:
                    all_masked_knowledge_graphs = result_chain
                    break 

            if not all_masked_knowledge_graphs:
                all_masked_knowledge_graphs = [random.choice(local_start_kgs)]

            this_fact_questions = []
            for line in all_masked_knowledge_graphs:
                try:
                    question = line.get("question")
                    masked_kg = line["masked_kg"]
                    num_hops = line["num_hops"]
                    aux_fact_ids = line.get("aux_fact_ids", [])
                    masked_keypoints_str = line["masked_keypoints_str"]

                    if not question:
                        questions = kg_based_qg_utils.generate_question_from_masked_kg_step_by_step(masked_kg, masked_keypoints_str)
                        question_progression_check = kg_based_qg_checker.check_correctness_of_question_progression(
                            generated_questions = questions, masked_knowledge_graph = masked_kg)

                        # questions = ["testing"]
                        # question_progression_check = True
                    else:
                        questions = [question]
                        question_progression_check = True

                    if question_progression_check:
                        this_fact_questions.append(
                            {
                                "question": questions[-1],
                                "num_hops": num_hops,
                                "aux_fact_ids": aux_fact_ids,
                                "paraphrase": line.get("paraphrase"),
                                "false_premise": line.get("false_premise") is True,
                                "masked_kg": masked_kg,
                                "masked_keypoints_str": masked_keypoints_str
                            }
                        )
                except Exception as e: continue

            if this_fact_questions:
                # Revert the key back to the local integer ID for the per-page JSON mapping
                fact_question_mapper[fact_id] = this_fact_questions

        if fact_question_mapper:
            to_save = {
                "title": dff_data.get("title"),
                "wiki_url": dff_data.get("wiki_url"),
                "topics": dff_data.get("topics"),
                "create_timestamp": dff_data.get("create_timestamp"),
                "timestamp": dff_data.get("timestamp"),
                "fact_question_mapper": fact_question_mapper
            }

            write_to_json(data = to_save, filename = output_file_path)


if __name__ == "__main__":
    main()