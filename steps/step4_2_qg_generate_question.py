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

def prune_masked_knowledge_graph(masked_kg, max_children=3):
    """
    Prunes a masked knowledge graph based on children constraints for masked nodes.
    
    Rules:
    1. If a masked head points to a masked tail, drop all non-masked tails for that head.
    2. Otherwise, allow it to keep up to `max_children` non-masked tails.
    3. Unmasked heads are left completely alone.
    4. VALIDATION: The final masked node in the chain must have at least 1 child.
    """
    # # Step 1: Do a quick scan to see which masked heads have masked tails
    # masked_head_has_masked_tail = {}
    
    # for triplet in masked_kg:
    #     head = triplet["head"]
    #     tail = triplet["tail"]
        
    #     if head.startswith("<Unknown>"):
    #         if head not in masked_head_has_masked_tail:
    #             masked_head_has_masked_tail[head] = False
            
    #         # Flag it if we find a mask-to-mask connection
    #         if isinstance(tail, str) and tail.startswith("<Unknown>"):
    #             masked_head_has_masked_tail[head] = True

    # # Step 2: Build the new, pruned graph
    # pruned_kg = []
    # normal_child_counts = {head: 0 for head in masked_head_has_masked_tail.keys()}

    # for triplet in masked_kg:
    #     head = triplet["head"]
    #     tail = triplet["tail"]
        
    #     # We only apply pruning rules to masked heads
    #     if head.startswith("<Unknown>"):
    #         has_masked_child = masked_head_has_masked_tail[head]
            
    #         if has_masked_child:
    #             # Rule A: It has a masked child, so ONLY keep the masked children. 
    #             # Drop everything else attached to this node.
    #             if isinstance(tail, str) and tail.startswith("<Unknown>"):
    #                 pruned_kg.append(triplet)
    #         else:
    #             # Rule B: No masked children. Keep normal children up to the limit.
    #             if normal_child_counts[head] < max_children:
    #                 pruned_kg.append(triplet)
    #                 normal_child_counts[head] += 1
    #     else:
    #         # Rule C: The head is not masked, so we don't prune it.
    #         pruned_kg.append(triplet)

    # Step 3: Validation - Check if the last masked node has at least 1 child
    masked_nodes = set()
    for triplet in masked_kg:
        if triplet["head"].startswith("<Unknown>"):
            masked_nodes.add(triplet["head"])
        if isinstance(triplet["tail"], str) and triplet["tail"].startswith("<Unknown>"):
            masked_nodes.add(triplet["tail"])
            
    if masked_nodes:
        # Extract the index number from strings like "<Unknown> #2" to find the "last" one
        def get_mask_index(node_str):
            match = re.search(r'#(\d+)', node_str)
            return int(match.group(1)) if match else -1
            
        last_masked_node = max(masked_nodes, key=get_mask_index)
        
        # Check if this terminal node acts as a head (meaning it has at least 1 child, masked or unmasked)
        has_child = any(triplet["head"] == last_masked_node for triplet in masked_kg)
        
        if not has_child:
            return None # Reject the graph because the final hop is a dead end

    return masked_kg


@hydra.main(version_base=None, config_path="../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig)-> None:
    decontextualized_facts_folder = cfg.step2_2.output_folder
    fact_groundedness_folder = cfg.step3.output_folder
    extracted_kg_folder = cfg.step4.output_folder + "_extracted_kg_checked"
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
    extracted_kg_files_full_path = [os.path.join(extracted_kg_folder, file) for file in files]
    output_files_full_path = [os.path.join(output_folder, file) for file in files]

    # must remove in real run
    # for dff_file_path, fgf_file_path, extracted_kg_path, output_file_path in tqdm(zip(decontextualized_facts_files_full_path, 
    #                                                                                 fact_groundedness_files_full_path, 
    #                                                                                 extracted_kg_files_full_path,
    #                                                                                 output_files_full_path), total = len(files)):
    #     if os.path.exists(output_file_path): continue
        
    #     try:
    #         dff_data = read_json_or_jsonl(dff_file_path)
    #         fgf_data = read_json_or_jsonl(fgf_file_path)
    #         ekg_data = read_json_or_jsonl(extracted_kg_path)
    #     except FileNotFoundError:
    #         continue


    #     keypoints_mapper = dff_data.get("keypoints_mapper")
    #     groundedness_check = fgf_data.get("groundedness_check")
    #     fact_kg_mapper = ekg_data.get("fact_kg_mapper")

    #     if not keypoints_mapper or not groundedness_check: continue

    #     groundedness_check = {tuple(k.split("--__--")): v for k,v in groundedness_check.items()}
    #     good_keypoints = set([])
    #     for k, v in groundedness_check.items():

    #         if v:
    #             good_keypoints.add(f"{k[0]}--__--{k[-1]}")

    #     keypoints_mapper = {int(k): v for k,v in keypoints_mapper.items()}
    #     fact_kg_mapper = {int(k): v for k,v in fact_kg_mapper.items()}
    #     keypoints_mapper_filtered = {}

    #     for fact_id, keypoints in keypoints_mapper.items():
    #         to_update = []
    #         for kp_index, kp in enumerate(keypoints):
    #             if f"{fact_id}--__--{kp_index}" in good_keypoints: to_update.append(kp)
            
    #         if to_update:
    #             keypoints_mapper_filtered[fact_id] = to_update
        
    #     keypoints_mapper = keypoints_mapper_filtered

    #     # prepare masked knowledge graph for single-hop questions
    #     single_hop_masked_kg_mapper = {}
    #     for fact_id, keypoints in keypoints_mapper.items():
    #         graph_data = fact_kg_mapper.get(fact_id)
    #         if not graph_data: continue
    #         try:
    #             masked_knowledge_graph_for_single_hop = kg_based_qg_utils.knowledge_graph_masking_single_hop_v2(
    #                 knowledge_graph = graph_data,
    #                 keypoints = keypoints
    #             )
    #             masked_knowledge_graph_for_single_hop = list(masked_knowledge_graph_for_single_hop)
    #         except Exception as e: 
    #             print(e)
    #             continue
    #         if masked_knowledge_graph_for_single_hop: single_hop_masked_kg_mapper[fact_id] = masked_knowledge_graph_for_single_hop

    #     # try to find pairs of single-hop masked kg to piece together to create two-hop questions
    #     fact_question_mapper = {}

    #     for fact_id, keypoints in keypoints_mapper.items():
    #         if not single_hop_masked_kg_mapper.get(fact_id): continue

    #         all_masked_knowledge_graphs = []

    #         # --- RECURSIVE HELPER FUNCTION ---
    #         def build_multi_hop_chain(current_kg, current_chain, used_fact_ids, target_hop):
    #             # Base case: We reached our target number of hops!
    #             if current_kg["num_hops"] == target_hop:
    #                 return current_chain

    #             # Look for the next connecting puzzle piece
    #             for next_fact_id, next_keypoints in keypoints_mapper.items():
    #                 # Prevent circular logic (don't reuse a fact we already attached)
    #                 if next_fact_id in used_fact_ids: continue
    #                 if not single_hop_masked_kg_mapper.get(next_fact_id): continue

    #                 for next_single_hop_kg in single_hop_masked_kg_mapper[next_fact_id]:
    #                     try:
    #                         # Use your new dynamic combining function
    #                         combined_kg = kg_based_qg_utils.masked_knowledge_graph_combining(
    #                             masked_kg_1=current_kg, 
    #                             masked_kg_2=next_single_hop_kg
    #                         )
                            
    #                         if combined_kg:
    #                             # Keep track of all auxiliary facts used to build this chain
    #                             aux_facts = current_kg.get("aux_fact_ids", [])
    #                             combined_kg["aux_fact_ids"] = aux_facts + [next_fact_id]
                                
    #                             # We found a match! Dive deeper to find the NEXT hop.
    #                             result = build_multi_hop_chain(
    #                                 current_kg = combined_kg, 
    #                                 current_chain = current_chain + [combined_kg], 
    #                                 used_fact_ids = used_fact_ids + [next_fact_id],
    #                                 target_hop = target_hop
    #                             )
    #                             # If the deeper dive was successful, bubble the result all the way up
    #                             if result: return result 
                                
    #                     except Exception as e:
    #                         print(f"Error combining graphs: {e}")
    #                         continue
                            
    #             # If we exhausted all options and found no connections, return None (dead end)
    #             return None
    #         # ---------------------------------

    #         # Try to build a chain starting from every single-hop graph associated with this fact_id
    #         for start_kg in single_hop_masked_kg_mapper[fact_id]:
    #             start_kg["aux_fact_ids"] = [] # Initialize empty aux tracking for the base graph
                
    #             # Kick off the recursive search
    #             for target_hop in reversed(range(2, MAX_HOPS + 1)):
    #                 result_chain = build_multi_hop_chain(
    #                     current_kg = start_kg, 
    #                     current_chain = [start_kg], 
    #                     used_fact_ids = [fact_id],
    #                     target_hop = target_hop
    #                 )
    #                 if result_chain: break
                
    #             if result_chain:
    #                 all_masked_knowledge_graphs = result_chain
    #                 break # We successfully built the full chain! Stop searching for this fact_id.

    #         # The Fallback: If no multi-hop chain could be built, default to a random 1-hop
    #         if not all_masked_knowledge_graphs:
    #             # all_masked_knowledge_graphs = [random.choice(single_hop_masked_kg_mapper[fact_id])]
    #             all_masked_knowledge_graphs = []

    #         for item in all_masked_knowledge_graphs:
    #             item["masked_kg"] = prune_masked_knowledge_graph(item["masked_kg"])

    #         all_masked_knowledge_graphs_paraphrased = [
    #             {**item, **kg_based_qg_utils.helper.knowledge_graph_paraphrase(item["masked_kg"], wikidump_date, create_false_premise = False)} 
    #             for item in all_masked_knowledge_graphs
    #         ]
    #         all_masked_knowledge_graphs_false_premise = [
    #             {**item, **kg_based_qg_utils.helper.knowledge_graph_paraphrase(item["masked_kg"], wikidump_date, create_false_premise = True)} 
    #             for item in all_masked_knowledge_graphs
    #         ]

    #         temp = []
    #         for i in range(len(all_masked_knowledge_graphs)):
    #             original = all_masked_knowledge_graphs[i]
    #             paraphrased = all_masked_knowledge_graphs_paraphrased[i]

    #             if not paraphrased["masked_kg"]: to_append = original
    #             else: to_append = paraphrased

    #             temp.append(to_append)
            
    #         all_masked_knowledge_graphs_false_premise = [item for item in all_masked_knowledge_graphs_false_premise if item["masked_kg"]]
    #         if all_masked_knowledge_graphs_false_premise:
    #             temp.append(random.choice(all_masked_knowledge_graphs_false_premise))

    #         all_masked_knowledge_graphs = temp

    #         this_fact_questions = []
    #         for line in all_masked_knowledge_graphs:
    #             try:
    #                 masked_kg = line["masked_kg"]
    #                 num_hops = line["num_hops"]
    #                 aux_fact_ids = line.get("aux_fact_ids", [])
    #                 masked_keypoints_str = line["masked_keypoints_str"]

    #                 questions = kg_based_qg_utils.generate_question_from_masked_kg_step_by_step(masked_kg)

    #                 question_progression_check = kg_based_qg_checker.check_correctness_of_question_progression(
    #                     generated_questions = questions, masked_knowledge_graph = masked_kg)

    #                 # questions = ["testing"]
    #                 # question_progression_check = True


    #                 if question_progression_check:
    #                     this_fact_questions.append(
    #                         {
    #                             "question": questions[-1],
    #                             "num_hops": num_hops,
    #                             "aux_fact_ids": aux_fact_ids,
    #                             "paraphrase": line.get("paraphrase"),
    #                             "false_premise": line.get("false_premise") is True,
    #                             "masked_kg": masked_kg,
    #                             "masked_keypoints_str": masked_keypoints_str
    #                         }
    #                     )
    #             except Exception as e: continue

    #         if this_fact_questions:
    #             fact_question_mapper[fact_id] = this_fact_questions

    #     if fact_question_mapper:
    #         to_save = {
    #             "title": dff_data.get("title"),
    #             "wiki_url": dff_data.get("wiki_url"),
    #             "topics": dff_data.get("topics"),
    #             "create_timestamp": dff_data.get("create_timestamp"),
    #             "timestamp": dff_data.get("timestamp"),
    #             "fact_question_mapper": fact_question_mapper
    #         }

    #         write_to_json(data = to_save, filename = output_file_path)




    global_single_hop_masked_kg_mapper = {}
    global_keypoints_mapper = {}
    pages_to_process = []

    print("Phase 1: Building global single-hop knowledge graph pool...")
    for dff_file_path, fgf_file_path, extracted_kg_path, output_file_path in tqdm(zip(
            decontextualized_facts_files_full_path, 
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

        wiki_title = dff_data.get("title")

        keypoints_mapper = dff_data.get("keypoints_mapper")
        groundedness_check = fgf_data.get("groundedness_check")
        fact_kg_mapper = ekg_data.get("fact_kg_mapper")

        if not keypoints_mapper or not groundedness_check: continue

        groundedness_check = {tuple(k.split("--__--")): v for k,v in groundedness_check.items()}
        good_keypoints = set([])
        for k, v in groundedness_check.items():
            if v: good_keypoints.add(f"{k[0]}--__--{k[-1]}")

        keypoints_mapper = {int(k): v for k,v in keypoints_mapper.items()}
        fact_kg_mapper = {int(k): v for k,v in fact_kg_mapper.items()}
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
            graph_data = fact_kg_mapper.get(fact_id)
            if not graph_data: continue
            try:
                masked_knowledge_graph_for_single_hop = kg_based_qg_utils.knowledge_graph_masking_single_hop_v2(
                    knowledge_graph = graph_data,
                    keypoints = keypoints
                )
                masked_knowledge_graph_for_single_hop = list(masked_knowledge_graph_for_single_hop)
            except Exception as e: 
                print(e)
                continue
                
            if masked_knowledge_graph_for_single_hop: 
                local_single_hop_masked_kg_mapper[fact_id] = masked_knowledge_graph_for_single_hop
                
                # --- GLOBAL ASSIGNMENT ---
                # Create a unique global ID (e.g., "5_12" -> file index 5, fact id 12)
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
                            # print(f"Error combining graphs: {e}")
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

            # --- POST PROCESSING & SAVING ---
            all_masked_knowledge_graphs_paraphrased = [
                {**item, **kg_based_qg_utils.helper.knowledge_graph_paraphrase(item["masked_kg"], wikidump_date, create_false_premise = False)} 
                for item in all_masked_knowledge_graphs
            ]
            all_masked_knowledge_graphs_false_premise = [
                {**item, **kg_based_qg_utils.helper.knowledge_graph_paraphrase(item["masked_kg"], wikidump_date, create_false_premise = True)} 
                for item in all_masked_knowledge_graphs
            ]

            temp = []
            for i in range(len(all_masked_knowledge_graphs)):
                original = all_masked_knowledge_graphs[i]
                paraphrased = all_masked_knowledge_graphs_paraphrased[i]

                if not paraphrased["masked_kg"]: to_append = original
                else: to_append = paraphrased

                temp.append(to_append)
            
            all_masked_knowledge_graphs_false_premise = [item for item in all_masked_knowledge_graphs_false_premise if item["masked_kg"]]
            if all_masked_knowledge_graphs_false_premise:
                temp.append(random.choice(all_masked_knowledge_graphs_false_premise))

            for item in all_masked_knowledge_graphs:
                item["masked_kg"] = prune_masked_knowledge_graph(item["masked_kg"])

            all_masked_knowledge_graphs = temp

            this_fact_questions = []
            for line in all_masked_knowledge_graphs:
                try:
                    masked_kg = line["masked_kg"]
                    num_hops = line["num_hops"]
                    aux_fact_ids = line.get("aux_fact_ids", [])
                    masked_keypoints_str = line["masked_keypoints_str"]

                    questions = ["testing now"]
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