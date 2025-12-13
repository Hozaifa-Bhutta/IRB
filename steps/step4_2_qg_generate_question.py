import json, hydra, os, random
import numpy as np
from collections import defaultdict
from omegaconf import DictConfig
from steps.utils.prompts import QUESTION_GENERATION_PROMPT_FROM_KG_SINGLE_STEP
from steps.utils.kg_based_qg import KGBasedQGUtils, KGBasedQGChecker
from steps.utils.generic import read_json_or_jsonl, write_to_json, maybe_create_folder
from llm_apis import init_llm
from tqdm import tqdm
from typing import List, Dict



@hydra.main(version_base=None, config_path="../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig)-> None:
    decontextualized_facts_folder = cfg.step2_2.output_folder
    fact_groundedness_folder = cfg.step3.output_folder
    extracted_kg_folder = cfg.step4.output_folder + "_extracted_kg"
    output_folder = cfg.step4.output_folder + "_generated_question"

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

        # prepare masked knowledge graph for single-hop questions
        single_hop_masked_kg_mapper = {}
        for fact_id, keypoints in keypoints_mapper.items():
            graph_data = fact_kg_mapper.get(fact_id)
            if not graph_data: continue
            try:
                masked_knowledge_graph_for_single_hop = kg_based_qg_utils.knowledge_graph_masking_single_hop(
                    knowledge_graph = graph_data,
                    keypoints = keypoints
                )
            except Exception: continue
            if masked_knowledge_graph_for_single_hop: single_hop_masked_kg_mapper[fact_id] = masked_knowledge_graph_for_single_hop

        # try to find pairs of single-hop masked kg to piece together to create two-hop questions
        fact_question_mapper = {}
        for fact_id, keypoints in keypoints_mapper.items():
            graph_data = fact_kg_mapper.get(fact_id)
            if not graph_data or not single_hop_masked_kg_mapper.get(fact_id): continue

            this_fact_questions = []
            all_masked_knowledge_graphs = [single_hop_masked_kg_mapper[fact_id]]

            for fact_id_2, keypoints_2 in keypoints_mapper.items():
                if fact_id == fact_id_2 or not single_hop_masked_kg_mapper.get(fact_id_2): continue

                try:
                    masked_knowledge_graph_for_two_hop = kg_based_qg_utils.knowledge_graph_masking_multi_hop(
                        masked_kg_1 = single_hop_masked_kg_mapper[fact_id],
                        masked_kg_2 = single_hop_masked_kg_mapper[fact_id_2]
                    )
                except Exception: continue

                if masked_knowledge_graph_for_two_hop: 
                    masked_knowledge_graph_for_two_hop["aux_fact_id"] = fact_id_2
                    all_masked_knowledge_graphs.append(masked_knowledge_graph_for_two_hop)

            all_masked_knowledge_graphs = all_masked_knowledge_graphs[:2]


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

            all_masked_knowledge_graphs = temp

            for line in all_masked_knowledge_graphs:
                try:
                    masked_kg = line["masked_kg"]
                    num_hops = line["num_hops"]
                    aux_fact_id = line.get("aux_fact_id")
                    masked_keypoints_str = line["masked_keypoints_str"]

                    questions = kg_based_qg_utils.generate_question_from_masked_kg_step_by_step(masked_kg)

                    question_progression_check = kg_based_qg_checker.check_correctness_of_question_progression(
                        generated_questions = questions, masked_knowledge_graph = masked_kg)
                    if question_progression_check:
                        this_fact_questions.append(
                            {
                                "question": questions[-1],
                                "num_hops": num_hops,
                                "aux_fact_id": aux_fact_id,
                                "paraphrase": line.get("paraphrase"),
                                "false_premise": line.get("false_premise") is True,
                                "masked_kg": masked_kg,
                                "masked_keypoints_str": masked_keypoints_str
                            }
                        )
                except Exception as e: continue

            if this_fact_questions:
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