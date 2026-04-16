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
from steps.utils.kg_based_qg import RuleBasedParaphrasing
from steps.utils.generic import read_json_or_jsonl, write_to_json, maybe_create_folder



@hydra.main(version_base=None, config_path="../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig)-> None:
    generated_questions_folder = cfg.step4.output_folder + "_generated_question"
    output_folder = cfg.step4.output_folder + "_generated_question_paraphrased"

    wikidump_date = cfg.general.wikidump_date

    maybe_create_folder(output_folder)

    paraphraser = RuleBasedParaphrasing()

    files = os.listdir(generated_questions_folder)
    files = [file for file in files if file.endswith('.json')]
    generated_questions_files_full_path = [os.path.join(generated_questions_folder, file) for file in files]
    output_files_full_path = [os.path.join(output_folder, file) for file in files]

    for generated_questions_file_path, output_file_path in tqdm(zip(
            generated_questions_files_full_path,
            output_files_full_path), total = len(files)):
        
        if os.path.exists(output_file_path): continue
        
        try:
            qg_data = read_json_or_jsonl(generated_questions_file_path)
        except FileNotFoundError:
            continue

        fact_question_mapper = {int(k): v for k,v in qg_data.get("fact_question_mapper").items()}
        new_fact_question_mapper = {}

        for fact_id, questions in fact_question_mapper.items():
            paraphrased_questions = paraphraser(questions, wikidump_date = wikidump_date, create_false_premise = False)
            false_premise_questions = paraphraser(questions, wikidump_date = wikidump_date, create_false_premise = True)

            temp = []
            for i in range(len(questions)):
                original = questions[i]
                paraphrased = paraphrased_questions[i]

                if not paraphrased["masked_kg"]: to_append = original
                else: to_append = paraphrased

                temp.append(to_append)
            
            false_premise_to_append = false_premise_questions[0]
            if false_premise_to_append["masked_kg"]:
                temp.append(false_premise_to_append)

            if temp:
                new_fact_question_mapper[fact_id] = temp

        if new_fact_question_mapper:
            to_save = {
                "title": qg_data.get("title"),
                "wiki_url": qg_data.get("wiki_url"),
                "topics": qg_data.get("topics"),
                "create_timestamp": qg_data.get("create_timestamp"),
                "timestamp": qg_data.get("timestamp"),
                "fact_question_mapper": new_fact_question_mapper
            }

            write_to_json(data = to_save, filename = output_file_path)




if __name__ == "__main__":
    main()