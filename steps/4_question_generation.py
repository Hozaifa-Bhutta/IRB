# script to generate questions
# the input should be the output of step 2_2
# the output of this script should have the following format
# {
#     "title": "...",
#     "wiki_url": "...",
#     "fact_question_mapper": {
#         "fact1 (sentence id)": "question"
#     }
# }

import os, json, hydra
from omegaconf import DictConfig
from argparse import ArgumentParser
from tqdm import tqdm
from typing import List
from utils.generic import read_json_or_jsonl, write_to_json
from utils.openai_utils import init_client, OPENAI_CLIENT
from utils.question_generation_prompt import QUESTION_GENERATION_SYSTEM_PROMPT, QUESTION_GENERATION_USER_PROMPT


def generate_question(keypoints: List[str], wiki_title: str):
    system_prompt = QUESTION_GENERATION_SYSTEM_PROMPT
    user_prompt = QUESTION_GENERATION_USER_PROMPT

    concatenated_keypoints = "\n".join(["- " + item for item in keypoints])
    user_prompt = user_prompt.replace("[ADD_KEYPOINTS_HERE]", concatenated_keypoints)
    user_prompt = user_prompt.replace("[ADD_TITLE_HERE]", wiki_title)

    resp = OPENAI_CLIENT["client"].chat.completions.create(
        model=OPENAI_CLIENT["model"],
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        # temperature=0.1,
        max_tokens = 512,
        # extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    result = resp.choices[0].message.content.strip()
    return result.replace("**QUESTION**:", "").strip()


@hydra.main(version_base=None, config_path="../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig):
    decontextualized_facts_folder = cfg.step2_2.output_folder
    fact_groundedness_folder = cfg.step3.output_folder
    output_folder = cfg.step4.output_folder

    utilize_fact_groundedness_check = cfg.general.utilize_fact_groundedness_check
    local_llm_port = cfg.general.local_llm_port
    local_llm_model = cfg.general.local_llm_model
    openai_model_name = cfg.general.openai_model_name

    openai_api_key = os.getenv("OPENAI_API_KEY")

    init_client(openai_api_key, 
                openai_model_name = openai_model_name,
                local = local_llm_port is not None, 
                port = local_llm_port, 
                model_name = local_llm_model)

    files = os.listdir(decontextualized_facts_folder)
    files = [file for file in files if file.endswith('.json')]
    decontextualized_facts_files_full_path = [os.path.join(decontextualized_facts_folder, file) for file in files]
    fact_groundedness_files_full_path = [os.path.join(fact_groundedness_folder, file) for file in files]
    output_files_full_path = [os.path.join(output_folder, file) for file in files]

    for dff_file_path, fgf_file_path, output_file_path in tqdm(zip(decontextualized_facts_files_full_path, 
                                                      fact_groundedness_files_full_path, 
                                                      output_files_full_path), total = len(files)):
        try:
            dff_data = read_json_or_jsonl(dff_file_path)
            fgf_data = read_json_or_jsonl(fgf_file_path)
        except FileNotFoundError:
            continue

        keypoints_mapper = dff_data.get("keypoints_mapper")
        groundedness_check = fgf_data.get("groundedness_check")
        print(fgf_data)

        if not keypoints_mapper or not groundedness_check: continue

        groundedness_check = {tuple(k.split("--__--")): v for k,v in groundedness_check.items()}
        good_keypoints = set([])
        for k, v in groundedness_check.items():
            if (utilize_fact_groundedness_check and v == 1) or not utilize_fact_groundedness_check:
                good_keypoints.add(f"{k[0]}--__--{k[-1]}")

        print(good_keypoints)
        keypoints_mapper = {int(k): v for k,v in keypoints_mapper.items()}
        keypoints_mapper_filtered = {}

        for fact_id, keypoints in keypoints_mapper.items():
            to_update = []
            for kp_index, kp in enumerate(keypoints):
                if f"{fact_id}--__--{kp_index}" in good_keypoints: to_update.append(kp)
            
            if to_update:
                keypoints_mapper_filtered[fact_id] = to_update
        
        keypoints_mapper = keypoints_mapper_filtered
        
        fact_question_mapper = {}
        for fact_id, keypoints in keypoints_mapper.items():
            question = generate_question(keypoints, wiki_title = dff_data.get("title"))
            fact_question_mapper[fact_id] = question

        
        to_save = {
            "title": dff_data.get("title"),
            "wiki_url": dff_data.get("wiki_url"),
            "fact_question_mapper": fact_question_mapper
        }

        write_to_json(data = to_save, filename = output_file_path)


if __name__ == "__main__":
    main()