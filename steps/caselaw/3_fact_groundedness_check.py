# script to check if molecular fact can be found within the content of cited url
# the input include the output of step 1, 2_1 and 2_2
# the output should look like 
# {
#     "title": "...",
#     "wiki_url": "...",
#     "groundedness_check": {
#         "(factid, url)": "label (int, 0 or 1)"
#     }
# }

import os, hydra
from omegaconf import DictConfig
from argparse import ArgumentParser
from steps.utils.generic import read_json_or_jsonl, write_to_json
from steps.utils.prompts import GROUNDEDNESS_CHECK_PROMPT
from steps.utils.openai_utils import init_client, OPENAI_CLIENT
from steps.utils.token_counting import init_enc as init_tiktoken_enc, TIKTOKEN_ENC
from tqdm import tqdm
from typing import List, Dict, Union, Optional


@hydra.main(version_base=None, config_path="../../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig)-> None:

    extracted_facts_folder = cfg.step1.output_folder
    crawled_url_content_folder = cfg.step2_1.output_folder
    decontextualized_facts_folder = cfg.step2_2.output_folder
    output_folder = cfg.step3.output_folder
    max_tokens = cfg.step3.max_tokens

    local_llm_port = cfg.general.local_llm_port
    local_llm_model = cfg.general.local_llm_model
    openai_model_name = cfg.general.openai_model_name

    openai_api_key = os.getenv("OPENAI_API_KEY")

    init_client(openai_api_key, 
                openai_model_name = openai_model_name,
                local = local_llm_port is not None, 
                port = local_llm_port, 
                model_name = local_llm_model)
    
    if max_tokens is not None:
        # This will raise an error if we do not use OpenAI's model
        init_tiktoken_enc(model_name = openai_model_name)

    # minicheck_ckpt_path = cfg.step3.minicheck_ckpt_path
    # init_minicheck(cache_dir = minicheck_ckpt_path)


    files = os.listdir(extracted_facts_folder)
    files = [file for file in files if file.endswith('.json')]
    extracted_facts_files_full_path = [os.path.join(extracted_facts_folder, file) for file in files]
    crawled_url_content_files_full_path = [os.path.join(crawled_url_content_folder, file) for file in files]
    decontextualized_facts_files_full_path = [os.path.join(decontextualized_facts_folder, file) for file in files]
    output_files_full_path = [os.path.join(output_folder, file) for file in files]


    for ef_file_path, cuc_file_path, dff_file_path, output_file_path in tqdm(zip(extracted_facts_files_full_path, 
                                                                                crawled_url_content_files_full_path, 
                                                                                decontextualized_facts_files_full_path,
                                                                                output_files_full_path), total = len(files)):
        try:
            ef_data = read_json_or_jsonl(ef_file_path)
            cuc_data = read_json_or_jsonl(cuc_file_path)
            dff_data = read_json_or_jsonl(dff_file_path)
        except FileNotFoundError: continue

        raw_facts = ef_data.get("raw_facts")
        url_content_mapper = cuc_data.get("url_content_mapper")
        keypoints_mapper = dff_data.get("keypoints_mapper")

        if not raw_facts or not url_content_mapper or not keypoints_mapper: continue

        keypoints_mapper = {int(k): v for k,v in keypoints_mapper.items()}

        groundedness_check = {}
        for fact in raw_facts:
            fact_id = fact.get("fact")
            citation_triplets = fact.get("citation_triplets")
            citation_positions = fact.get("pos")
            keypoints = keypoints_mapper.get(fact_id)

            for kp_index, kp in enumerate(keypoints):
                for pos, triplet in zip(citation_positions, citation_triplets):
                    str_triplet = " ".join(triplet)
                    temp = url_content_mapper.get(str_triplet)
                    if not temp or kp_index != pos: continue
                    groundedness_check[f"{fact_id}--__--{str_triplet}--__--{kp_index}"] = True



        to_save = {
            "title": ef_data.get("title"),
            "wiki_url": ef_data.get("wiki_url"),
            "topics": ef_data.get("topics"),
            "create_timestamp": ef_data.get("create_timestamp"),
            "timestamp": ef_data.get("timestamp"),
            "groundedness_check": groundedness_check
        }
        write_to_json(data = to_save, filename = output_file_path)


if __name__ == "__main__":
    main()
