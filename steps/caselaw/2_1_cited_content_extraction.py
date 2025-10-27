import os, hydra
from omegaconf import DictConfig
from tqdm import tqdm
from steps.utils.generic import read_json_or_jsonl, write_to_json


@hydra.main(version_base=None, config_path="../../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig)-> None:

    input_folder = cfg.step1.output_folder
    output_folder = cfg.step2_1.output_folder
    max_facts_per_page = cfg.step2_1.max_facts_per_page
    max_concurrents = cfg.step2_1.max_concurrents
    start_from = cfg.general.start_from

    files = os.listdir(input_folder)
    files = [file for file in files if file.endswith('.json')]
    input_files_full_path = [os.path.join(input_folder, file) for file in files]
    output_files_full_path = [os.path.join(output_folder, file) for file in files]

    for input_file_path, output_file_path in tqdm(zip(input_files_full_path, output_files_full_path), total = len(input_files_full_path)):
        if os.path.exists(output_file_path): continue
        url_content_mapper = {}
        all_urls = set()
        # read input
        input_data = read_json_or_jsonl(input_file_path)
        raw_facts = input_data.get("raw_facts")
        if not raw_facts: continue
        raw_facts = list(sorted(raw_facts, key = lambda x: x["fact"])) # sort based on position


        for fact in raw_facts[:max_facts_per_page]:
            citation_triplets = fact.get("citation_triplets", [])
            for triplet in citation_triplets:
                url_content_mapper[" ".join(triplet)] = {
                "url": "",
                "accessible": True,
                "url_content": "To be updated",
                "error": "",
                "published_date": "2024-05-10",
                "lang": "en"
            }

        to_save = {
            "title": input_data.get("title"),
            "create_timestamp": input_data.get("create_timestamp"),
            "timestamp": input_data.get("timestamp"),
            "url_content_mapper": url_content_mapper
        }

        write_to_json(data = to_save, filename = output_file_path)

if __name__ == "__main__":
    main()