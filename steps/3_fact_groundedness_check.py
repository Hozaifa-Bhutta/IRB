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

import os
from argparse import ArgumentParser
from utils.generic import read_json_or_jsonl, write_to_json
from minicheck.minicheck import MiniCheck
from tqdm import tqdm

MINICHECK = {
    "model": None,
    "model_name": None
}
def init_minicheck(model_name='flan-t5-large', cache_dir='./ckpts'):
    if MINICHECK["model_name"] != model_name:
        print(f"Initializing Minicheck ({model_name})")
        model = MiniCheck(model_name=model_name, cache_dir=cache_dir)
        MINICHECK["model"] = model



def main():
    parser = ArgumentParser()
    parser.add_argument("--extracted_facts_folder", type = str, required = True)
    parser.add_argument("--crawled_url_content_folder", type = str, required = True)
    parser.add_argument("--decontextualized_facts_folder", type = str, required = True)
    parser.add_argument("--output_folder", type = str, required = True)

    args = parser.parse_args()

    extracted_facts_folder = args.extracted_facts_folder
    crawled_url_content_folder = args.crawled_url_content_folder
    decontextualized_facts_folder = args.decontextualized_facts_folder
    output_folder = args.output_folder

    init_minicheck(cache_dir="/scratch/lamdo/minicheck_ckpts")

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
        modified_fact_mapper = dff_data.get("modified_fact_mapper")

        if not raw_facts or not url_content_mapper or not modified_fact_mapper: continue

        modified_fact_mapper = {int(k): v for k,v in modified_fact_mapper.items()}

        groundedness_check = {}
        for fact in raw_facts:
            fact_id = fact.get("fact")
            citation_urls = fact.get("citation_urls")
            modified_fact = modified_fact_mapper.get(fact_id)
            if not modified_fact or not citation_urls: continue

            for url in citation_urls:
                content = url_content_mapper.get(url)["url_content"]
                if url_content_mapper.get(url, {}).get("error") or not content: continue

                groundedness_pred, raw_prob, _, _ = MINICHECK["model"].score(docs=[content], claims=[modified_fact])

                groundedness_check[f"{fact_id}--__--{url}"] = groundedness_pred[0] if groundedness_pred else 0

        to_save = {
            "title": ef_data.get("title"),
            "wiki_url": ef_data.get("wiki_url"),
            "groundedness_check": groundedness_check
        }
        write_to_json(data = to_save, filename = output_file_path)


if __name__ == "__main__":
    main()
