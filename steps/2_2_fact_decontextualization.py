# script to decontextualize facts (create molecular facts)
# input is output of step 1
# output are the molecular facts (modified facts) and should have the following format
# {
#     "title": "...",
#     "wiki_url": "...",
#     "modified_facts_mapper": {
#         "fact1 (sentence id)": "modified fact" 
#     }
# }


import os, json
from argparse import ArgumentParser
from tqdm import tqdm
from utils.generic import read_json_or_jsonl, write_to_json
from utils.openai_utils import init_client, OPENAI_CLIENT
from utils.molecular_prompt import MOLECULAR_SYSTEM_PROMPT, MOLECULAR_USER_PROMPT


def create_molecular_fact(fact: int, extracted_sentences: list, context_window_size: int, molecular_system_prompt = MOLECULAR_SYSTEM_PROMPT, molecular_user_prompt = MOLECULAR_USER_PROMPT):
    fact_sentence = extracted_sentences[fact]

    surrounding_context = extracted_sentences[max(0, fact - context_window_size): min(len(extracted_sentences), fact + context_window_size)]
    surrounding_context = " ".join(surrounding_context)

    user_prompt = molecular_user_prompt.replace("[ADD_CLAIM_HERE]", fact_sentence)
    user_prompt = user_prompt.replace("[ADD_CONTEXT_HERE]", surrounding_context)

    resp = OPENAI_CLIENT["client"].chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": MOLECULAR_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.1,
        max_tokens = 100,
    )

    result = resp.choices[0].message.content.strip()
    try:
        return [x.strip() for x in result.split("##DECONTEXTUALIZED CLAIM##:")][1]
    except IndexError: return None



def main():
    parser = ArgumentParser()
    parser.add_argument("--extracted_facts_folder", type = str, required = True)
    parser.add_argument("--crawled_url_content_folder", type = str, required = True)
    parser.add_argument("--output_folder", type = str, required = True)
    parser.add_argument("--context_window_size", type = int, default = 5)
    parser.add_argument("--openai_api_key", type = str, required = True,
                        help = "OpenAI API key")

    args = parser.parse_args()

    extracted_facts_folder = args.extracted_facts_folder
    crawled_url_content_folder = args.crawled_url_content_folder
    output_folder = args.output_folder
    context_window_size = args.context_window_size
    openai_api_key = args.openai_api_key

    init_client(openai_api_key)

    files = os.listdir(extracted_facts_folder)
    files = [file for file in files if file.endswith('.json')]
    extracted_facts_files_full_path = [os.path.join(extracted_facts_folder, file) for file in files]
    crawled_url_content_files_full_path = [os.path.join(crawled_url_content_folder, file) for file in files]
    output_files_full_path = [os.path.join(output_folder, file) for file in files]

    for ef_file_path, cuc_file_path, output_file_path in tqdm(zip(extracted_facts_files_full_path, 
                                                                  crawled_url_content_files_full_path, 
                                                                  output_files_full_path), total = len(files)):
        try:
            ef_data = read_json_or_jsonl(ef_file_path)
            cuc_data = read_json_or_jsonl(cuc_file_path)
        except FileNotFoundError: continue

        raw_facts = ef_data.get("raw_facts")
        url_content_mapper = cuc_data.get("url_content_mapper")
        if not raw_facts or not url_content_mapper: continue

        raw_facts = list(sorted(raw_facts, key = lambda x: x["fact"])) # sort based on position

        extracted_sentences = ef_data.get("extracted_sentences")

        modified_fact_mapper = {}
        for fact in raw_facts:
            citation_urls = fact.get("citation_urls")
            citation_urls = [url for url in citation_urls if url in url_content_mapper] if citation_urls else []
            if not citation_urls: continue

            molecular_fact = create_molecular_fact(fact["fact"], extracted_sentences=extracted_sentences, context_window_size=context_window_size)
            if not molecular_fact: continue
            modified_fact_mapper[int(fact["fact"])] = molecular_fact

        to_save = {
            "title": ef_data.get("title"),
            "wiki_url": ef_data.get("wiki_url"),
            "modified_fact_mapper": modified_fact_mapper
        }

        write_to_json(data = to_save, filename = output_file_path)


if __name__ == "__main__":
    main()