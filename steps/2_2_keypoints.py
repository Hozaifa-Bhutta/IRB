# script to extract and decontextualize keypoints from facts
# input is output of step 1
# output are the keypoints extracted for each fact, with the following format
# {
#     "title": "...",
#     "wiki_url": "...",
#     "keypoints_mapper": {
#         "fact1 (sentence id)": ["keypoint1", "keypoint2"]
#     }
# }

import os, json, hydra, time
from datetime import datetime
from omegaconf import DictConfig
from argparse import ArgumentParser
from tqdm import tqdm
from typing import List, Union
from utils.generic import read_json_or_jsonl, write_to_json
from utils.openai_utils import init_client, OPENAI_CLIENT
from utils.prompts import KEYPOINT_EXTRACTION_PROMPT


def get_first_paragraph(extracted_sentences: List[str]) -> str:
    """Get the first paragraph from the extracted sentences.
    Parameters
    ----------
        extracted_sentences : List[str]
            List of sentences extracted from the wiki page.
    Returns
    -------
        str
            The first paragraph as a string.
    """
    results = []
    for sent in extracted_sentences:
        if sent.startswith("SECTION:"): break
        results.append(sent)
    
    return " ".join(results)


def get_section_context(fact: int, extracted_sentences: List[str]) -> str:
    """Return the section name and the sentences in that section up to the given fact.
    Parameters
    ----------
        fact : int
            The index of the fact sentence.
        extracted_sentences : List[str]
            List of sentences extracted from the wiki page.
    Returns
    -------
        str
            A formatted string containing the section name and the sentences leading 
        up to the fact.
    """
    results = []
    section_name = None

    assert fact < len(extracted_sentences)

    for i in reversed(range(fact + 1)):
        sent = extracted_sentences[i]
        if sent.startswith("SECTION:"):
            section_name = sent.replace("SECTION:", "").strip()

            break
        results.append(sent)

    results = list(reversed(results))

    return f"Section Name: {section_name}\n\n" + " ".join(results)




def create_keypoints(fact: int, 
                     marked_sentences: List[str],
                     extracted_sentences: List[str], 
                     wiki_title: str = None,
                     last_updated_date: str = None) -> Union[List[str], None]:
    """Create keypoints from a fact sentence using OpenAI's GPT model.
    Parameters
    ----------
        fact : int
            The index of the fact sentence in the marked_sentences list.
        marked_sentences : List[str]
            List of sentences with [KP] markers indicating keypoints.
        extracted_sentences : List[str]
            List of sentences extracted from the wiki page.
        context_window_size : int  NOTE: This parameter is currently not used.
        wiki_title : str, optional
            The title of the wiki page, by default None.
    Returns
    -------
        Union[List[str], None]
            A list of keypoints if successful, otherwise None.
    """
    
    fact_sentence = marked_sentences[fact]
    keypoint_count = fact_sentence.count("[KP]")

    if not keypoint_count:
        return None

    # surrounding context include: Title of the wiki page + The first paragraph (abstract) of the wiki page
    # + the previous context within the section the fact belongs to
    first_paragraph = get_first_paragraph(extracted_sentences)
    section_context = get_section_context(fact, extracted_sentences)
    surrounding_context = first_paragraph + "\n...\n" + section_context

    if wiki_title:
        surrounding_context = f"Document Title: {wiki_title}\n\n" + surrounding_context

    user_prompt = KEYPOINT_EXTRACTION_PROMPT["user"][:]\
        .replace("[ADD_CLAIM_HERE]", fact_sentence)\
        .replace("[ADD_CONTEXT_HERE]", surrounding_context)\
        .replace("[ADD_KEYPOINTS_COUNT_HERE]", str(keypoint_count))\
        .replace("[ADD_LAST_UPDATED_DATE]", last_updated_date)

    try:
        resp = OPENAI_CLIENT["client"].chat.completions.create(
            model=OPENAI_CLIENT["model"],
            messages=[
                {
                    "role": "system",
                    "content": KEYPOINT_EXTRACTION_PROMPT["system"]
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            max_tokens = 1024,
        )
    except Exception: return None

    result = resp.choices[0].message.content.strip()
    try:
        temp = result.replace("##KEYPOINTS##:", "").strip()
        res = json.loads(temp)
        print(keypoint_count, res)
        assert res and len(res) == keypoint_count

        return res
    
        raise ValueError
    except Exception: return None



@hydra.main(version_base=None, config_path="../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig)-> None:
    extracted_facts_folder = cfg.step1.output_folder
    crawled_url_content_folder = cfg.step2_1.output_folder
    output_folder = cfg.step2_2.output_folder
    local_llm_port = cfg.general.local_llm_port
    local_llm_model = cfg.general.local_llm_model
    openai_model_name = cfg.general.openai_model_name

    openai_api_key = os.getenv("OPENAI_API_KEY")

    init_client(openai_api_key, 
                openai_model_name = openai_model_name,
                local = local_llm_port is not None, 
                port = local_llm_port, 
                model_name = local_llm_model)

    files = os.listdir(extracted_facts_folder)
    files = [file for file in files if file.endswith('.json')]
    extracted_facts_files_full_path = [os.path.join(extracted_facts_folder, file) for file in files]
    crawled_url_content_files_full_path = [os.path.join(crawled_url_content_folder, file) for file in files]
    output_files_full_path = [os.path.join(output_folder, file) for file in files]

    for ef_file_path, cuc_file_path, output_file_path in tqdm(zip(extracted_facts_files_full_path, 
                                                                  crawled_url_content_files_full_path, 
                                                                  output_files_full_path), total = len(files)):
        if os.path.exists(output_file_path):
            continue
        try:
            ef_data = read_json_or_jsonl(ef_file_path)
            cuc_data = read_json_or_jsonl(cuc_file_path)
        except FileNotFoundError: continue

        assert ef_data.get("title") == cuc_data.get("title")

        raw_facts = ef_data.get("raw_facts")
        url_content_mapper = cuc_data.get("url_content_mapper")
        wiki_page_last_updated_date = ef_data.get("timestamp").split("T")[0]

        if not raw_facts or not url_content_mapper: continue

        url_content_mapper = {k: v for k,v in url_content_mapper.items() if v.get("accessible") is True and v.get("url_content")}

        raw_facts = list(sorted(raw_facts, key = lambda x: x["fact"])) # sort based on position

        extracted_sentences = ef_data.get("extracted_sentences")
        marked_sentences = ef_data.get("marked_sentences")

        keypoints_mapper = {}
        for fact in tqdm(raw_facts, desc = "Extracting and decontextualizing keypoints from facts"):
            citation_urls = fact.get("citation_urls")
            citation_urls = [url for url in citation_urls if url in url_content_mapper] if citation_urls else []
            if not citation_urls: continue

            keypoints_from_fact = create_keypoints(
                fact["fact"], 
                marked_sentences = marked_sentences,
                extracted_sentences=extracted_sentences, 
                wiki_title = ef_data.get("title"),
                last_updated_date = wiki_page_last_updated_date
            )

            if not keypoints_from_fact: continue
            keypoints_mapper[int(fact["fact"])] = keypoints_from_fact

            if not local_llm_model:
                time.sleep(0.2)

        to_save = {
            "title": ef_data.get("title"),
            "wiki_url": ef_data.get("wiki_url"),
            "topics": ef_data.get("topics"),
            "create_timestamp": ef_data.get("create_timestamp"),
            "timestamp": ef_data.get("timestamp"),
            "keypoints_mapper": keypoints_mapper
        }

        write_to_json(data = to_save, filename = output_file_path)


if __name__ == "__main__":
    main()