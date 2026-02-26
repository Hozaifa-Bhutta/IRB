# script to check if molecular fact can be found within the content of cited url
# the input include the output of step 1, 2_1 and 2_2
# the output should look like 
# {
#     "title": "...",
#     "wiki_url": "...",
#     "topics": ["...", ...],
#     "create_timestamp": "...",
#     "timestamp": "...",
#     "groundedness_check": {
#         "{factid}--__--{url}--__--{keypoint_index}": "label (int, 0 or 1)"
#     }
# }

import os, hydra, time
from omegaconf import DictConfig
from llm_apis import init_llm, BaseLLMAPI
from tqdm import tqdm
from typing import List, Dict, Union, Optional
from langcodes import Language

from steps.utils.generic import read_json_or_jsonl, write_to_json
from steps.utils.prompts import GROUNDEDNESS_CHECK_PROMPT
from steps.utils.token_counting import init_enc as init_tiktoken_enc, TIKTOKEN_ENC


def lang_display_name_from_code(code: str) -> str:
    LANG_OVERRIDES = {
        "pt-br": "Portuguese (Brazil)",
        "zh-cn": "Chinese (China)",
        "zh-tw": "Chinese (Taiwan)",
        "yue": "Cantonese",
    }
    
    if not code: return code
    normalized = code.replace("_", "-").lower()
    if normalized in LANG_OVERRIDES:
        return LANG_OVERRIDES[normalized]
    try:
        return Language.get(normalized).display_name("en")
    except Exception:
        base = normalized.split("-")[0]
        try:
            return Language.get(base).display_name("en")
        except Exception:
            return code

def llm_based_groundedness_check_helper(
        kp: str, 
        content: str, 
        published_date: str, 
        lang: str,
        LLM: BaseLLMAPI) -> bool:
    
    if not kp or not content: return False

    lang_display_name = lang_display_name_from_code(code = lang)
    
    user_prompt = GROUNDEDNESS_CHECK_PROMPT["user"][:]\
        .replace("[ADD_KEYPOINT_HERE]", kp)\
        .replace("[ADD_CONTEXT_PUBLISHED_DATE_HERE]", published_date if published_date else "N/A")\
        .replace("[ADD_CONTEXT_LANGUAGE_HERE]", lang_display_name if lang else "N/A")\
        .replace("[ADD_CONTEXT_HERE]", content)

    try:
        _result = LLM.generate(
            system_prompt = GROUNDEDNESS_CHECK_PROMPT["system"],
            user_prompt = user_prompt,
            max_output_tokens = 16
        ).strip()

        time.sleep(0.1)
    except Exception: 
        print("Unsuccessful API call")
        return False

    if "Not Grounded" in _result:
        return False
    elif "Grounded" in _result:
        return True
    else: return False


def llm_based_groundedness_check_func(
        raw_facts: List, 
        keypoints_mapper: Dict[Union[int, str], List[str]], 
        url_content_mapper: Dict[str, Dict],
        max_tokens: Optional[int] = None,
        LLM: BaseLLMAPI = None) -> Dict[str, int]:

    """Check the groundedness of keypoints against the content of cited URLs.
    ----------
        raw_facts : List
            List of raw fact dictionaries, each containing 'fact', 'citation_urls', and 'pos'.
        keypoints_mapper : Dict[Union[int, str], str]
            A mapping from fact IDs to their corresponding keypoints.
        url_content_mapper : Dict[str, Dict[str, Union[str, bool]]]
            A mapping from URLs to their content and error status.
        max_tokens : int, optional
            The maximum number of tokens to consider from the URL content, by default 2000.
    Returns
    -------
        Dict[str, int]
            A mapping from "(factid)--__--(url)--__--(keypoint_index)" to groundedness label (0 or 1).
    """

    res = {}
    keypoints_contexts_pairs = []
    skipped = []
    for fact in raw_facts:
        fact_id = fact.get("fact")
        citation_urls = fact.get("citation_urls")
        citation_positions = fact.get("pos")
        keypoints = keypoints_mapper.get(fact_id)
        if not keypoints or not citation_urls: continue

        for kp_index, kp in enumerate(keypoints):
            for pos, url in zip(citation_positions, citation_urls):
                temp = url_content_mapper.get(url)
                if not temp or kp_index != pos: continue

                content = temp["url_content"]
                published_date = temp["published_date"]
                lang = temp["lang"]

                if max_tokens:
                    content = TIKTOKEN_ENC[LLM.model_name]["enc"].decode(
                        TIKTOKEN_ENC[LLM.model_name]["enc"].encode(content)[:max_tokens])

                if url_content_mapper.get(url, {}).get("error") or not content: continue

                keypoints_contexts_pairs.append([f"{fact_id}--__--{url}--__--{kp_index}", kp, content, published_date, lang])

    keypoints_contexts_pairs = sorted(keypoints_contexts_pairs, key = lambda x: x[0])
    groundedness_pred = [llm_based_groundedness_check_helper(kp, content, published_date, lang, LLM) \
                         for _, kp, content, published_date, lang in keypoints_contexts_pairs]
    
    assert len(groundedness_pred) == len(keypoints_contexts_pairs)
    for i in range(len(keypoints_contexts_pairs)):
        res[keypoints_contexts_pairs[i][0]] = groundedness_pred[i] if groundedness_pred else 0

    return res

    


@hydra.main(version_base=None, config_path="../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig)-> None:

    extracted_facts_folder = cfg.step1.output_folder
    crawled_url_content_folder = cfg.step2_1.output_folder
    decontextualized_facts_folder = cfg.step2_2.output_folder
    output_folder = cfg.step3.output_folder
    max_tokens = cfg.step3.max_tokens

    llm_model_name = cfg.general.llm_model_name

    LLM = init_llm(llm_model_name)
    
    if max_tokens is not None:
        # This will raise an error if we do not use OpenAI's model
        init_tiktoken_enc(model_name = llm_model_name)


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
        if os.path.exists(output_file_path): continue
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

        groundedness_check = llm_based_groundedness_check_func(
            raw_facts = raw_facts,
            keypoints_mapper = keypoints_mapper,
            url_content_mapper = url_content_mapper,
            max_tokens = max_tokens,
            LLM = LLM
        )

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
