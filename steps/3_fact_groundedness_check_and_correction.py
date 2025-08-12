# script to check if molecular fact can be found within the content of cited url
# the input include the output of step 1, 2_1 and 2_2
# the output should look like 
{
    "title": "...",
    "wiki_url": "...",
    "groundedness_check": {
        "{fact_id}--__--{url}": "label (int, 0 or 1)"
    },
    "fact_correction": {
        "{fact_id}--__--{url}": "corrected facts, if need correction"
    }
}

import os, hydra, re
from omegaconf import DictConfig
from utils.generic import read_json_or_jsonl, write_to_json
from minicheck.minicheck import MiniCheck
from tqdm import tqdm
from utils.openai_utils import init_client, OPENAI_CLIENT

MINICHECK = {
    "model": None,
    "model_name": None
}
def init_minicheck(model_name='flan-t5-large', cache_dir='./ckpts'):
    if MINICHECK["model_name"] != model_name:
        print(f"Initializing Minicheck ({model_name})")
        model = MiniCheck(model_name=model_name, cache_dir=cache_dir)
        MINICHECK["model"] = model


def groundedness_check(raw_facts: list, 
                       modified_fact_mapper: dict, 
                       url_content_mapper):
    
    res = {}
    facts_contexts_pairs = []
    for fact in raw_facts:
        fact_id = fact.get("fact")
        citation_urls = fact.get("citation_urls")
        modified_fact = modified_fact_mapper.get(fact_id)
        if not modified_fact or not citation_urls: continue

        for url in citation_urls:
            temp = url_content_mapper.get(url)
            if not temp: continue
            
            content = temp["url_content"]
            if url_content_mapper.get(url, {}).get("error") or not content: continue

            facts_contexts_pairs.append([f"{fact_id}--__--{url}", modified_fact, content])

    groundedness_pred, raw_prob, _, _ = MINICHECK["model"].score(docs=[line[2] for line in facts_contexts_pairs], 
                                                                 claims=[line[1] for line in facts_contexts_pairs])
    
    assert len(groundedness_pred) == len(facts_contexts_pairs)
    for i in range(len(facts_contexts_pairs)):
        res[facts_contexts_pairs[i][0]] = groundedness_pred[i] if groundedness_pred else 0


    return res


def content_truncation(url_content):
    words = re.split(r'\s+', url_content.strip())
    return " ".join(words[:5000])


def fact_correction_func(all_modified_facts, url_content):
    system_prompt = """The following claim is either not fully supported, or is contradicted by the context. Revise the claim so that all information 
    is fully supported by the context. If any information cannot be found in the context, remove it. Make only minimal changes to the original 
    claim if possible. Provide your revised claim only, without any additional explanation.
    """

    user_prompt = f"""The following are the attempts that you made to correct the fact:
    {';'.join(all_modified_facts)}
    
    URL Content:
    {content_truncation(url_content)}

Now try to correct the fact."""

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
        temperature=0.1,
        max_tokens = 100,
    )

    result = resp.choices[0].message.content.strip()
    return result




def groundedness_check_and_correct_facts(raw_facts: list, 
                                        modified_fact_mapper: dict, 
                                        url_content_mapper,
                                        num_iterations = 1):
    
    # first stage check
    init_groundedness_check_results = groundedness_check(raw_facts, modified_fact_mapper, url_content_mapper)

    previous_attempt_groundedness_check_results = [init_groundedness_check_results]
    previous_attempt_modified_fact = [modified_fact_mapper]
    for attempt in range(num_iterations):
        negative_results = {k:v for k,v in previous_attempt_groundedness_check_results[attempt].items() if not v}

        this_step_modified_fact_mapper = {}
        this_step_raw_facts = []
        for k in negative_results:
            fact_id, url = k.split("--__--")
            fact_id = int(fact_id)

            modified_fact_from_previous_attempts = [previous_attempt_modified_fact[j].get(fact_id) for j in range(attempt + 1)]
            url_content = url_content_mapper.get(url)

            if not modified_fact_from_previous_attempts or not url_content: continue

            print(modified_fact_from_previous_attempts)
            modified_fact_this_attempt = fact_correction_func(modified_fact_from_previous_attempts, url_content.get("url_content"))

            this_step_modified_fact_mapper[fact_id] = modified_fact_this_attempt

            this_step_raw_facts.append(
                {"fact": fact_id, "citation_urls": [url]}
            )

        this_step_groundedness_check_results = groundedness_check(this_step_raw_facts, this_step_modified_fact_mapper, url_content_mapper)
        print(this_step_groundedness_check_results)


        previous_attempt_groundedness_check_results.append(this_step_groundedness_check_results)
        previous_attempt_modified_fact.append(this_step_modified_fact_mapper)

    combined_groundedness_check_results = {}
    for groundedness_check_results in previous_attempt_groundedness_check_results:
        for k,v in groundedness_check_results.items():
            if k not in combined_groundedness_check_results or v > 0:
                combined_groundedness_check_results[k] = v

    
    fact_correction = {}
    for modified_fact in previous_attempt_modified_fact:
        for k, v in modified_fact.items():
            fact_correction[k] = v

    import json
    with open("test_gitig_.json", "w") as f:
        json.dump([previous_attempt_groundedness_check_results, previous_attempt_modified_fact], f, indent = 4)

    return combined_groundedness_check_results, fact_correction





@hydra.main(version_base=None, config_path="../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig):

    extracted_facts_folder = cfg.step1.output_folder
    crawled_url_content_folder = cfg.step2_1.output_folder
    decontextualized_facts_folder = cfg.step2_2.output_folder
    output_folder = cfg.step3.output_folder
    local_llm_port = cfg.general.local_llm_port
    local_llm_model = cfg.general.local_llm_model

    init_minicheck(cache_dir="/scratch/lamdo/minicheck_ckpts")
    
    openai_api_key = os.getenv("OPENAI_API_KEY")

    init_client(openai_api_key, 
                local = local_llm_port is not None, 
                port = local_llm_port, 
                model_name = local_llm_model)

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
        url_content_mapper = {k: v for k,v in url_content_mapper.items() if v.get("accessible") is True and v.get("url_content")}

        # groundedness_check_results = groundedness_check(raw_facts, modified_fact_mapper, url_content_mapper)
        groundedness_check_results, fact_correction_results = groundedness_check_and_correct_facts(raw_facts, modified_fact_mapper, url_content_mapper, num_iterations=2)

        to_save = {
            "title": ef_data.get("title"),
            "wiki_url": ef_data.get("wiki_url"),
            "groundedness_check": groundedness_check_results,
            "fact_correction": fact_correction_results
        }
        write_to_json(data = to_save, filename = output_file_path)


if __name__ == "__main__":
    main()
