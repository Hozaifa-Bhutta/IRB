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

import os, json
from argparse import ArgumentParser
from tqdm import tqdm
from utils.generic import read_json_or_jsonl, write_to_json
from utils.openai_utils import init_client, OPENAI_CLIENT
from utils.question_generation_prompt import QUESTION_GENERATION_SYSTEM_PROMPT, QUESTION_GENERATION_USER_PROMPT


def generate_question(modified_fact):
    system_prompt = QUESTION_GENERATION_SYSTEM_PROMPT
    user_prompt = QUESTION_GENERATION_USER_PROMPT
    user_prompt = user_prompt.replace("[ADD_FACT_HERE]", modified_fact)

    resp = OPENAI_CLIENT["client"].chat.completions.create(
        model="gpt-4o",
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


def main():
    parser = ArgumentParser()
    parser.add_argument("--step2_2_output_folder", type = str, required = True)
    parser.add_argument("--step3_output_folder", type = str, required = True)
    parser.add_argument("--step4_output_folder", type = str, required = True)
    parser.add_argument("--openai_api_key", type = str, required = True,
                        help = "OpenAI API key")

    args = parser.parse_args()
    decontextualized_facts_folder = args.step2_2_output_folder
    fact_groundedness_folder = args.step3_output_folder
    output_folder = args.step4_output_folder
    openai_api_key = args.openai_api_key

    init_client(openai_api_key)

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

        modified_fact_mapper = dff_data.get("modified_fact_mapper")
        groundedness_check = fgf_data.get("groundedness_check")
        print(fgf_data)

        if not modified_fact_mapper or not groundedness_check: continue

        groundedness_check = {tuple(k.split("--__--")): v for k,v in groundedness_check.items()}
        good_facts = set([])
        for k, v in groundedness_check.items():
            if v == 1:
                good_facts.add(int(k[0]))

        print(good_facts)
        modified_fact_mapper = {int(k): v for k,v in modified_fact_mapper.items()}
        modified_fact_mapper = {k:v for k,v in modified_fact_mapper.items() if k in good_facts}
        
        fact_question_mapper = {}
        for fact_id, modified_fact in modified_fact_mapper.items():
            question = generate_question(modified_fact)
            fact_question_mapper[fact_id] = question

        
        to_save = {
            "title": dff_data.get("title"),
            "wiki_url": dff_data.get("wiki_url"),
            "fact_question_mapper": fact_question_mapper
        }

        write_to_json(data = to_save, filename = output_file_path)


if __name__ == "__main__":
    main()