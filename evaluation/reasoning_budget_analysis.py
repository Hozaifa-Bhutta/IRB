

import json, os
import pandas as pd
from argparse import ArgumentParser
from collections import defaultdict
from steps.utils.generic import read_json_or_jsonl

def get_num_reasoning_tokens_openai(raw_response_query_id):
    try:
        return raw_response_query_id["usage"]["output_tokens_details"]["reasoning_tokens"]
    except Exception: return 0

def get_num_reasoning_tokens_bedrock(raw_response_query_id):
    return 0


def main():
    parser = ArgumentParser()
    parser.add_argument("--qa_metadata_folder", type = str, required = True)
    
    args = parser.parse_args()

    qa_metadata_folder = args.qa_metadata_folder

    folders = os.listdir(qa_metadata_folder)

    for folder in folders:
        eval_metadata_path = os.path.join(qa_metadata_folder, folder, "eval_metadata.json")
        eval_metadata = read_json_or_jsonl(eval_metadata_path)

        raw_response = eval_metadata["raw_response"]
        for query_id in raw_response:
            raw_response_query_id = raw_response[query_id]
            