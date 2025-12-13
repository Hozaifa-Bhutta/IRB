# python -m evaluation.combined_view --result_folder ./results_llmbased_gitig_14Nov2025

import json, os
import pandas as pd
from argparse import ArgumentParser
from collections import defaultdict

def format_dict(data: dict) -> str:
    parts = [f"{k}-{v}" for k, v in data.items()]
    
    return "_".join(parts)

def rag_correctness(result_folder):
    folders = [item for item in os.listdir(result_folder) if item.startswith("irb__")]

    data = defaultdict()
    data["model"] = []
    for folder in folders:
        file = os.path.join(result_folder, folder, "all.csv")
        df = pd.read_csv(file)

        data["model"].append(folder)
        for line in df.to_dict(orient='records'):
            split_name = line["split"].replace(",", "")
            if split_name not in data: data[split_name] = []
            data[split_name].append(round(line["correct"] * 100, 1))

    return pd.DataFrame(data)

def rag_correct_incorrect_noretrieval(result_folder):
    folders = [item for item in os.listdir(result_folder) if item.startswith("irb__")]

    data = defaultdict()
    data["model"] = []
    for folder in folders:
        file = os.path.join(result_folder, folder, "all.csv")


def retriever_performance(result_folder, metric = "recall"):
    folders = [item for item in os.listdir(result_folder) if item.startswith("retriever")]
    data = defaultdict()
    data["model"] = []
    for folder in folders:
        file = os.path.join(result_folder, folder, "retriever.csv")
        df = pd.read_csv(file)

        data["model"].append(folder)
        for line in df.to_dict(orient='records'):
            split_name = line["split"].replace(",", "")
            if not line[metric]: continue
            if split_name not in data: data[split_name] = []
            data[split_name].append(round(line[metric] * 100, 1))

    return pd.DataFrame(data)


def main():
    parser = ArgumentParser()
    parser.add_argument("--result_folder", type = str, required = True)
    
    args = parser.parse_args()

    result_folder = args.result_folder
    output_folder = os.path.join(result_folder, "combined_view")

    os.makedirs(output_folder, exist_ok = True)

    output_correctness_file = os.path.join(output_folder, "correctness.csv")
    output_retrieval_recall_file = os.path.join(output_folder, "retrieval_recall.csv")
    output_retrieval_ndcg_file = os.path.join(output_folder, "retrieval_ndcg.csv")

    rag_correctness(result_folder).to_csv(output_correctness_file, index = False)
    retriever_performance(result_folder, metric = "recall").to_csv(output_retrieval_recall_file, index = False)
    retriever_performance(result_folder, metric = "ndcg").to_csv(output_retrieval_ndcg_file, index = False)

if __name__ == "__main__":
    main()