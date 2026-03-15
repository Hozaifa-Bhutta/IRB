# python -m evaluation.combined_view --result_folder ./"results_llmbased_gitig_7March2026 (main)"

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

def rag_correct_incorrect_noretrieval(result_folder, hard = False):
    folders = [item for item in os.listdir(result_folder) if item.startswith("irb__")]

    target_split_name = "general" if not hard else "general_hard"

    data = defaultdict()
    data["model"] = []
    for folder in folders:
        if "rc0" in folder: continue
        data["model"].append(folder)
        for mode in ["retrieval_correct", "retrieval_incorrect"]:
            file = os.path.join(result_folder, folder, f"{mode}.csv")

            df = pd.read_csv(file)
            for line in df.to_dict(orient='records'):
                split_name = line["split"].replace(",", "")
                if split_name != target_split_name: continue

                for answer_type in ["correct", "incorrect"]:
                    if f"{mode}__{answer_type}" not in data: data[f"{mode}__{answer_type}"] = []
                    data[f"{mode}__{answer_type}"].append(round(line[answer_type] * 100, 1))

    return pd.DataFrame(data)




def retriever_performance(result_folder, metric = "recall"):
    folders = [item for item in os.listdir(result_folder) if item.startswith("retriever")]
    data = defaultdict()
    data["model"] = []
    for folder in folders:
        for mode in ["", "__reranked"]:
            file = os.path.join(result_folder, folder, f"retriever{mode}.csv")
            if not os.path.exists(file): continue
            df = pd.read_csv(file)

            data["model"].append(f"{folder}{mode}")
            for line in df.to_dict(orient='records'):
                split_name = line["split"].replace(",", "")
                if not line[metric]: continue
                if split_name not in data: data[split_name] = []
                data[split_name].append(round(line[metric] * 100, 1))

    return pd.DataFrame(data)


def reasoning_tokens(result_folder):
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
            data[split_name].append(round(line["avg_reasoning_tokens"], 1))

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
    output_correct_incorrect_rag_file = os.path.join(output_folder, "correct_incorrect.csv")
    output_correct_incorrect_hard_rag_file = os.path.join(output_folder, "correct_incorrect_hard.csv")
    output_reasoning_file = os.path.join(output_folder, "reasoning_tokens.csv")

    rag_correctness(result_folder).to_csv(output_correctness_file, index = False)
    retriever_performance(result_folder, metric = "recall").to_csv(output_retrieval_recall_file, index = False)
    retriever_performance(result_folder, metric = "ndcg").to_csv(output_retrieval_ndcg_file, index = False)
    rag_correct_incorrect_noretrieval(result_folder).to_csv(output_correct_incorrect_rag_file, index = False)
    rag_correct_incorrect_noretrieval(result_folder, hard = True).to_csv(output_correct_incorrect_hard_rag_file, index = False)
    reasoning_tokens(result_folder).to_csv(output_reasoning_file, index = False)

if __name__ == "__main__":
    main()