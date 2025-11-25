# this is an optional step. The purpose is to down sample the dataset while keeping the distribution of the attributes as evenly as possible

import os, hydra, itertools, random, shutil
from omegaconf import DictConfig
from steps.utils.generic import read_json_or_jsonl, write_to_jsonl, write_to_json, maybe_create_folder
from typing import List, Dict


def attribute_binning(attributes: List[Dict]):
    # produce string version of attributes, and put query_id into bins
    res = {}
    for line in attributes:
        query_id = line["_id"]
        num_keypoints = line.get("num_keypoints")
        num_hops = line.get("num_hops")
        topics = set([top.split(".")[0] for top in line.get("topics")])

        _publication_years = list(itertools.chain.from_iterable(line.get("evidence_attr", {}).get("published_dates"))) + [line.get("wiki_create_timestamp")]
        _publication_years = [item[:4] for item in _publication_years if item]
        publication_year = min(_publication_years)

        langs = list(itertools.chain.from_iterable(line.get("evidence_attr", {}).get("langs")))
        is_multilingual = any([lang for lang in langs if lang != "en"])

        for top in topics:
            string_attr = f"{publication_year}__{num_keypoints}__{num_hops}__{top}__{is_multilingual}"
            if string_attr not in res: res[string_attr] = set([])
            res[string_attr].add(query_id)

    return {k: list(v) for k, v in res.items()}

def sample_based_on_attributes(attributes: List[Dict], num_samples: int):
    if len(attributes) < num_samples:
        return set([line["_id"] for line in attributes])
    bins = attribute_binning(attributes)

    print("Number of bins", len(bins))
    sampled_ids = set()
    while len(sampled_ids) < num_samples:
        for bin_name in bins:
            if not bins[bin_name]: continue
            index = random.choice(range(len(bins[bin_name])))
            sampled_id = bins[bin_name][index]
            sampled_ids.add(sampled_id)

            if sampled_id.endswith("--2"):
                
                sampled_id_ = sampled_id[:-3] + "--1"
                print(sampled_id, sampled_id_)
                sampled_ids.add(sampled_id_)

            bins[bin_name].pop(index)

            if len(sampled_ids) >= num_samples: break


    return sampled_ids



@hydra.main(version_base=None, config_path="../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig):
    full_benchmark_folder = cfg.step5.output_folder
    output_folder = cfg.step6.output_folder
    num_samples = int(cfg.step6.num_samples)


    sampled_queries_file = os.path.join(output_folder, "queries.jsonl")
    sampled_answers_file = os.path.join(output_folder, "answers.jsonl")
    sampled_attributes_file = os.path.join(output_folder, "attributes.jsonl")


    full_bench_queries = read_json_or_jsonl(os.path.join(full_benchmark_folder, "queries.jsonl"))
    full_bench_answers = read_json_or_jsonl(os.path.join(full_benchmark_folder, "answers.jsonl"))
    full_bench_attributes = read_json_or_jsonl(os.path.join(full_benchmark_folder, "attributes.jsonl"))

    sampled_ids = sample_based_on_attributes(full_bench_attributes, num_samples)


    sampled_queries = [line for line in full_bench_queries if line["_id"] in sampled_ids]
    sampled_answers = [line for line in full_bench_answers if line["_id"] in sampled_ids]
    sampled_attributes = [line for line in full_bench_attributes if line["_id"] in sampled_ids]


    write_to_jsonl(data = sampled_queries, filename = sampled_queries_file)
    write_to_jsonl(data = sampled_answers, filename = sampled_answers_file)
    write_to_jsonl(data = sampled_attributes, filename = sampled_attributes_file)

    full_bench_corpus_file = os.path.join(full_benchmark_folder, "corpus.jsonl")
    sampled_corpus_file = os.path.join(output_folder, "corpus.jsonl")
    shutil.copy(full_bench_corpus_file, sampled_corpus_file)

    full_bench_qrels_file = os.path.join(full_benchmark_folder, "qrels")
    sampled_qrels_file = os.path.join(output_folder, "qrels")
    shutil.copytree(full_bench_qrels_file, sampled_qrels_file)



if __name__ == "__main__":
    main()