# this is an optional step. The purpose is to down sample the dataset while keeping the distribution of the attributes as evenly as possible

import os, hydra, itertools, random, shutil, re
from omegaconf import DictConfig
from typing import List, Dict
from collections import Counter

from steps.utils.generic import read_json_or_jsonl, write_to_jsonl, write_to_json, maybe_create_folder

def attribute_binning(attributes: List[Dict]):
    # produce string version of attributes, and put query_id into bins
    res = {}
    for line in attributes:
        query_id = line["_id"]
        num_hops = line.get("num_hops")
        false_premise = line.get("false_premise")
        topics = set([top.split(".")[0] for top in line.get("topics")])

        _publication_years = list(line.get("evidence_attr", {}).get("published_dates")) + [line.get("wiki_create_timestamp")]
        _publication_years = [item[:4] for item in _publication_years if item]
        publication_year = min(_publication_years)

        langs = list(line.get("evidence_attr", {}).get("langs"))
        is_multilingual = any([lang for lang in langs if lang != "en"])

        for top in topics:
            string_attr = f"{publication_year}__{num_hops}__{top}__{is_multilingual}__{false_premise}"
            if string_attr not in res: res[string_attr] = set([])
            res[string_attr].add(query_id)

    return {k: list(v) for k, v in res.items()}


def get_all_dependencies(query_id: str, queries_ids: set) -> set:
    """
    Generalized dependency rule: For any N-hop question (ending in --N),
    returns a set of all lower-hop dependencies (from --1 up to --(N-1)).
    """
    deps = set()
    
    # Extract the trailing number after '--'
    match = re.search(r'--(\d+)$', query_id)
    
    # We only fetch dependencies for valid premises (not starting with '~')
    if match and not query_id.startswith("~"):
        current_hop = int(match.group(1))
        prefix = query_id[:match.start()]
        
        # Loop from 1 to N-1 and collect valid dependencies
        for hop in range(1, current_hop):
            dep_id = f"{prefix}--{hop}"
            if dep_id in queries_ids:
                deps.add(dep_id)
                
    return deps

def sample_based_on_attributes(attributes: List[Dict], num_samples: int, false_premise_limit: int, max_sample_per_page: int = 3):
    if len(attributes) < num_samples:
        return set([line["_id"] for line in attributes])
        
    queries_ids = set([line["_id"] for line in attributes])
    sampled_ids = set()
    sample_per_page = Counter()

    # --- PHASE 1: Retain ALL multi-hop questions and their dependencies ---
    # Assuming multi-hop is defined by num_hops > 1 in the dictionary
    multi_hop_ids = [line["_id"] for line in attributes if line.get("num_hops", 1) > 1]
    
    for q_id in multi_hop_ids:
        if len(sampled_ids) >= num_samples: break
        title = q_id.split("--")[0].replace("~", "")
        
        # Add the multi-hop question
        if q_id not in sampled_ids:
            sampled_ids.add(q_id)
            sample_per_page[title] += 1
            
        # Apply generalized dependency rule
        dependencies = get_all_dependencies(q_id, queries_ids)
        for dep_id in dependencies:
            if dep_id not in sampled_ids:
                sampled_ids.add(dep_id)
                dep_title = dep_id.split("--")[0].replace("~", "")
                sample_per_page[dep_title] += 1

    print(f"Pre-loaded {len(sampled_ids)} multi-hop questions and their dependencies.")

    # Check if retaining all multi-hops already filled or exceeded our quota
    if len(sampled_ids) >= num_samples:
        print("Target sample size reached entirely by multi-hop questions.")
        return sampled_ids

    # --- PHASE 2: Sample the remaining from bins ---
    bins = attribute_binning(attributes)
    print("Number of bins:", len(bins))
    
    while len(sampled_ids) < num_samples:
        added_this_round = False # Safety switch to prevent infinite loops
        
        for bin_name in bins:
            if not bins[bin_name]: continue

            index = random.choice(range(len(bins[bin_name])))
            sampled_id = bins[bin_name][index]
            title = sampled_id.split("--")[0].replace("~", "")

            # Constraint: Max per page
            if sample_per_page[title] >= max_sample_per_page:
                bins[bin_name].pop(index) # Remove from bin so we don't keep picking it
                continue

            # Constraint: False premise limit
            if sampled_id.startswith("~") and len([sid for sid in sampled_ids if sid.startswith("~")]) >= false_premise_limit:
                bins[bin_name].pop(index)
                continue
            
            # Since all multi-hops are pre-loaded in Phase 1, anything sampled here 
            # will likely be a 1-hop question, but we can safely call the dependency 
            # rule again just in case your dataset has edge cases.
            deps = get_all_dependencies(sampled_id, queries_ids)
            for dep_id in deps:
                if dep_id not in sampled_ids:
                    sampled_ids.add(dep_id)
                    dep_title = dep_id.split("--")[0].replace("~", "")
                    sample_per_page[dep_title] += 1

            sampled_ids.add(sampled_id)
            sample_per_page[title] += 1
            bins[bin_name].pop(index)
            added_this_round = True

            if len(sampled_ids) >= num_samples: 
                break
                
        # If we looped through all bins and couldn't add anything due to constraints, stop to avoid infinite loop
        if not added_this_round:
            print("Warning: Could not find enough valid samples that satisfy the constraints.")
            break

    return sampled_ids



@hydra.main(version_base=None, config_path="../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig):
    full_benchmark_folder = cfg.step5.output_folder
    output_folder = cfg.step6.output_folder
    num_samples = int(cfg.step6.num_samples)
    false_premise_limit = int(cfg.step6.false_premise_limit)


    sampled_queries_file = os.path.join(output_folder, "queries.jsonl")
    sampled_answers_file = os.path.join(output_folder, "answers.jsonl")
    sampled_attributes_file = os.path.join(output_folder, "attributes.jsonl")


    full_bench_queries = read_json_or_jsonl(os.path.join(full_benchmark_folder, "queries.jsonl"))
    full_bench_answers = read_json_or_jsonl(os.path.join(full_benchmark_folder, "answers.jsonl"))
    full_bench_attributes = read_json_or_jsonl(os.path.join(full_benchmark_folder, "attributes.jsonl"))

    sampled_ids = sample_based_on_attributes(full_bench_attributes, num_samples, false_premise_limit)


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