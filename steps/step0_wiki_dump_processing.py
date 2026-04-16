# script to process wiki dump zip file. The output would be a folder, with json files, names formatted like "{wiki page title}.json"
# Each file will contain
# {
#     "title": "...", # wiki page title
#     "wiki_url": "...", # wiki page url
#     "source": "wikitext...", # raw text of the wiki page
#     "create_timestamp": "...", # creation timestamp of the wiki page
#     "timestamp": "...", # last updated timestamp
#     "topics": ["...", ...], # predicted outlink topics for the wiki page
# }


import json, bz2, os, hydra, re, requests
import concurrent.futures
from datetime import datetime
from omegaconf import DictConfig
from tqdm import tqdm
from typing import Optional, List, Dict, Any

from steps.utils.generic import maybe_create_folder, write_to_json


MIN_NUM_EXTERNAL_LINKS = 10 # keep only articles with at least this number of external links (references)

def get_articletopics_with_scores(weighted_tags: List[str]) -> List[Dict[str, Any]]:
    topic_list = []
    if not weighted_tags: return topic_list
    
    topic_pattern = re.compile(r'classification\.prediction\.articletopic/(.*?)\|(\d+)$')

    for tag in weighted_tags:
        match = topic_pattern.search(tag)
        if match:
            topic_path = match.group(1)
            raw_score = match.group(2)
            
            try:
                score = int(raw_score) / 1000.0
            except ValueError:
                continue
            
            topic_list.append({
                'topic': topic_path,
                'score': score
            })

    return topic_list


def read_wiki_dump_and_write(input_file: str, output_folder: str, max_pages: int, offset: int = 0, target_year: str = "2025", target_articles: set = None) -> None:
    """Reads a gzipped Wikipedia dump file and writes each page to a separate JSON file in the specified output folder.

    Parameters
    ----------
        input_file : str
            Path to the gzipped Wikipedia dump file (ends in .gz).
        output_folder : str
            Path to the folder where the output JSON files will be saved.
        max_pages : int
            Maximum number of wikipedia pages to process from the dump file.
        offset : int, optional
            Number of pages to skip from the start of the dump file. Defaults to 0.
        target_year : str, optional
            year string in the format "YYYY" to filter to only pages created in this year. Defaults to None (all pages are included).
    """
    
    assert input_file.endswith(".json.bz2")
    assert target_year is not None
    

    length_data = 0 # number of pages written
    count = 0 # number of pages iterated (including those not written due to offset or date filter)
    # unzip and read line by line
    with bz2.open(input_file, 'rt', encoding='utf-8') as f:
        pbar = tqdm(total = max_pages)
        for idx, line in enumerate(f):
            if (idx + 1) % 10000 == 0: print(f"{idx + 1} pages iterated")
            obj = json.loads(line) 
            # wikipage format: {"title": "...", "source_text": "...", "create_timestamp": "...", "page_id": ...}
            # some json objects are not wiki pages which is why we check for the "source_text" field
            if isinstance(obj, dict) and obj.get("source_text") is not None:
                # only start writing after 'offset' pages
                if count >= offset:
                    title = obj.get("title")
                    source = obj.get("source_text")
                    create_timestamp = obj.get("create_timestamp") # format: "%Y-%m-%dT%H:%M:%SZ"
                    timestamp = obj.get("timestamp") # format: "%Y-%m-%dT%H:%M:%SZ"
                    weighted_tags = obj.get("weighted_tags")

                    external_link = obj.get("external_link", [])
                    if len(external_link) < MIN_NUM_EXTERNAL_LINKS: continue

                    popularity_score = obj.get("popularity_score", 0)

                    create_year = create_timestamp[:4]
                    if (target_articles and title not in target_articles) or create_year != target_year:
                        continue

                    url = f"https://en.wikipedia.org/?curid={obj.get('page_id')}"

                    outlink_topics = get_articletopics_with_scores(weighted_tags)
                    topics = [topic['topic'] for topic in outlink_topics]

                    to_write = {
                        "title": title, # wiki page title
                        "wiki_url": url, # wiki page url
                        "source": source, # raw text of the wiki page
                        "create_timestamp": create_timestamp, # creation timestamp of the wiki page
                        "timestamp": timestamp, # last updated timestamp
                        "topics": topics, # predicted outlink topics for the wiki page
                        "popularity_score": popularity_score,
                    }
                    try:
                        write_to_json(to_write, os.path.join(output_folder, f"{title}.json")) # write each page to a separate json file
                    except Exception as e:
                        continue
                    
                    length_data += 1
                    pbar.update(1)
            
                count += 1

            if max_pages and length_data == max_pages: break

@hydra.main(version_base=None, config_path="../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig) -> None:

    input_folder = cfg.step0.input_folder
    output_folder = cfg.step0.output_folder
    offset = cfg.step0.offset
    max_pages = cfg.step0.max_pages
    target_year = str(cfg.general.target_year)
    assert os.path.exists(input_folder)


    input_files = os.listdir(input_folder)
    input_files = list(sorted([os.path.join(input_folder, input_file) for input_file in input_files if input_file.endswith(".json.bz2")]))


     # Set up the Process Pool 
    max_workers = 16
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for input_file in input_files:
            file_max_pages = int(max_pages / len(input_files)) + 1
            
            future = executor.submit(
                read_wiki_dump_and_write,
                input_file=input_file,
                output_folder=output_folder,
                max_pages=file_max_pages,
                offset=offset,
                target_year=target_year
            )
            futures.append(future)

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Total Files Processed"):
            try:
                future.result() 
            except Exception as exc:
                print(f"A process generated an exception: {exc}")


if __name__ == "__main__":
    main()