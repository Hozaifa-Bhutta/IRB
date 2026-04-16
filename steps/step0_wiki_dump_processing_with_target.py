# script to process wiki dump zip file. The output would be a folder, with json files, names formatted like "{wiki page title}.json"
import json, bz2, os, hydra, re
import concurrent.futures
from omegaconf import DictConfig
from tqdm import tqdm
from typing import List, Dict, Any

from steps.utils.generic import maybe_create_folder, write_to_json

MIN_NUM_EXTERNAL_LINKS = 10 

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
    """Reads a gzipped Wikipedia dump file and writes each page to a separate JSON file."""
    
    assert input_file.endswith(".json.bz2")
    assert target_year is not None
    
    length_data = 0 
    count = 0 

    with bz2.open(input_file, 'rt', encoding='utf-8') as f:
        file_name = os.path.basename(input_file)
        pbar = tqdm(total=max_pages, desc=f"Processing {file_name}", leave=False, position=0)
        
        for idx, line in enumerate(f):
            try:
                obj = json.loads(line) 
            except json.JSONDecodeError:
                continue

            if isinstance(obj, dict) and obj.get("source_text") is not None:
                if count >= offset:
                    title = obj.get("title")
                    source = obj.get("source_text")
                    create_timestamp = obj.get("create_timestamp", "") 
                    timestamp = obj.get("timestamp") 
                    weighted_tags = obj.get("weighted_tags")

                    external_link = obj.get("external_link", [])
                    if len(external_link) < MIN_NUM_EXTERNAL_LINKS: 
                        count += 1
                        continue

                    popularity_score = obj.get("popularity_score", 0)

                    create_year = create_timestamp[:4] if create_timestamp else ""
                    if (target_articles and title not in target_articles) or create_year != target_year:
                        count += 1
                        continue

                    url = f"https://en.wikipedia.org/?curid={obj.get('page_id')}"
                    outlink_topics = get_articletopics_with_scores(weighted_tags)
                    topics = [topic['topic'] for topic in outlink_topics]

                    to_write = {
                        "title": title, 
                        "wiki_url": url, 
                        "source": source, 
                        "create_timestamp": create_timestamp, 
                        "timestamp": timestamp, 
                        "topics": topics, 
                        "popularity_score": popularity_score,
                    }
                    
                    try:
                        write_to_json(to_write, os.path.join(output_folder, f"{title}.json")) 
                    except Exception as e:
                        print(f"Error writing {title}: {e}")
                        pass
                    else:
                        length_data += 1
                        pbar.update(1)
            
                count += 1

            if max_pages and length_data >= max_pages: 
                break
                
        pbar.close()

@hydra.main(version_base=None, config_path="../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig) -> None:
    input_folder = cfg.step0.input_folder
    output_folder = cfg.step0.output_folder
    offset = cfg.step0.offset
    max_pages = cfg.step0.max_pages
    target_year = str(cfg.general.target_year)
    target_articles_file = cfg.step0.target_articles_file
    
    assert os.path.exists(input_folder)
    maybe_create_folder(output_folder) # Ensure output folder exists
    
    target_articles = None
    if os.path.exists(target_articles_file):
        with open(target_articles_file) as f:
            target_articles = set(line.strip() for line in f.readlines())

    input_files = os.listdir(input_folder)
    input_files = list(sorted([os.path.join(input_folder, input_file) for input_file in input_files if input_file.endswith(".json.bz2")]))

    # Set up the Process Pool 
    max_workers = 16
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for input_file in input_files:
            file_max_pages = int(max_pages / len(input_files)) + 1 if not target_articles else len(target_articles)
            
            future = executor.submit(
                read_wiki_dump_and_write,
                input_file=input_file,
                output_folder=output_folder,
                max_pages=file_max_pages,
                offset=offset,
                target_year=target_year,
                target_articles=target_articles
            )
            futures.append(future)

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Total Files Processed"):
            try:
                future.result() 
            except Exception as exc:
                print(f"A process generated an exception: {exc}")

if __name__ == "__main__":
    main()