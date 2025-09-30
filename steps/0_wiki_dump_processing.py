# script to process wiki dump zip file. The output would be a folder, with json files, names formatted like "{wiki page title}.json"
# Each file will contain
# {
#     "title": "...", # wiki page title
#     "wiki_url": "...", # wiki page url
#     "source": "wikitext...", # raw text of the wiki page
# }
import json, gzip, os, hydra
from datetime import datetime
from omegaconf import DictConfig
from argparse import ArgumentParser
from tqdm import tqdm
from utils.generic import maybe_create_folder, write_to_json
from typing import Optional


def read_wiki_dump_and_write(input_file: str, output_folder: str, max_pages: int, offset: int = 0, start_from: Optional[str] = None) -> None:
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
        start_from : str, optional
            Timestamp string in the format "%Y-%m-%d" to filter to only pages created after this date. Defaults to None (all pages are included).

    """
    
    assert input_file.endswith(".gz")

    # If 'start_from' is not provided, we assume all pages are included. 
    # So we set it to a date before Wikipedia was created to include all pages.
    if start_from is not None:
        start_from_date_obj = datetime.strptime(start_from, "%Y-%m-%d") 
    else:
        print("'start_from' not provided, default to 1990-01-01")
        start_from_date_obj = datetime(1990, 1, 1) 

    

    length_data = 0 # number of pages written
    count = 0 # number of pages iterated (including those not written due to offset or date filter)
    # unzip and read line by line
    with gzip.open(input_file, 'rt', encoding='utf-8') as f:
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

                    if not create_timestamp:
                        create_timestamp_obj = datetime(1998, 1, 1)
                    else: create_timestamp_obj = datetime.strptime(create_timestamp, "%Y-%m-%dT%H:%M:%SZ")

                    if create_timestamp_obj < start_from_date_obj: 
                        # skip pages created before the 'start_from' date to filter only relevant pages for IRB New
                        continue

                    # Construct the Wikipedia URL using the page ID for reference
                    url = f"https://en.wikipedia.org/?curid={obj.get('page_id')}"

                    to_write = {
                        "title": title, # wiki page title
                        "wiki_url": url, # wiki page url
                        "source": source, # raw text of the wiki page
                        "create_timestamp": create_timestamp # creation timestamp of the wiki page
                    }
                    try:
                        write_to_json(to_write, os.path.join(output_folder, f"{title}.json")) # write each page to a separate json file
                    except FileNotFoundError:
                        continue
                    
                    length_data += 1
                    pbar.update(1)
            
                count += 1
            # stop if we have written 'max_pages' pages
            if length_data == max_pages: break

@hydra.main(version_base=None, config_path="../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig) -> None:
    # parser = ArgumentParser()

    # parser.add_argument("--input_file", type = str, required = True)
    # parser.add_argument("--step0_output_folder", type = str, required = True)
    # parser.add_argument("--offset", type = int, default = cfg.step0.offset)
    # parser.add_argument("--max_pages", type = int, default = cfg.step0.max_pages)

    # args = parser.parse_args()

    input_file = cfg.step0.input_file #args.input_file
    output_folder = cfg.step0.output_folder #args.step0_output_folder
    offset = cfg.step0.offset
    max_pages = cfg.step0.max_pages
    start_from = cfg.general.start_from
    assert os.path.exists(input_file)

    read_wiki_dump_and_write(
        input_file = input_file,
        output_folder=output_folder,
        max_pages=max_pages,
        offset=offset,
        start_from=start_from
    )


if __name__ == "__main__":
    main()