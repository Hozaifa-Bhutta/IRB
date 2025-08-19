# script to process wiki dump. The output would be a folder, with json files, names formatted like "{wiki page title}.json"
# Each file will contain
# {
#     "title": "...",
#     "wiki_url": "...",
#     "source": "wikitext...",
# }
import json, gzip, os, hydra, datetime
from omegaconf import DictConfig
from argparse import ArgumentParser
from tqdm import tqdm
from utils.generic import maybe_create_folder, write_to_json


def read_wiki_dump_and_write(input_file, output_folder, max_pages, offset = 0, start_from = None):
    assert input_file.endswith(".gz")

    if start_from is not None:
        start_from_date_obj = datetime.strptime(start_from, "%Y-%m-%d") #(start_from, "%Y-%m-%dT%H:%M:%SZ")
    else:
        start_from_date_obj = datetime(1999, 1, 1, 0, 0, 0) # before wikipedia exist

    

    length_data = 0
    count = 0
    with gzip.open(input_file, 'rt', encoding='utf-8') as f:
        pbar = tqdm(total = max_pages)
        for line in f:
            obj = json.loads(line)

            if isinstance(obj, dict) and obj.get("source_text") is not None:
                if count >= offset:
                    title = obj.get("title")
                    source = obj.get("source_text")
                    create_timestamp = obj.get("create_timestamp")

                    if not create_timestamp:
                        create_timestamp_obj = datetime(1998, 1, 1, 0, 0, 0)
                    else: create_timestamp_obj = datetime.strptime(create_timestamp, "%Y-%m-%dT%H:%M:%SZ")

                    if create_timestamp_obj < start_from_date_obj: continue

                    url = f"https://en.wikipedia.org/?curid={obj.get('page_id')}"

                    to_write = {
                        "title": title,
                        "wiki_url": url,
                        "source": source
                    }
                    try:
                        write_to_json(to_write, os.path.join(output_folder, f"{title}.json"))
                    except FileNotFoundError:
                        continue
                    
                    length_data += 1
                    pbar.update(1)
                    print(title)
            
                count += 1

            if length_data == max_pages: break

@hydra.main(version_base=None, config_path="../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig):
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
    start_from = cfg.start_from

    assert os.path.exists(input_file)

    read_wiki_dump_and_write(
        input_file = input_file,
        output_folder=output_folder,
        max_pages=max_pages,
        offset=offset
    )


if __name__ == "__main__":
    main()