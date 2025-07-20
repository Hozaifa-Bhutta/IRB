# script to process wiki dump. The output would be a folder, with json files, names formatted like "{wiki page title}.json"
# Each file will contain
# {
#     "title": "...",
#     "wiki_url": "...",
#     "source": "wikitext...",
# }
import json, gzip, os
from argparse import ArgumentParser
from tqdm import tqdm
from utils.generic import maybe_create_folder, write_to_json


def read_wiki_dump_and_write(input_file, output_folder, max_pages, offset = 0):
    assert input_file.endswith(".gz")

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
                    url = f"https://en.wikipedia.org/?curid={obj.get('page_id')}"

                    to_write = {
                        "title": title,
                        "wiki_url": url,
                        "source": source
                    }
                    write_to_json(to_write, os.path.join(output_folder, f"{title}.json"))
                    length_data += 1
                    pbar.update(1)
                    print(title)
            
                count += 1

            if length_data == max_pages: break


def main():
    parser = ArgumentParser()

    parser.add_argument("--input_file", type = str, required = True)
    parser.add_argument("--step0_output_folder", type = str, required = True)
    parser.add_argument("--offset", type = int, default = 0)
    parser.add_argument("--max_pages", type = int, default = 100)

    args = parser.parse_args()

    input_file = args.input_file
    output_folder = args.step0_output_folder
    offset = args.offset
    max_pages = args.max_pages

    assert os.path.exists(input_file)

    read_wiki_dump_and_write(
        input_file = input_file,
        output_folder=output_folder,
        max_pages=max_pages,
        offset=offset
    )


if __name__ == "__main__":
    main()