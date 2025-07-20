# Crawl the content of the citation urls. The input for this script is the output of step 1: fact extraction (1_fact_extraction.py)
# the output of this script will be a json file for each wikipedia page with names formatted like "{wiki page title}.json"
# Each file will contain
# {
#     "title": "...",
#     "wiki_url": "...",
#     "url_content_mapper": {
#         "url1": {
#             "accessible": "...",
#             "url_content": "...",
#             "title": "...",
#             "error": "...",
#         },
#         "url2": {
#             "accessible": "...",
#             "url_content": "...",
#             "title": "...",
#             "error": "...",
#         },
#     }
# }

import os, requests, time, io
from argparse import ArgumentParser
from tqdm import tqdm
from urllib.parse import urlparse
from PyPDF2 import PdfReader
from utils.generic import read_json_or_jsonl, write_to_json
from utils.archive_downloader import getArchiveContent, clear_downloads_dir
from utils.html_extraction import extract_text_from_html

request_counters = {}
MAX_REQUESTS_PER_MINUTE = 10  # Maximum requests per minute per domain

def rate_limited(func):
    def wrapper(url, *args, **kwargs):
        domain = urlparse(url).netloc
        request_counters.setdefault(domain, 0)

        # Wait if the domain has reached its request limit
        while request_counters[domain] >= MAX_REQUESTS_PER_MINUTE:
            print(f"Rate limit hit for {domain}. Waiting...")
            time.sleep(1)  # Check every second for available slots

        # Increment the counter for this domain
        request_counters[domain] += 1
        print(f"Request count for {domain}: {request_counters[domain]}")
       

        try:
            return func(url, *args, **kwargs)
        finally:
            # Reset the counter for the domain after 60 seconds
            time.sleep(60 / MAX_REQUESTS_PER_MINUTE)
            request_counters[domain] -= 1
            print(f"Decremented request count for {domain}: {request_counters[domain]}")
            

    return wrapper


@rate_limited
def is_url_accessible(url):
    print(f"Processing URL: {url}....")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError(f"Invalid URL format: {url}")
        domain = parsed_url.netloc.lower()

        if domain == "archive.org" or "internetarchive.org" in domain:
            print("Internet archive request...using library to download items")
            content = getArchiveContent(url)
            return True, content

        

        if "music" in domain:
            return False, "Music-related domain, not useful for fact extraction"

        if "porn" in domain or "adult" in domain:
            return False, "Sensored content in the URL"

        response = requests.get(url, timeout=20, headers=headers)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '')
        print(f"Response code for {url}: {response.status_code}")

        if "application/pdf" in content_type or url.endswith(".pdf"):
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) < 512:  # Minimum size check
                return False,"Incomplete PDF file"
            try:
                with io.BytesIO(response.content) as open_pdf_file:
                    reader = PdfReader(open_pdf_file)
                    text = "\n".join(page.extract_text() for page in reader.pages)
                    return True, text
            except Exception as pdf_error:
                return False,f"PDF error: {pdf_error}"

        if "text/html" in content_type:
            text = extract_text_from_html(response.text)
            if text:
                return True, text
            return False, "Empty HTML content"

        return False, f"Content type is {content_type}"
    except Exception as e:
        print(f"Error accessing URL {url}: {e}")
        return False, str(e)



def get_content_from_url(url):
    accessible, content = is_url_accessible(url)
    obj = {
        "url": url,
        "accessible": accessible,
        "url_content": content if accessible else "",
        "error": "" if accessible else str(content),
        # "id": f"{title}_para-{paragraph['id']}_url-{url_count}"
    }

    return obj


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type = str, required = True)
    parser.add_argument("--output_folder", type = str, required = True)
    parser.add_argument("--max_urls_per_page", type = int, default = 20)

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    max_urls_per_page = args.max_urls_per_page

    files = os.listdir(input_folder)
    files = [file for file in files if file.endswith('.json')]
    input_files_full_path = [os.path.join(input_folder, file) for file in files]
    output_files_full_path = [os.path.join(output_folder, file) for file in files]

    for input_file_path, output_file_path in tqdm(zip(input_files_full_path, output_files_full_path), total = len(input_files_full_path)):
        url_content_mapper = {}
        # read input
        input_data = read_json_or_jsonl(input_file_path)
        raw_facts = input_data.get("raw_facts")
        if not raw_facts: continue
        raw_facts = list(sorted(raw_facts, key = lambda x: x["fact"])) # sort based on position


        for fact in raw_facts[:max_urls_per_page]:
            citation_urls = fact.get("citation_urls", [])
            for url in citation_urls:
                content = get_content_from_url(url)
                if content:
                    url_content_mapper[url] = content

    
        to_save = {
            "title": input_data.get("title"),
            "wiki_url": input_data.get("wiki_url"),
            "url_content_mapper": url_content_mapper
        }

        write_to_json(data = to_save, filename = output_file_path)


if __name__ == "__main__":
    main()