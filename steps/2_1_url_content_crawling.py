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
#             "published_date":"...",
#         },
#         "url2": {
#             "accessible": "...",
#             "url_content": "...",
#             "title": "...",
#             "error": "...",
#             "published_date":"...",
#         },
#     }
# }

import os, requests, time, io, asyncio, hydra
from datetime import datetime
from omegaconf import DictConfig
from tqdm import tqdm
from urllib.parse import urlparse
from cleantext import clean
from fast_langdetect import detect as language_detection_func
from utils.generic import read_json_or_jsonl, write_to_json
from utils.archive_downloader import getArchiveContent
from utils.html_extraction import extract_text_from_html, get_publication_date

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

def is_valid_date(published_time_str, start_from):
    if not start_from:
        return True
    if not published_time_str:
        return False
    try:
        published_date = datetime.strptime(published_time_str, "%Y-%m-%d")
        start_from_date = datetime.strptime(start_from, "%Y-%m-%d")
        return published_date >= start_from_date
    except ValueError:
        return False

@rate_limited
def is_url_accessible(url, start_from):
    time.sleep(0.2)
    # print(f"Processing URL: {url}....")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    is_accessible = False
    content_dict = {
        "content": None,
        "published_date": None,
        "lang": None
    }

    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError(f"Invalid URL format: {url}")
        domain = parsed_url.netloc.lower()

        if domain == "archive.org" or "internetarchive.org" in domain:
            print("Internet archive request...using library to download items")
            content = getArchiveContent(url)

            is_accessible = True
            content_dict["content"] = content
            return is_accessible, content_dict

        

        if "music" in domain:
            content_dict["content"] = "Music-related domain, not useful for fact extraction"
            return is_accessible, content_dict 

        if "porn" in domain or "adult" in domain:
            content_dict["content"] = "Sensored content in the URL"
            return is_accessible, content_dict 
        
        if ".pdf" in url:
            content_dict["content"] = "PDF file not supported"
            return is_accessible, content_dict

        response = requests.get(url, timeout=20, headers=headers)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '')
        print(f"Response code for {url}: {response.status_code}")

        if "application/pdf" in content_type or url.endswith(".pdf"):
            raise NotImplementedError("Pdf files not supported")
            

        if "text/html" in content_type:
            text = extract_text_from_html(response.content)
            published_date = get_publication_date(response.content)
            lang = None
            print(f"Published date for {url}: {published_date}. Start from: {start_from}. Valid: {is_valid_date(published_date, start_from)}")
            if not is_valid_date(published_date, start_from):
                content_dict["content"] = "Published date is before the start_from date"
                content_dict["published_date"] = published_date
                content_dict["lang"] = lang
                return is_accessible, content_dict
            elif not text:
                content_dict["content"] = "Empty HTML content"
                content_dict["published_date"] = published_date
                content_dict["lang"] = lang
                return is_accessible, content_dict
            else:
                is_accessible = True
                content_dict["content"] = text
                content_dict["published_date"] = published_date
                content_dict["lang"] = language_detection_func(text)[0]["lang"]
                return is_accessible, content_dict

        content_dict["content"] = f"Content type is {content_type}"
        return is_accessible, content_dict
    except Exception as e:
        print(f"Error accessing URL {url}: {e}")
        content_dict["content"] = str(e)
        return is_accessible, content_dict



async def is_url_accessible_async(url, start_from, semaphore):
    loop = asyncio.get_event_loop()
    async with semaphore:
        return await loop.run_in_executor(None, is_url_accessible, url, start_from)

async def check_urls_in_parallel(url_list, start_from, max_concurrent=5):
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [is_url_accessible_async(url, start_from, semaphore) for url in url_list]
    return await asyncio.gather(*tasks)





def get_content_from_resp(resp, url):
    accessible, page_data = resp

    content = page_data.get("content")
    published_date = page_data.get("published_date", None)
    lang = page_data.get("lang")
    obj = {
        "url": url,
        "accessible": accessible,
        "url_content": content if accessible else "",
        "error": "" if accessible else str(content),
        "published_date": published_date,
        "lang": lang
    }

    return obj

@hydra.main(version_base=None, config_path="../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig):

    input_folder = cfg.step1.output_folder
    output_folder = cfg.step2_1.output_folder
    max_facts_per_page = cfg.step2_1.max_facts_per_page
    max_concurrents = cfg.step2_1.max_concurrents
    start_from = cfg.general.start_from

    files = os.listdir(input_folder)
    files = [file for file in files if file.endswith('.json')]
    input_files_full_path = [os.path.join(input_folder, file) for file in files]
    output_files_full_path = [os.path.join(output_folder, file) for file in files]

    for input_file_path, output_file_path in tqdm(zip(input_files_full_path, output_files_full_path), total = len(input_files_full_path)):
        url_content_mapper = {}
        all_urls = set()
        # read input
        input_data = read_json_or_jsonl(input_file_path)
        raw_facts = input_data.get("raw_facts")
        if not raw_facts: continue
        raw_facts = list(sorted(raw_facts, key = lambda x: x["fact"])) # sort based on position


        for fact in raw_facts[:max_facts_per_page]:
            citation_urls = fact.get("citation_urls", [])
            all_urls.update(citation_urls)

        all_urls = list(all_urls)
        responses = asyncio.run(check_urls_in_parallel(all_urls, start_from, max_concurrent=max_concurrents))

        assert len(all_urls) == len(responses)

        for resp, url in zip(responses, all_urls):
            content = get_content_from_resp(resp, url)
            if content: url_content_mapper[url] = content

    
        to_save = {
            "title": input_data.get("title"),
            "wiki_url": input_data.get("wiki_url"),
            "url_content_mapper": url_content_mapper
        }

        write_to_json(data = to_save, filename = output_file_path)


if __name__ == "__main__":
    main()