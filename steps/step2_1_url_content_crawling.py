# Crawl the content of the citation urls. The input for this script is the output of step 1: fact extraction (1_fact_extraction.py)
# the output of this script will be a json file for each wikipedia page with names formatted like "{wiki page title}.json"
# Each file will contain
# {
#     "title": "...",
#     "wiki_url": "...",
#     "topics": ["...", ...],
#     "create_timestamp": "...",
#     "timestamp": "...",
#     "url_content_mapper": {
#         "url1": {
#             "accessible": "...",
#             "url_content": "...",
#             "title": "...",
#             "error": "...",
#             "published_date":"...",
#             "lang": "...",
#         },
#         "url2": {
#             "accessible": "...",
#             "url_content": "...",
#             "title": "...",
#             "error": "...",
#             "published_date":"...",
#             "lang": "...",
#         },
#     }
# }

import os, requests, time, io, asyncio, hydra
from datetime import datetime
from omegaconf import DictConfig
from tqdm import tqdm
from urllib.parse import urlparse
from cleantext import clean
from fast_langdetect import LangDetectConfig, LangDetector
from typing import Callable, Any
from collections import defaultdict
from fake_useragent import UserAgent

from steps.utils.generic import read_json_or_jsonl, write_to_json
from steps.utils.archive_downloader import getArchiveContent
from steps.utils.html_extraction import extract_text_from_html
from steps.utils.bad_domains import BAD_DOMAINS

request_counters = {}
MAX_REQUESTS_PER_MINUTE = 10  # Maximum requests per minute per domain
# MAX_URLS_PER_WIKI_ARTICLE = 100 # this is a hard cap on the number of URLs we will be collecting for each wikipedia page
LANG_DETECTOR = LangDetector(LangDetectConfig(max_input_length=256))
UA = UserAgent()

class DomainRateLimiter:
    def __init__(self, requests_per_minute):
        self.delay = 60.0 / requests_per_minute
        self.next_allowed_time = defaultdict(float)

    async def wait_for_slot(self, url):
        domain = urlparse(url).netloc.lower()
        now = time.time()
        
        allowed_time = self.next_allowed_time[domain]
        
        wait_time = max(0, allowed_time - now)
        
        self.next_allowed_time[domain] = now + wait_time + self.delay
        
        if wait_time > 0:
            print(f"Pacing ('{domain}')... sleeping {wait_time:.2f}s")
            await asyncio.sleep(wait_time)

def is_valid_date(published_time_str: str, start_from: str) -> bool:
    """Check if the published date is valid based on the start_from date.
    Parameters
    ----------
    published_time_str :str
        The published date string in "YYYY-MM-DD" format.
    start_from : str
        The start_from date string in "YYYY-MM-DD" format.
    Returns
    -------
        bool
            True if the published date is on or after the start_from date, False otherwise.
    """

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

def passes_pre_flight_checks(url: str, published_date: str, start_from: str) -> tuple[bool, str]:
    """
    Checks if a URL should be fetched. 
    Returns a tuple: (should_fetch: bool, reason_if_skipped: str)
    """
    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc:
        return False, f"Invalid URL format: {url}"
    
    domain = parsed_url.netloc.lower()

    if "archive.org" in domain:
        return False, "Archive domains not supported"

    social_domains = ["facebook.com", "twitter.com", "x.com", "github.com", "google.com", "amazon.com", "youtube.com"]
    if domain in BAD_DOMAINS or any(s in domain for s in social_domains):
        return False, "Bad or social media domain"
    
    if "music" in domain:
        return False, "Music-related domain, not useful for fact extraction" 
    
    if "pdf" in url.lower():
        return False, "PDF file in URL not supported"

    if not is_valid_date(published_date, start_from):
        return False, f"Publication not detected or is before start_from: {start_from}"

    return True, ""


def fetch_url_content(url: str, published_date: str) -> tuple[bool, dict]:
    headers = {'User-Agent': UA.random}
    is_accessible = False
    content_dict = {
        "content": None,
        "published_date": published_date,
        "lang": None
    }

    try:
        response = requests.get(url, timeout=20, headers=headers)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '').lower()
        print(f"Response code for {url}: {response.status_code}")

        if "application/pdf" in content_type:
            content_dict["content"] = "PDF files not supported via Content-Type"
            return is_accessible, content_dict
        
        if "text/html" in content_type:
            text = extract_text_from_html(response.content) 
            
            if not text:
                content_dict["content"] = "Empty HTML content"
                return is_accessible, content_dict
            
            is_accessible = True
            content_dict["content"] = text
            content_dict["lang"] = LANG_DETECTOR.detect(text)[0]["lang"]
            return is_accessible, content_dict

        content_dict["content"] = f"Unhandled Content-Type: {content_type}"
        return is_accessible, content_dict

    except requests.exceptions.RequestException as e:
        print(f"Network error accessing URL {url}: {e}")
        content_dict["content"] = f"Network error: {str(e)}"
        return is_accessible, content_dict
    except Exception as e:
        print(f"Unexpected error accessing URL {url}: {e}")
        content_dict["content"] = f"Unexpected error: {str(e)}"
        return is_accessible, content_dict
    


async def is_url_accessible_async(url: str, published_date: str, start_from: str, limiter: DomainRateLimiter, semaphore: asyncio.Semaphore):
    should_fetch, skip_reason = passes_pre_flight_checks(url, published_date, start_from)
    
    if not should_fetch:
        content_dict = {
            "content": skip_reason,
            "published_date": published_date,
            "lang": None
        }
        return False, content_dict

    await limiter.wait_for_slot(url)
    
    loop = asyncio.get_event_loop()
    async with semaphore:
        return await loop.run_in_executor(None, fetch_url_content, url, published_date)
    


async def check_urls_in_parallel(url_list: list[str], url2date: dict[str, str], start_from: str, max_concurrents: int = 5):
    limiter = DomainRateLimiter(requests_per_minute=MAX_REQUESTS_PER_MINUTE)
    semaphore = asyncio.Semaphore(max_concurrents) 

    tasks = [
        is_url_accessible_async(url, url2date[url], start_from, limiter, semaphore) 
        for url in url_list
    ]
    return await asyncio.gather(*tasks)





def get_content_from_resp(resp: tuple[bool, dict], url: str) -> dict:
    """Process the response from URL accessibility check and extract content.
    Parameters
    ----------
        resp : tuple[bool, dict]
            The response tuple from the URL accessibility check.
        url : str
            The original URL that was checked.
    Returns
    -------
        dict
            A dictionary containing the URL, accessibility status, content, error message, and published date.
    """
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
def main(cfg: DictConfig)-> None:

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
        if os.path.exists(output_file_path): continue
        url_content_mapper = {}
        # read input
        input_data = read_json_or_jsonl(input_file_path)
        raw_facts = input_data.get("raw_facts")

        if not raw_facts: continue
        raw_facts = list(sorted(raw_facts, key = lambda x: x["fact"])) # sort based on position

        all_urls = set()
        url2date = {}
        for fact in raw_facts[:max_facts_per_page]:
            citation_urls = fact.get("citation_urls", [])
            published_dates = fact.get("dates", [])

            assert len(citation_urls) == len(published_dates)

            all_urls.update(citation_urls)
            url2date.update({url:date for url, date in zip(citation_urls, published_dates)})

            # if len(all_urls) >= MAX_URLS_PER_WIKI_ARTICLE: break

        all_urls = [url for url in all_urls if url2date.get(url) is not None]
        _start_date = start_from
        responses = asyncio.run(check_urls_in_parallel(all_urls, url2date, _start_date, max_concurrents=max_concurrents))

        assert len(all_urls) == len(responses)

        for resp, url in zip(responses, all_urls):
            content = get_content_from_resp(resp, url)
            if content: url_content_mapper[url] = content

    
        to_save = {
            "title": input_data.get("title"),
            "wiki_url": input_data.get("wiki_url"),
            "topics": input_data.get("topics"),
            "create_timestamp": input_data.get("create_timestamp"),
            "timestamp": input_data.get("timestamp"),
            "popularity_score": input_data.get("popularity_score"),
            "url_content_mapper": url_content_mapper
        }

        write_to_json(data = to_save, filename = output_file_path)


if __name__ == "__main__":
    main()