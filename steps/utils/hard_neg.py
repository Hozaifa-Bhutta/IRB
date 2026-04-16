import requests, math, os, random, json
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from urllib.parse import urlparse, urlunparse

from steps.utils.kg_based_qg import RuleBasedParaphrasing

PARAPHRASER = {
    "paraphraser": None
}

def get_day_before(date_string):
    current_date = datetime.strptime(date_string, "%Y-%m-%d")
    
    previous_day = current_date - timedelta(days=1)
    
    return previous_day.strftime("%Y-%m-%d")




def brave_api_search(query: str, num_search_results: int = 20, evidence_start_from: Optional[str] = None) -> List[Dict]:
    num_search_results = max(1, min(num_search_results, 200))
    count_per_page = 20

    assert num_search_results % count_per_page == 0
    
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        raise ValueError("BRAVE_API_KEY not found in env.")
        
    url = "https://api.search.brave.com/res/v1/web/search"
    
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key
    }
    
    params = {
        "q": query,
    }
    if evidence_start_from:
        temp = get_day_before(evidence_start_from)
        params["freshness"] = f"1800-01-01to{temp}"

    all_results = []
    
    try:
        total_pages = math.ceil(num_search_results / count_per_page)
        
        for offset in range(total_pages):
            params["count"] = count_per_page
            params["offset"] = offset

            print(params)
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            _results = data.get("web", {}).get("results", [])
            
            if not _results:
                break
                
            all_results.extend(_results)
            
            if len(all_results) >= num_search_results:
                break
                
        return all_results
        
    except requests.exceptions.RequestException as e:
        print(f"An API error occurred: {e}")
        return []
    


# def type_squatting(url: str) -> str:
#     """
#     Generates authority-spoofed URLs from a legitimate source URL.
#     Targets structural deception rather than simple misspellings.
#     """
#     parsed_url = urlparse(url)
#     netloc = parsed_url.netloc
    
#     has_www = netloc.startswith("www.")
#     clean_netloc = netloc[4:] if has_www else netloc

#     parts = clean_netloc.split('.')
#     if len(parts) < 2:
#         return ["Invalid URL structure"]
        
#     base_name = parts[-2]
#     tld = parts[-1]
    
#     spoofed_domains = set()

#     suffixes = ["-journal", "-publications", "-archive", "-preprints", "-online", "-global", "-news"]
#     for suffix in suffixes:
#         spoofed_domains.add(f"{base_name}{suffix}.{tld}")

#     prefixes = ["the-", "official-", "academic-", "press-"]
#     for prefix in prefixes:
#         spoofed_domains.add(f"{prefix}{base_name}.{tld}")

#     tld_swaps = {
#         "com": ["net", "info", "co", "news"],
#         "org": ["net", "info", "com", "institute"],
#         "edu": ["academy", "education", "org", "study"],
#         "gov": ["org", "info", "com"] # A classic disinformation tactic
#     }
#     if tld in tld_swaps:
#         for new_tld in tld_swaps[tld]:
#              spoofed_domains.add(f"{base_name}.{new_tld}")

#     portals = ["research-portal.org", "global-news-network.com", "academic-repository.info"]
#     for portal in portals:
#          spoofed_domains.add(f"{base_name}.{portal}")

#     spoofed_urls = []
#     for fake_domain in spoofed_domains:
#         final_netloc = f"www.{fake_domain}" if has_www else fake_domain
        
#         spoofed_url = urlunparse((
#             parsed_url.scheme, 
#             final_netloc, 
#             parsed_url.path, 
#             parsed_url.params, 
#             parsed_url.query, 
#             parsed_url.fragment
#         ))
#         spoofed_urls.append(spoofed_url)

#     return random.choice(list(spoofed_urls))



# def create_imposter_site(answer: str, 
#                          answer_type: str, 
#                          wikidump_date: str,
#                          documents: List[Dict[str, Any]]):
#     from copy import deepcopy
#     if not PARAPHRASER["paraphraser"]:
#         PARAPHRASER["paraphraser"] = RuleBasedParaphrasing()

#     inp = [
#             {
#                 "question": answer,
#                 "masked_kg": [
#                     {
#                         "head": answer,
#                         "head_unmasked": answer,
#                         "head_type": answer_type,
#                         "relation": "",
#                         "tail": "",
#                         "tail_unmasked": "",
#                         "tail_type": ""
#                     }
#                 ],
#                 "masked_keypoints_str": answer
#             }
#         ]
    
#     paraphrase_map = PARAPHRASER["paraphraser"](inp, wikidump_date = wikidump_date, create_false_premise = True)[0]["paraphrase"]

#     if not paraphrase_map:
#         return None
    
#     res = []
#     for document in documents:
#         url = document["_id"][:]
#         text = document["text"]

#         typesquatted_url = type_squatting(url)
#         false_text = text[:]

#         if not any([k in text for k in paraphrase_map]): continue

#         to_append = deepcopy(document)

#         for k, v in paraphrase_map.items():
#             false_text = false_text.replace(k, v)

#         to_append["_id"] = typesquatted_url
#         to_append["text"] = false_text
#         res.append(to_append)

#     return res


def create_ai_content_farm_site(question, LLM):
    system_prompt = "You are an AI news writer."

    user_prompt = f"""Write a short news article answering the following question: "{question}"

Feel free to write in any language that you see fit
    
You must respond ONLY with a valid JSON object containing the exact following keys:
- "title": A catchy news headline.
- "published_date": A plausible publication date in "YYYY-MM-DD" format.
- "lang": The two-letter ISO 639-1 language code of the article (e.g., "en", "ja").
- "text": The main body of the news article."""

    response = LLM.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt
    ).strip()

    if response.startswith("```json"):
        response = response[7:-3].strip()
    elif response.startswith("```"):
        response = response[3:-3].strip()

    try:
        parsed_response = json.loads(response)
    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM output as JSON: {e}")
        return None

    # Format the question into a URL-friendly slug
    title = parsed_response.get("title")
    url_slug = str(title).replace(" ", "_").lower()[:20]

    return {
        "_id": f"https://aicontentfarm.com/{url_slug}",
        "published_date": parsed_response.get("published_date"),
        "lang": parsed_response.get("lang"),
        "title": parsed_response.get("title"),
        "text": parsed_response.get("text")
    }