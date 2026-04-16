# script to fetch cirrusdoc of wikipedia pages and process them. The output would be a folder, with json files, names formatted like "{wiki page title}.json"
# Each file will contain
# {
#     "title": "...", # wiki page title
#     "is_main": "..."
#     "wiki_url": "...", # wiki page url
#     "source": "wikitext...", # raw text of the wiki page
#     "create_timestamp": "...", # creation timestamp of the wiki page
#     "timestamp": "...", # last updated timestamp
#     "topics": ["...", ...], # predicted outlink topics for the wiki page
# }


import json, bz2, os, hydra, re, requests, mwparserfromhell
from datetime import datetime
from omegaconf import DictConfig
from tqdm import tqdm
from typing import Optional, List, Dict, Any
from collections import Counter

from steps.utils.generic import maybe_create_folder, write_to_json

def get_wikipedia_json_batch(titles):
    email = os.environ.get("MY_EMAIL")
    headers = {
        'User-Agent': f'IRB/1.0 ({email})'
    }

    URL = "https://en.wikipedia.org/w/api.php"
    
    # Join titles with the pipe character '|'
    params = {
        "action": "query",
        "format": "json",
        "titles": "|".join(titles),
        "prop": "cirrusdoc"
    }
    
    response = requests.get(URL, params=params, headers=headers)
    response.raise_for_status()
    
    data = response.json()
    pages = data.get('query', {}).get('pages', {})
    
    # Create a mapping of title -> page data
    results = {}
    for page_id, content in pages.items():
        title = content.get('title')
        # If page_id is negative, the page doesn't exist
        results[title] = content if int(page_id) > 0 else None
        
    return results

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

def process_cirrusdoc(cirrusdoc: Dict, is_main: bool = True):
    title = cirrusdoc.get("title")
    source = cirrusdoc.get("source_text")
    create_timestamp = cirrusdoc.get("create_timestamp", "") 
    timestamp = cirrusdoc.get("timestamp") 
    weighted_tags = cirrusdoc.get("weighted_tags")
    popularity_score = cirrusdoc.get("popularity_score", 0)

    url = f"https://en.wikipedia.org/?curid={cirrusdoc.get('page_id')}"
    outlink_topics = get_articletopics_with_scores(weighted_tags)
    topics = [topic['topic'] for topic in outlink_topics]

    res = {
        "title": title, 
        "is_main": is_main,
        "wiki_url": url, 
        "source": source, 
        "create_timestamp": create_timestamp, 
        "timestamp": timestamp, 
        "topics": topics, 
        "popularity_score": popularity_score,
    }

    return res

def get_top_wikilinks_from_list(wikitext_list, top_n=5):
    EXCLUDED_NAMESPACES = ('category:', 'file:', 'image:', 'wikipedia:', 'template:', 'help:', 'draft:')
    SKIP_SECTIONS = {'see also', 'references', 'external links', 'further reading', 'notes'}

    master_counter = Counter()
    
    for wikitext in wikitext_list:
        wikicode = mwparserfromhell.parse(wikitext)
        
        sections_to_keep = []
        for section in wikicode.get_sections(include_lead=True):
            headings = section.filter_headings()
            if headings:
                # Get the title of the section heading
                heading_title = str(headings[0].title).strip().lower()
                if heading_title in SKIP_SECTIONS:
                    continue # Skip this whole section
            sections_to_keep.append(section)
            
        clean_wikicode = mwparserfromhell.parse("".join(str(s) for s in sections_to_keep))
        
        for tag in clean_wikicode.filter_tags():
            if str(tag.tag).lower() in ('ref', 'table', 'gallery'):
                try:
                    clean_wikicode.remove(tag)
                except ValueError:
                    pass

        for template in clean_wikicode.filter_templates():
            try:
                clean_wikicode.remove(template)
            except ValueError:
                pass
                
        links = clean_wikicode.filter_wikilinks()
        
        page_titles = []
        for link in links:
            title = str(link.title).strip()
            
            if title.startswith('#'):
                continue
                
            if title.lower().startswith(EXCLUDED_NAMESPACES):
                continue
                
            if title:
                standardized_title = title[0].upper() + title[1:]
                page_titles.append(standardized_title)
                
        master_counter.update(page_titles)
        
    return master_counter.most_common(top_n)


def get_articles_contents(titles, is_main: bool = True):
    res = {}
    batch_size = 10
    for i in tqdm(range(0, len(titles), batch_size), desc = "Fetching articles"):
        batch_titles = titles[i:i+batch_size]

        resp = get_wikipedia_json_batch(batch_titles)

        for title in batch_titles:
            if not resp.get(title): continue

            cirrusdoc = resp.get(title)["cirrusdoc"]
            if not cirrusdoc: continue

            cirrusdoc = cirrusdoc[0]["source"]
            content = process_cirrusdoc(cirrusdoc, is_main)
            res[title] = content

    return res


@hydra.main(version_base=None, config_path="../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig) -> None:

    # input_folder = cfg.step0.input_folder
    output_folder = cfg.step0.output_folder
    # offset = cfg.step0.offset
    # max_pages = cfg.step0.max_pages
    # target_year = str(cfg.general.target_year)
    target_articles_file = cfg.step0.target_articles_file
    
    if os.path.exists(target_articles_file):
        with open(target_articles_file) as f:
            target_articles = [line.strip() for line in set(f.readlines())]

    # first get the target articles
    target_articles_content = get_articles_contents(titles = target_articles, is_main = True)


    # next get articles that are mentioned in the target articles
    aux_articles = set([])
    main2aux = {title: [] for title in target_articles}
    for title in target_articles:
        if not target_articles_content.get(title): continue

        content = target_articles_content[title]
        source_text = content.get("source")
        mentioned_articles = get_top_wikilinks_from_list([source_text], top_n = 5)

        for aux_title, _ in mentioned_articles:
            aux_articles.add(aux_title)
            main2aux[title].append(aux_title)

    aux_articles = list(aux_articles)

    print(aux_articles)

    # then get the content of aux articles
    aux_articles_content = get_articles_contents(titles = aux_articles, is_main = False)


    for title in target_articles_content:
        to_write = target_articles_content[title]
        to_write["aux_articles"] = main2aux[title]
        outfile = os.path.join(output_folder, f"{title}.json")

        if not os.path.exists(outfile): write_to_json(to_write, outfile)

    for title in aux_articles_content:
        to_write = aux_articles_content[title]
        to_write["aux_articles"] = None
        outfile = os.path.join(output_folder, f"{title}.json")

        if not os.path.exists(outfile): write_to_json(to_write, outfile)


if __name__ == "__main__":
    main()