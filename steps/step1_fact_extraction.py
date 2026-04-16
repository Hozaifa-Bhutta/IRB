# script to extract fact sentences from the processed wikidump (output of step 0)
# the output of this script will be a json file for each wikipedia page with names formatted like "{wiki page title}.json"
# Each file will contain
# {
#     "title": "...",
#     "wiki_url": "...",
#     "topics": ["...", ...],
#     "create_timestamp": "...",
#     "timestamp": "...",
#     "marked_sentences": ["sent1", "sent2"],
#     "extracted_sentences": ["sent1", "sent2"],
#     "raw_facts": [
#         {
#             "fact": "the sentence id",
#             "citation_urls": ["url1", "url2"],
#             "pos": [0, 1],
#             "published_dates": ["", None]         
#         }
#     ]
# }

from typing import Union, Any, Dict, List, Tuple
import json, re, mwparserfromhell, os, hydra, dateutil
from omegaconf import DictConfig
from tqdm import tqdm
from cleantext import clean
from nltk.tokenize import sent_tokenize
from concurrent.futures import ProcessPoolExecutor, as_completed

from steps.utils.generic import read_json_or_jsonl, write_to_json, split_sentence_with_newlines, sentence_filtering
from steps.utils.bad_domains import BAD_DOMAINS


def clean_text_func(text: str) -> str:
    """
    Clean up the input text by fixing Unicode errors, converting to ASCII,
    and doing general text normalization.

    Parameters
    ----------
    text : str
        The text string to be cleaned.

    Returns
    -------
    str
        The cleaned text.
    """
    return clean(
        text,
        fix_unicode=True,  # misencoded characters fixed
        to_ascii=True,     # non-ASCII characters converted to closest ASCII representation
        lang="en",         # english specific cleaning (e.g., standardize quotes/apostrophes)
        lower=False        # keep capitalization of original text
    )


def is_pdf(url: str) -> bool:
    """Checks if a URL points to a PDF document.
    
    Parameters
    ----------
    url : str
        The URL to be checked.
        
    Returns
    -------
    bool
        True if the URL points to a PDF document, False otherwise.

    """
    if not url: return False
    if url.endswith(".pdf") or "/pdf/" in url or "/pdfs/" in url: return True
    return False


def slight_text_processing(wiki_raw_text: str) -> str:
    """Removes period after 'et al.' in the text.

    Parameters
    ----------
    wiki_raw_text : str
        The raw text of the Wikipedia page.

    Returns
    -------
    str
        The processed text with the period removed after all instances of 'et al.'.

    """

    res = wiki_raw_text.replace("et al.", "et al")

    return res

def get_starting_refs(ref_tags: list[str], raw_text: str) -> list[str]:
    """ Returns a list of consecutive reference tags that appear at the start of raw_text.
    Parameters
    ----------
    ref_tags : list
        List of reference tags (e.g., ['[REF-0]', '[REF-1]', ...]).
    raw_text : str
        The raw sentence from the Wikipedia page.
    Returns
    -------
    list
        A list of reference tags that appear at the start of the raw_text.
    """

    current = 0
    res = []
    for tag in ref_tags:
        pos = raw_text.index(str(tag))
        if 5 > pos - current >= 0: 
            res.append(tag)
            current = pos + len(tag)
        else: break

    return res

def get_all_refs(text: str) -> list[str]:
    """Returns a list of all reference tags in the text.
    Parameters
    ----------
    text : str
        The text from which to extract reference tags.
    Returns
    -------
    list
        A list of reference tags found in the text.
    """
    matches = re.findall(r"\[REF-\d+\]", text)
    return matches

def get_info_from_raw_text(raw_text: str) -> str:
    """
    Strips MediaWiki markup from the text and performs general cleaning.
    Parameters
    ----------
    raw_text : str
        The raw text from the Wikipedia page.
    Returns
    -------
    str
        The cleaned text.
    """
    wikicode = mwparserfromhell.parse(raw_text)

    processed_text = wikicode.strip_code()
    cleaned_text = clean_text_func(str(processed_text))


    return cleaned_text


def shift_tags(wiki_info_sentences: list[str]) -> list[str]:
    """
    For each reference that starts the sentence, shift it to the previous sentence
    Parameters
    ----------
    wiki_info_sentences : list
        List of sentences from the Wikipedia page.
    Returns
    -------
    list
        The modified list of sentences with starting reference tags shifted to the previous sentence.
    """
    for i, sent in enumerate(wiki_info_sentences):
        tags = get_all_refs(sent)
        starting_tags = get_starting_refs(tags, sent)
        if starting_tags:
            wiki_info_sentences[i-1] = wiki_info_sentences[i-1][:-1] + " " + " ".join([str(tag) for tag in starting_tags]) + wiki_info_sentences[i-1][-1]
            # now we need to remove the tags from the current one
            for tag in starting_tags:
                wiki_info_sentences[i] = wiki_info_sentences[i].replace(str(tag), "")

    return wiki_info_sentences
    

def process_wikilinks_and_replace_ref(raw_text: str, prefer_webarchive: bool = True) -> tuple[str, dict[str, str], dict[str, str]]:
    """
    Performs a comprehensive cleaning of raw MediaWiki text by removing tables and comments, processing headings, handling templates, replacing reference tags with placeholders, and converting wikilinks to plain text.
    Parameters
    ----------
    raw_text : str
        The raw text from the Wikipedia page.
    Returns
    -------
    tuple
        A tuple containing:
        - The processed text with reference tags replaced by placeholders.
        - A dictionary mapping placeholders to their original reference tags.
        - A dictionary mapping reference tag names to their associated URLs.
    """
    wikicode = mwparserfromhell.parse(raw_text)

    # STEP: remove comments
    for comment in wikicode.filter_comments():
        try:
            wikicode.remove(comment)
        except Exception as e:
            continue

    # STEP: remove tables:
    for table in wikicode.filter_tags(matches=lambda node: node.tag == "table"):
        try:
            wikicode.remove(table)
        except ValueError: continue

    # STEP: replace ref
    tag_name_2_url = {}
    tag_name_2_date = {}
    placeholder_mapper = {}
    for i, node in enumerate(wikicode.filter_tags(matches=lambda node: node.tag == 'ref')):
        ref_string = str(node)
        to_replace = f"[REF-{i}]"
        placeholder_mapper[to_replace] = ref_string
        wikicode.replace(node, to_replace)

        try:
            tag_name = node.get("name")
            urls = _extract_urls_from_text(str(node.contents), prefer_webarchive)
            published_date = _extract_dates_from_text(str(node.contents))
            if urls:
                tag_name_2_url[str(tag_name)] = urls
                tag_name_2_date[str(tag_name)] = published_date
        except ValueError:
            pass


    # STEP: processing the templates
    templates_to_replace = {}
    for template in wikicode.ifilter_templates():
        template_name = template.name.lower()
        display_text = None

        if not any([desirable in template_name for desirable in ["cite"]]):
            display_text = ""

        if isinstance(display_text, str):
            try:
                wikicode.replace(template, display_text)
            except ValueError:
                templates_to_replace[str(template)] = display_text

    # STEP: process the headings
    path_list = []
    for heading in wikicode.filter_headings():
        # Get the heading's level (e.g., ==Title== is level 2)
        level = heading.level
        title = heading.title.strip()

        path_list = path_list[:level - 2]

        # Now, append the current heading's title
        path_list.append(title)

        # Print the full, correct path
        to_replace = "SECTION: " + " > ".join(path_list)

        wikicode.replace(heading, to_replace)


    # STEP: wiki internal link processing. Replace them with placeholders
    wikilink_mapper = {}
    for i, node in enumerate(wikicode.filter_wikilinks(recursive=True)):
        try:
            visible = str(node.text) if node.text else str(node.title)
            title = str(node.title).strip()
            placeholder = f"[WIKI-{i}]"
            wikilink_mapper[placeholder] = {"visible": visible, "title": title}
            wikicode.replace(node, placeholder)
        except ValueError:
            templates_to_replace[str(node)] = str(node.text) if node.text else str(node.title)

    # STEP: remove references section:
    sections = wikicode.get_sections(matches="References") 
    for section in sections:
        wikicode.remove(section)

    str_wikicode = str(wikicode)
    for k, v in templates_to_replace.items():
        str_wikicode = str_wikicode.replace(k, v)

    return str_wikicode, placeholder_mapper, tag_name_2_url, tag_name_2_date, wikilink_mapper


def _extract_urls_from_text(text: str, prefer_webarchive: bool = True) -> str | None:
    """
    The first URL found
    Parameters
    ----------
    text : str
        The text from which to extract URLs.
    Returns
    -------
    str | None
        The first URL found in the text, or None if no URLs are found.
    """
    if not text: return None
    url_pattern = r'https?://[\w\-.]+(?:\.[a-z]{2,})+(?:/[\w\-.~:/?#[\]@!$&\'()*+,;=%]*)?'
    urls = re.findall(url_pattern, text)
    # If more than one URL and one is from web.archive.org, return that one
    if len(urls) > 1:
        if prefer_webarchive:
            for url in urls:
                if 'web.archive.org' in url:
                    return url
        else:
            for url in urls:
                if "web.archive.org" in url:
                    continue
                return url
                
    return urls[0] if urls else None


def _extract_dates_from_text(text: str) -> str:
    """Extract the date inside a paratheses. Input will look like so:
    + <ref>{{Cite web|url=https://www.latimes.com/socal/burbank-leader/entertainment/tn-blr-me-community-20180604-story.html|title=Community: New exhibit piece at Burbank museum is a real knockout|last=Rudolph|first=Joyce|website=[[Los Angeles Times]]|date=June 4, 2018|access-date=2019-05-05}}</ref>
    """

    pattern = r'(?:\||\{\{)\s*date\s*=\s*([^|}]+)'
    
    match = re.search(pattern, text, re.IGNORECASE)
    
    if not match:
        return None
        
    raw_date_str = match.group(1).strip()
    
    if not raw_date_str:
        return None
        
    try:
        parsed_date = dateutil.parser.parse(raw_date_str)
        return parsed_date.strftime('%Y-%m-%d')
    except (ValueError, TypeError, OverflowError):
        return None


def extract_urls(tag: mwparserfromhell.nodes.Tag, tag_name_2_url: dict[str, str], prefer_webarchive: bool = True) -> str | None:
    """
    Extracts URLs from a MediaWiki tag.
    Parameters
    ----------
    tag : mwparserfromhell.nodes.Tag
        The MediaWiki tag from which to extract URLs.
    tag_name_2_url : dict[str, str]
        A dictionary mapping tag names to their associated URLs.
    Returns
    -------
    str | None
        The extracted URL, or None if no URL is found.
    """
    text = str(tag.contents)
    res = _extract_urls_from_text(text, prefer_webarchive)

    if res: return res

    if tag_name_2_url:
        try: 
            tag_name = str(tag.get("name"))
            return tag_name_2_url.get(tag_name)
        except ValueError: return None
    else: return None

def extract_date(tag: mwparserfromhell.nodes.Tag, tag_name_2_date: dict[str, str]) -> str | None:
    text = str(tag.contents)
    res = _extract_dates_from_text(text)

    if res: return res

    if tag_name_2_date:
        try:
            tag_name = str(tag.get("name"))
            return tag_name_2_date.get(tag_name)
        except ValueError: return None

    else: return None


def restore_wikilinks(sentence: str, wikilink_mapper: dict[str, dict[str, str]]) -> tuple[str, list[str]]:
    """
    Restores visible text for wikilinks and extracts the mentioned Wikipedia articles.
    """
    mentioned = []
    for placeholder, data in wikilink_mapper.items():
        if placeholder in sentence:
            sentence = sentence.replace(placeholder, data["visible"])
            mentioned.append(data["title"])
    return sentence, mentioned

def wikiinfo(cleaned_text: str, pos: list[int], tag_name_2_url: dict[str, str], tag_name_2_date: dict[str, str], prefer_webarchive: bool = True) -> dict[str, Any]:
    """
    Extracts URLs and their grouped positions from a sentence containing MediaWiki reference tags.
    Parameters
    ----------
    cleaned_text : str
        The cleaned text from the Wikipedia page.
    pos : list[int]
        List of positions for each reference tag in the text.
    tag_name_2_url : dict[str, str]
        A dictionary mapping tag names to their associated URL.
    tag_name_2_dict: dict[str, str]
        A dictionary mapping tag names to their published date
    Returns
    -------
    dict
        A dictionary containing the cleaned text, list of extracted URL, and their positions.

    """
    # gets all urls from text and strips code
    wikicode = mwparserfromhell.parse(cleaned_text)
    ref_tags = [tag for tag in wikicode.filter_tags(matches=lambda node: node.tag == 'ref')]

    processed_text = wikicode
    for tag in wikicode.ifilter_tags(matches='ref'):
        try:
            processed_text.replace(tag, "")
        except ValueError as e: continue
    processed_text = processed_text.strip_code()
    
    external_urls = [extract_urls(tag, tag_name_2_url, prefer_webarchive) for tag in ref_tags] 
    published_dates = [extract_date(tag, tag_name_2_date) for tag in ref_tags]

    assert len(external_urls) == len(published_dates)

    # remove all of the non exisitent urls and adjusts positions accordingly
    res_urls = []
    res_pos = []
    res_dates = []
    prev = None
    count = 0
    for i, (url, date) in enumerate(zip(external_urls, published_dates)):
        if url:
            res_urls.append(url)
            res_dates.append(date)
            if (prev is not None and pos[i] != prev):
                count += 1
            res_pos.append(count)

            prev = pos[i]


    return {
        "text": processed_text,
        "urls": res_urls,
        "pos": res_pos,
        "dates": res_dates
    }

def find_pos(raw_text: str) -> list[int]:
    """
    Returns a list of positions for each reference tag in the text. First tag starts off at index 0 and nearby tags are assigned the same index.
    Parameters
    ----------
    raw_text : str
        The raw text from which to extract reference tag positions.
    Returns
    -------
    list[int]
        A list of positions for each reference tag in the text.
    """
    prev_pos = float('-inf')
    cur_ind = -1
    res = []
    for tag in get_all_refs(raw_text):
        pos = raw_text.index(str(tag))
        if abs(pos - prev_pos) >= 5: # distance threshold is 5
            cur_ind += 1
        res.append(cur_ind)
        prev_pos = pos + len(tag)

    return res


def fact_marking(raw_text: str) -> str:
    """
    Marks the fact sentences by appending [KP] after each fact.
    Parameters
    ----------
    raw_text : str
        The raw text from the Wikipedia page.
    Returns 
    -------
    str
        The marked text with [KP] appended to each fact.
    """
    prev_pos = float('-inf')
    cur_ind = 0

    res = ""

    for tag in get_all_refs(raw_text):
        pos = raw_text.index(str(tag))
        if abs(pos - prev_pos) >= 5: # distance threshold is 5
            _keypoint = re.sub(r"\[REF-\d+\]", "", raw_text[cur_ind: pos + len(tag)]).strip() + " [KP] "
            cur_ind = pos + len(tag)
            res = res + _keypoint
        prev_pos = pos + len(tag)

    return res


def put_back_ref(sentence: str, placeholder_mapper: dict[str, str]) -> str:
    """
    Puts back the references in the sentence using the placeholder mapper.
    Parameters
    ----------
    sentence : str
        The sentence with placeholders.
    placeholder_mapper : dict[str, str]
        A dictionary mapping placeholders to their original reference tags.
    Returns
    -------
    str
        The sentence with original reference tags put back in place.
    """
    for k in placeholder_mapper:
        if k in sentence:
            sentence = sentence.replace(k, placeholder_mapper[k])

    return sentence



def get_file_paths(cfg:DictConfig) -> tuple[list[str], list[str]]:
    """Generates lists of full input and output file paths based on the configuration.
    Parameters
    ----------
    cfg : DictConfig
        The configuration object containing input and output folder paths.
    Returns
    -------
    tuple[list[str], list[str]]
        A tuple containing:
        - A list of input file paths.
        - A list of output file paths.
    """
    input_folder = cfg.step0.output_folder
    output_folder = cfg.step1.output_folder

    files = os.listdir(input_folder)
    files = [file for file in files if file.endswith('.json')]
    input_files_full_path = [os.path.join(input_folder, file) for file in files]
    output_files_full_path = [os.path.join(output_folder, file) for file in files]

    return input_files_full_path, output_files_full_path


def remove_bad_urls(reference_urls: list[str], pos: list[int], dates: list[str]) -> tuple[list[str], list[int], list[str]]:
    """
    Removes URLs that are from bad domains or point to PDF files and adjusts positions accordingly.
    Parameters
    ----------
    reference_urls : list[str]
        List of reference URLs.
    pos : list[int]
        List of positions for each reference URL.
    dates: list[str]
        List of dates for each reference
    Returns
    -------
    tuple[list[str], list[int]]
        A tuple containing:
        - A list of cleaned URLs (excluding those from bad domains).
        - A list of adjusted positions corresponding to the cleaned URLs.
    """
    cleaned_urls = []
    cleaned_pos = []
    cleaned_dates = []
    count = 0
    prev = None
    for j, (item, date) in enumerate(zip(reference_urls, dates)):
        if not any([bad_domain in item for bad_domain in BAD_DOMAINS]) and not is_pdf(item):
            cleaned_urls.append(item)
            cleaned_dates.append(date)
            if (prev is not None and pos[j] != prev):
                count += 1
            cleaned_pos.append(count)
            prev = pos[j]

    return cleaned_urls, cleaned_pos, cleaned_dates

def process_file(input_file_path: str, output_file_path: str, prefer_webarchive: bool):
    """
    Worker function to process a single Wikipedia file.
    Must be defined at the module level (outside main) for multiprocessing to work.
    """
    if os.path.exists(output_file_path):
        return True, None

    try:
        wiki_page_data = read_json_or_jsonl(input_file_path)
        wiki_raw_text = slight_text_processing(wiki_page_data.get("source"))
        wiki_raw_text, placeholder_mapper, tag_name_2_url, tag_name_2_date, wikilink_mapper = process_wikilinks_and_replace_ref(wiki_raw_text, prefer_webarchive)
        
        wiki_raw_text_sentences = sent_tokenize(wiki_raw_text)
        
        cleaned_sentences = [get_info_from_raw_text(raw) for raw in wiki_raw_text_sentences]
        cleaned_sentences = [sent for sent in cleaned_sentences if sent]

        fixed_sentences = shift_tags(cleaned_sentences)
        fixed_sentences = split_sentence_with_newlines(fixed_sentences)

        restored_fixed_sentences = []
        mentioned_articles_per_sent = []
        for sent in fixed_sentences:
            cleaned_sent, mentioned = restore_wikilinks(sent, wikilink_mapper)
            restored_fixed_sentences.append(cleaned_sent)
            mentioned_articles_per_sent.append(list(set(mentioned))) 

        marked_sentences = [fact_marking(sent) for sent in restored_fixed_sentences]

        positions = [find_pos(sent) for sent in restored_fixed_sentences]

        replaced_sentences = [put_back_ref(sent, placeholder_mapper) for sent in restored_fixed_sentences]

        wiki_info_sentences = [wikiinfo(sent, positions[i], tag_name_2_url, tag_name_2_date, prefer_webarchive) for i,sent in enumerate(replaced_sentences)]
        
        extracted_sentences = [item["text"] for item in wiki_info_sentences]

        good_sentence_indices = set(sentence_filtering(extracted_sentences))
        for i in range(len(extracted_sentences)):
            if i not in good_sentence_indices:
                marked_sentences[i] = ""

        raw_facts = []

        for i in range(0, len(wiki_info_sentences)):
            reference_urls = wiki_info_sentences[i]["urls"]
            pos = wiki_info_sentences[i]["pos"]
            dates = wiki_info_sentences[i]["dates"]

            cleaned_urls, cleaned_pos, cleaned_dates = remove_bad_urls(reference_urls, pos, dates)

            assert len(cleaned_urls) == len(cleaned_pos) == len(cleaned_dates)

            if not cleaned_urls: continue

            raw_facts.append({
                "fact": i,
                "citation_urls": cleaned_urls,
                "pos": cleaned_pos,
                "dates": cleaned_dates,
                "mentioned_articles": mentioned_articles_per_sent[i]
            })

        if raw_facts:
            to_save = {
                "title": wiki_page_data.get("title"),
                "wiki_url": wiki_page_data.get("wiki_url"),
                "topics": wiki_page_data.get("topics"),
                "create_timestamp": wiki_page_data.get("create_timestamp"),
                "timestamp": wiki_page_data.get("timestamp"),
                "popularity_score": wiki_page_data.get("popularity_score"),
                "marked_sentences": marked_sentences,
                "extracted_sentences": extracted_sentences,
                "raw_facts": raw_facts
            }
        else: 
            to_save = {}
            
        write_to_json(data=to_save, filename=output_file_path)
        
        return True, None

    except Exception as e:
        return False, input_file_path


@hydra.main(version_base=None, config_path="../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig) -> None:
    input_files_full_path, output_files_full_path = get_file_paths(cfg)
    prefer_webarchive = cfg.step1.prefer_webarchive

    max_workers = 16
    
    error_counter = 0

    print(f"Starting parallel processing with {max_workers or os.cpu_count()} workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_file, inp, out, prefer_webarchive)
            for inp, out in zip(input_files_full_path, output_files_full_path)
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            success, problematic_file = future.result()
            
            if not success:
                error_counter += 1
                print(f"Problematic file: {problematic_file}. # Errors: {error_counter}")


if __name__ == "__main__":
    main()