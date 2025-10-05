# script to extract fact sentences from the processed wikidump (output of step 0)
# the output of this script will be a json file for each wikipedia page with names formatted like "{wiki page title}.json"
# Each file will contain
# {
#     "title": "...",
#     "wiki_url": "...",
#     "extracted_sentences": ["sent1", "sent2"],
#     "raw_facts": [
#         {
#             "fact": "the sentence id",
#             "citation_urls": ["url1", "url2"],
#             "pos": [0, 1]
#         }
#     ]
# }

from typing import Union, Any, Dict, List, Tuple
import json, re, mwparserfromhell, os, hydra
from omegaconf import DictConfig
from argparse import ArgumentParser
from tqdm import tqdm
from cleantext import clean
from nltk.tokenize import sent_tokenize
from utils.generic import read_json_or_jsonl, write_to_json, split_sentence_with_newlines, sentence_filtering
from utils.bad_domains import BAD_DOMAINS


# this function cleans up text 
# clean_text_func = lambda text: clean(text,
#     fix_unicode=True,               # fix various unicode errors
#     to_ascii=True,                  # transliterate to closest ASCII representation
#     lang="en",                       # set to 'de' for German special handling,
#     lower = False
# )
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
    

def process_wikilinks_and_replace_ref(raw_text: str) -> tuple[str, dict[str, str], dict[str, str]]:
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
    placeholder_mapper = {}
    for i, node in enumerate(wikicode.filter_tags(matches=lambda node: node.tag == 'ref')):
        ref_string = str(node)
        to_replace = f"[REF-{i}]"
        placeholder_mapper[to_replace] = ref_string
        wikicode.replace(node, to_replace)

        try:
            tag_name = node.get("name")
            urls = _extract_urls_from_text(str(node.contents))
            if urls:
                tag_name_2_url[str(tag_name)] = urls
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



    # STEP: wiki internal link processing. Basically replace them with ordinary text
    for node in wikicode.filter_wikilinks(recursive=True):
        # node.text is the visible part; if not present, use the title
        try:
            visible = str(node.text) if node.text else str(node.title)
            wikicode.replace(node, visible)
        except ValueError:
            templates_to_replace[str(node)] = visible


    # STEP: remove references section:
    sections = wikicode.get_sections(matches="References")  # returns list of sections with that heading
    for section in sections:
        wikicode.remove(section)

    str_wikicode = str(wikicode)
    for k, v in templates_to_replace.items():
        str_wikicode = str_wikicode.replace(k, v)

    return str_wikicode, placeholder_mapper, tag_name_2_url


def _extract_urls_from_text(text: str) -> str | None:
    """
    The first URL found, with preference given to a 'web.archive.org' URL if multiple are present
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
        for url in urls:
            if 'web.archive.org' in url:
                return url
            
    return urls[0] if urls else None


def extract_urls(tag: mwparserfromhell.nodes.Tag, tag_name_2_url: dict[str, str]) -> str | None:
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
    res = _extract_urls_from_text(text)

    if res: return res

    if tag_name_2_url:
        try: 
            tag_name = str(tag.get("name"))
            return tag_name_2_url.get(tag_name)
        except ValueError: return None
    else: return None


def wikiinfo(cleaned_text: str, pos: list[int], tag_name_2_url: dict[str, str]) -> dict[str, Any]:
    """
    Extracts URLs and their grouped positions from a sentence containing MediaWiki reference tags.
    Parameters
    ----------
    cleaned_text : str
        The cleaned text from the Wikipedia page.
    pos : list[int]
        List of positions for each reference tag in the text.
    tag_name_2_url : dict[str, str]
        A dictionary mapping tag names to their associated URLs.
    Returns
    -------
    dict
        A dictionary containing the cleaned text, list of extracted URLs, and their positions.

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
    
    external_urls = [extract_urls(tag, tag_name_2_url) for tag in ref_tags] 

    # remove all of the non exisitent urls and adjusts positions accordingly
    res_urls = []
    res_pos = []
    prev = None
    count = 0
    for i, url in enumerate(external_urls):
        if url:
            res_urls.append(url)
            if (prev is not None and pos[i] != prev):
                count += 1
            res_pos.append(count)

            prev = pos[i]


    return {
        "text": processed_text,
        "urls": res_urls,
        "pos": res_pos
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


def remove_bad_urls(reference_urls: list[str], pos: list[int]) -> tuple[list[str], list[int]]:
    """
    Removes URLs that are from bad domains or point to PDF files and adjusts positions accordingly.
    Parameters
    ----------
    reference_urls : list[str]
        List of reference URLs.
    pos : list[int]
        List of positions for each reference URL.
    Returns
    -------
    tuple[list[str], list[int]]
        A tuple containing:
        - A list of cleaned URLs (excluding those from bad domains).
        - A list of adjusted positions corresponding to the cleaned URLs.
    """
    cleaned_urls = []
    cleaned_pos = []
    count = 0
    prev = None
    for j, item in enumerate(reference_urls):
        if not any([bad_domain in item for bad_domain in BAD_DOMAINS]) and not is_pdf(item):
            cleaned_urls.append(item)
            if (prev is not None and pos[j] != prev):
                count += 1
            cleaned_pos.append(count)
            prev = pos[j]

    return cleaned_urls, cleaned_pos

@hydra.main(version_base=None, config_path="../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg:DictConfig) -> None:
    input_files_full_path, output_files_full_path = get_file_paths(cfg)


    for input_file_path, output_file_path in tqdm(zip(input_files_full_path, output_files_full_path), total = len(input_files_full_path)):
        # Read and process each wiki page
        wiki_page_data = read_json_or_jsonl(input_file_path)
        wiki_raw_text = slight_text_processing(wiki_page_data.get("source"))
        wiki_raw_text, placeholder_mapper, tag_name_2_url = process_wikilinks_and_replace_ref(wiki_raw_text)
        # placeholder_mapper is of the format {"[REF_I]": url_i}

        # Tokenizes the sentences
        wiki_raw_text_sentences = sent_tokenize(wiki_raw_text)
        # output is each sentence (with placeholder referenes)

        
        # Clean sentences, strips code (keeps placeholders)
        cleaned_sentences = [get_info_from_raw_text(raw) for raw in wiki_raw_text_sentences]
        cleaned_sentences = [sent for sent in cleaned_sentences if sent]
        # output is each sentence cleaned up with reference tags

        # shift references back when needed
        fixed_sentences = shift_tags(cleaned_sentences)
        fixed_sentences = split_sentence_with_newlines(fixed_sentences)

        # get marked facts
        marked_sentences = [fact_marking(sent) for sent in fixed_sentences]

        # gets relative positions of each reference
        positions = [find_pos(sent) for sent in fixed_sentences]

        # replaces the reg tags with original references
        replaced_sentences =  [put_back_ref(sent, placeholder_mapper) for sent in fixed_sentences]

        # put the cleaned text, url, and positions together
        wiki_info_sentences = [wikiinfo(sent, positions[i],tag_name_2_url) for i,sent in enumerate(replaced_sentences)]


        # Extracts the sentences into a list
        extracted_sentences = [item["text"] for item in wiki_info_sentences]

        # Sentence filtering based on length
        good_sentence_indices = set(sentence_filtering(extracted_sentences))
        for i in range(len(extracted_sentences)):
            if i not in good_sentence_indices:
                marked_sentences[i] = ""

        raw_facts = []

        # For each sentence, clean up the reference urls and remove fact if no urls
        # also adjust the positions accordingly
        for i in range(0, len(wiki_info_sentences)):
            
            reference_urls = wiki_info_sentences[i]["urls"]
            pos = wiki_info_sentences[i]["pos"]


            # remove urls in bad domain, and adjust positons accordingly
            cleaned_urls, cleaned_pos = remove_bad_urls(reference_urls, pos)

            if not cleaned_urls: continue

            raw_facts.append({
                "fact": i,
                "citation_urls": cleaned_urls,
                "pos": cleaned_pos
            })

        if raw_facts:
            to_save = {
                "title": wiki_page_data.get("title"),
                "wiki_url": wiki_page_data.get("wiki_url"),
                "topics": wiki_page_data.get("topics"),
                "marked_sentences": marked_sentences,
                "extracted_sentences": extracted_sentences,
                "raw_facts": raw_facts
            }
        else: to_save = {}
        write_to_json(data = to_save, filename = output_file_path)


if __name__ == "__main__":
    main()