# script to extract fact sentences from the processed opinions (output of step 0)
# the output of this script will be a json file for each opinion  with names formatted like "{opinion_id}.json"
# Each file will contain
# {
#     "title": "...",
#     "extracted_sentences": ["sent1", "sent2"],
#     "raw_facts": [
#         {
#             "fact": "the sentence id",
#             "citation_triplet": [(<volume>, <reporter>, <first_page>), (volume2, reporter2, first_page2)],
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
from steps.utils.generic import read_json_or_jsonl, write_to_json, split_sentence_with_newlines, sentence_filtering
from steps.utils.bad_domains import BAD_DOMAINS


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

def extract_citation_info(citation: str) -> tuple[str, str, str]:
    """
    Extracts volume number, reporter, and first page from a citation string.
    Parameters
    ----------
    citation : str
        The citation string containing the legal reference.
    Returns
    -------
    tuple
        A tuple containing volume number, reporter, and first page.
    """
    # Regex to match volume number, reporter, and first page
    match = re.search(r"<em>.*?</em>\s*(\d+)\s+([a-zA-Z0-9\.]+)\s+(\d+)", citation, re.DOTALL)
    if match:
        volume = match.group(1)
        reporter = match.group(2)
        first_page = match.group(3)
        return volume, reporter, first_page
    else:
        return "", "", ""
def process_citations_and_replace_ref(raw_text: str) -> tuple[str, dict[str, str], dict[str, str], dict[str, str]]:
    """
    Performs a comprehensive cleaning of raw source text by removing most tags replacing <em> tags with placeholders
    Parameters
    ----------
    raw_text : str
        The raw text from the Wikipedia page.
    Returns
    -------
    tuple
        A tuple containing:
        - The processed text with citation tags replaced by placeholders.
        - A dictionary mapping placeholders to their original citation content (e.g., [REF-0] -> "<em>Cook v. Boorstin,</em> 763 F.2d 1462").
        - A dictionary mapping citation tag names to their triplet (e.g., "Cook v. Boorstin" -> volume, reporter, first page).
        - A dictionary mapping placeholders to their case name (e.g., [REF-0] -> "Cook v. Boorstin,")
    """
    # citation is defined as <em>...</em> followed by a number (volume num) a short string (reporter) and a number (first page)
    # example:  "<em>\n   Cook v. Boorstin,\n  </em>\n  763 F.2d 1462 "
    # not a valid example: "<em>\n   Nuesse\n  </em>\n  even addressed the issue of standing."

   # normalize spaces/newlines inside <em> tags and around citation
    normalized_text = re.sub(r"\s+", " ", raw_text)

    # regex matches <em>...</em> followed by volume, reporter, first page
    citation_regex = re.compile(
    r"<em>\s*([^\n<>]+?)\s*</em>\s*"  # case name inside <em>, no tags or newlines inside (GROUP 1)
    r"(\d+)\s*"                        # volume number (GROUP 2)
    r"([A-Za-z0-9\.]+)\s*"             # reporter (single word, no spaces) (GROUP 3)
    r"(\d+)"                            # first page number (GROUP 4)
    , re.DOTALL
    )



    placeholder_mapper = {}
    citation_triplet_mapper = {}
    placeholder_to_case_name = {} # <-- NEW: Store case names

    # counter for placeholder index
    placeholder_index = 0

    def replace_match(m):
        nonlocal placeholder_index
        citation = m.group(0) # The full match
        case_name = m.group(1).strip() # <-- NEW: Get case name from group 1
        
        # Original logic
        volume, reporter, first_page = extract_citation_info(citation)
        placeholder = f"[REF-{placeholder_index}]"
        placeholder_mapper[placeholder] = citation
        citation_triplet_mapper[citation] = (volume, reporter, first_page)
        
        # <-- NEW: Store the mapping from placeholder to case name
        placeholder_to_case_name[placeholder] = case_name
        
        placeholder_index += 1
        return placeholder

    processed_text = citation_regex.sub(replace_match, normalized_text)

    # now remove all other tags, extra spaces, etc.
    # code = mwparserfromhell.parse(processed_text)
    # processed_text = code.strip_code()
    # processed_text = clean_text_func(str(processed_text))
    processed_text = get_info_from_raw_text(processed_text)

    # remove all remaining tags but keep placeholders
    processed_text = re.sub(r"<[^>]+>", "", processed_text)

    # collapse multiple spaces/newlines into one space
    processed_text = re.sub(r"\s+", " ", processed_text).strip()

    return processed_text, placeholder_mapper, citation_triplet_mapper, placeholder_to_case_name


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
    
    if not re.search(r"\[REF-\d+\]", raw_text): return ""
    return re.sub(r"\[REF-\d+\]", " [KP] ", raw_text)

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

def get_sentence_info(
    sentence_with_placeholders: str, 
    pos: list[int], 
    placeholder_mapper: dict[str, str], 
    citation_triplet_mapper: dict[str, tuple[str, str, str]],
    placeholder_to_case_name: dict[str, str] # <-- NEW: Pass in the case name mapper
) -> dict[str, Any]:
    """
    Extracts clean text, citation triplets, and positions from a sentence.
    This replaces the old `wikiinfo` function.

    Parameters
    ----------
    sentence_with_placeholders : str
        The sentence with [REF-...] placeholders.
    pos : list[int]
        List of grouped positions for citations (matches the placeholders).
    placeholder_mapper : dict
        Maps placeholders [REF-X] to full <em> citation strings.
    citation_triplet_mapper : dict
        Maps full <em> citation strings to (vol, rep, page) triplets.
    placeholder_to_case_name : dict
        Maps placeholders [REF-X] to the case name string (e.g., "Cook v. Boorstin,").

    Returns
    -------
    dict
        A dictionary containing:
        - "text": The cleaned sentence text (with placeholders replaced by case names).
        - "citations": A list of (vol, rep, page) triplets.
        - "pos": The filtered list of positions (matching valid citations).
    """
    
    # 1. Get the final clean text
    # <-- MODIFIED: Use a lambda function with re.sub to replace each placeholder
    # with its corresponding case name. Use .get() for safety, defaulting to ""
    # if a placeholder is somehow not in the map (shouldn't happen).
    def replace_func(match):
        placeholder = match.group(0)
        return placeholder_to_case_name.get(placeholder, "")

    cleaned_text = re.sub(r"\[REF-\d+\]", replace_func, sentence_with_placeholders)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    
    # 2. Find all placeholders
    placeholders = get_all_refs(sentence_with_placeholders) # e.g., ["[REF-0]", "[REF-1]"]

    valid_triplets = []
    valid_pos = []
    
    # 3. Iterate over placeholders and positions *simultaneously*
    # (This logic remains unchanged as it correctly builds the raw_facts)
    for placeholder, position in zip(placeholders, pos):
        if placeholder in placeholder_mapper:
            citation_key = placeholder_mapper[placeholder]
            if citation_key in citation_triplet_mapper:
                triplet = citation_triplet_mapper[citation_key]
                # 4. Filter out empty triplets (e.g., from extract_citation_info failing)
                if all(triplet): # (vol, rep, page) must all be non-empty
                    valid_triplets.append(triplet)
                    valid_pos.append(position)

    return {
        "text": cleaned_text, # This text now contains case names
        "citations": valid_triplets, # This will be used for raw_facts
        "pos": valid_pos
    }

@hydra.main(version_base=None, config_path="../../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg:DictConfig) -> None:
    input_files_full_path, output_files_full_path = get_file_paths(cfg)

    count = 0
    for input_file_path, output_file_path in tqdm(zip(input_files_full_path, output_files_full_path), total = len(input_files_full_path)):
        # Read and process each wiki page
        if count == 50: break
        page_data = read_json_or_jsonl(input_file_path)

        raw_text = page_data.get("source")
        # <-- MODIFIED: Unpack the new placeholder_to_case_name dictionary
        processed_text, placeholder_mapper, tag_name_2_triplet, placeholder_to_case_name = process_citations_and_replace_ref(raw_text)


        # placeholder_mapper is of the format {"[REF-I]": "<em>...</em> 123 ABC 456"}
        # Tokenizes the sentences
        processed_text_sentences = sent_tokenize(processed_text)
        # output is each sentence (with placeholder referenes)

        # get marked facts
        marked_sentences = [fact_marking(sent) for sent in processed_text_sentences]

        # gets relative positions of each reference
        positions = [find_pos(sent) for sent in processed_text_sentences]

        # replaces the reg tags with original references
        replaced_sentences =  [put_back_ref(sent, placeholder_mapper) for sent in processed_text_sentences]

        # put the cleaned text, url, and positions together
        wiki_info_sentences = [
            get_sentence_info(
                sent, 
                positions[i], 
                placeholder_mapper, 
                tag_name_2_triplet,
                placeholder_to_case_name # <-- MODIFIED: Pass the new dict here
            ) 
            for i, sent in enumerate(processed_text_sentences)
        ]

       # Extracts the sentences into a list
       # This list will now contain sentences with case names instead of empty strings
        extracted_sentences = [item["text"] for item in wiki_info_sentences]

        # Sentence filtering based on length
        good_sentence_indices = set(sentence_filtering(extracted_sentences))
        for i in range(len(extracted_sentences)):
            if i not in good_sentence_indices:
                marked_sentences[i] = ""



        raw_facts = []

        # For each sentence, clean up the reference urls and remove fact if no urls
        # also adjust the positions accordingly
        # (This logic is unchanged and correct)
        for i in range(0, len(wiki_info_sentences)):
            
            # Skip sentences that were filtered out
            if i not in good_sentence_indices:
                continue

            sentence_citations = wiki_info_sentences[i]["citations"]
            pos = wiki_info_sentences[i]["pos"]

            # The 'get_sentence_info' function already filtered for valid, non-empty triplets
            # We just need to check if any remain.
            if not sentence_citations: 
                continue

            raw_facts.append({
                "fact": i,
                "citation_triplets": sentence_citations, # Using name from spec, but value is list of triplets
                "pos": pos
            })

        if raw_facts:
            to_save = {
                "title": page_data.get("title"),
                "create_timestamp": page_data.get("create_timestamp"),
                "timestamp": page_data.get("timestamp"),
                # Fixed NameError: wiki_page_data -> page_data
                "extracted_sentences": extracted_sentences, # This list is now populated as requested
                "marked_sentences": marked_sentences,
                "raw_facts": raw_facts
            }
        else: to_save = {}
        write_to_json(data = to_save, filename = output_file_path)
        count += 1


if __name__ == "__main__":
    main()