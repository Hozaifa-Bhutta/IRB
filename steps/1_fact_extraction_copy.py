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
#             "position": [0, 1] 
#         }
#     ]
# }



import json, re, mwparserfromhell, os, hydra
from omegaconf import DictConfig
from argparse import ArgumentParser
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from cleantext import clean
from utils.generic import read_json_or_jsonl, write_to_json
from utils.bad_domains import BAD_DOMAINS



clean_text_func = lambda text: clean(text,
    fix_unicode=True,               # fix various unicode errors
    to_ascii=True,                  # transliterate to closest ASCII representation
    lang="en",                       # set to 'de' for German special handling,
    lower = False
)



def slight_text_processing(wiki_raw_text: str):
    res = wiki_raw_text.replace(",<ref", ".<ref")
    res = res.replace("et al.", "et al")

    return res


def find_all_occurrences_re(main_string, pattern):
    positions = []
    for match in re.finditer(pattern, main_string):
        positions.append(match.start())
    return positions


def ref_tag_count(raw_text: str):
    # count_beginning = raw_text.count("<ref>")
    # count_end = raw_text.count("</ref>")
    # return min(count_beginning, count_end)

    occurrences_ref_start = find_all_occurrences_re(raw_text, "<ref")
    occurrences_ref_end = find_all_occurrences_re(raw_text, "</ref>")

    num_refs = min(len(occurrences_ref_end), len(occurrences_ref_start))

    if num_refs <= 1: return num_refs

    res = 1
    for i in range(1, num_refs):
        current_start= occurrences_ref_start[i]
        previous_end = occurrences_ref_end[i - 1]

        if current_start - previous_end > (5 + len("</ref>")): break
        res += 1

    return res

def ref_at_sentence_start(raw_text: str):
    # check if reference tag is at the start of the sentence
    try:
        return 0 <= raw_text.index("<ref>") < 5
    except ValueError:
        return -1


def remove_html_tags(text: str):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def get_starting_refs(ref_tags, raw_text):
    # print(ref_tags[:5])
    current = 0
    res = []
    for tag in ref_tags:
        pos = raw_text.index(str(tag))
        # print(pos - current, pos, current)
        if 5 > pos - current >= 0: 
            res.append(tag)
            current = pos + len(tag)
        else: break

    return res

def get_info_from_raw_text(raw_text, tag_name_2_url = {}):
    wikicode = mwparserfromhell.parse(raw_text)

    processed_text = wikicode.strip_code()
    prefix = "|Himself (Voice of D"

    debug = raw_text.startswith(prefix)

    # ref_tags = extract_ref_tags(raw_text) #[tag for tag in wikicode.filter_tags(matches=lambda node: node.tag == 'ref')]
    ref_tags = get_starting_refs([tag for tag in wikicode.filter_tags(matches=lambda node: node.tag == 'ref')], raw_text)
    if debug:
        print(ref_tags)
    
    external_urls = [extract_urls(tag, tag_name_2_url, debug) for tag in ref_tags] # basically, just get 
    external_urls = [url for url in external_urls if url]
    # if debug:
    #     print(raw_text)
    #     for tag in ref_tags:
    #         print(tag)
    #         print("-"*70)
    # external_urls = [url for url in external_urls] # if "web.archive.org" not in url]

    return {
        "text": clean_text_func(str(processed_text)),
        "urls": list([str(item) for item in external_urls])
    }

def process_wikilinks_and_replace_ref(raw_text: str):
    wikicode = mwparserfromhell.parse(raw_text)

    # STEP0: replace ref
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


    # STEP1: processing the templates
    templates_to_replace = {}
    for template in wikicode.ifilter_templates():
        template_name = template.name.lower()
        display_text = None


        if template_name.startswith("infobox"):
            display_text = ""
        

        elif template_name.strip() == "ill" and len(template.params) > 0:
            display_text = str(template.params[0].value.strip())

        # Handle citation templates
        elif template_name.startswith("cite") and template.has("title"):
            display_text = ""

        
        # Handle stub, hatnote, or maintenance templates (remove or skip)
        elif template_name in ("stub", "cleanup", "notability") or template_name.startswith(("other ", "main", "disambiguation")):
            display_text = ""  # Remove these templates

        if isinstance(display_text, str):
            try:
                wikicode.replace(template, display_text)
            except ValueError:
                templates_to_replace[str(template)] = display_text

    # STEP2: throw away the headings
    for heading in wikicode.filter_headings():
        wikicode.remove(heading)



    # STEP3: wiki internal link processing. Basically replace them with ordinary text
    for node in wikicode.filter_wikilinks(recursive=True):
        # node.text is the visible part; if not present, use the title
        try:
            visible = str(node.text) if node.text else str(node.title)
            wikicode.replace(node, visible)
        except ValueError:
            templates_to_replace[str(node)] = visible

    str_wikicode = str(wikicode)
    for k, v in templates_to_replace.items():
        str_wikicode = str_wikicode.replace(k, v)

    return str_wikicode, placeholder_mapper, tag_name_2_url


def _extract_urls_from_text(text, debug = False):
    if not text: return None
    url_pattern = r'https?://[\w\-.]+(?:\.[a-z]{2,})+(?:/[\w\-.~:/?#[\]@!$&\'()*+,;=%]*)?'
    urls = re.findall(url_pattern, text)
    if debug:
        print(urls)
    # If more than one URL and one is from web.archive.org, return that one
    if len(urls) > 1:
        for url in urls:
            if 'web.archive.org' in url:
                return url
            
    return urls[0] if urls else None


def extract_urls(tag, tag_name_2_url = {}, debug = False):

    text = str(tag.contents)
    if debug:
        print(text)
    res = _extract_urls_from_text(text, debug)

    if res: return res

    if tag_name_2_url:
        try: 
            tag_name = str(tag.get("name"))
            return tag_name_2_url.get(tag_name)
        except ValueError: return None
    else: return None


def put_back_ref(sentence, placeholder_mapper):
    for k in placeholder_mapper:
        if k in sentence:
            sentence = sentence.replace(k, placeholder_mapper[k])

    return sentence

@hydra.main(version_base=None, config_path="../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg:DictConfig):
    input_folder = cfg.step0.output_folder
    output_folder = cfg.step1.output_folder

    files = os.listdir(input_folder)
    files = [file for file in files if file.endswith('.json')]
    input_files_full_path = [os.path.join(input_folder, file) for file in files]
    output_files_full_path = [os.path.join(output_folder, file) for file in files]

    for input_file_path, output_file_path in tqdm(zip(input_files_full_path, output_files_full_path), total = len(input_files_full_path)):
        file = "Clarence Nash.json"
        if len(input_file_path) < len(file) or input_file_path[-len(file):] != file: continue
        wiki_page_data = read_json_or_jsonl(input_file_path)
        wiki_raw_text = slight_text_processing(wiki_page_data.get("source"))
        wiki_raw_text, placeholder_mapper, tag_name_2_url = process_wikilinks_and_replace_ref(wiki_raw_text)


        raw_facts = []
        wiki_raw_text_sentences = [put_back_ref(sent, placeholder_mapper) for sent in sent_tokenize(wiki_raw_text)]

        wiki_info_sentences = [get_info_from_raw_text(raw, tag_name_2_url) for raw in wiki_raw_text_sentences]


        extracted_sentences = [item["text"] for item in wiki_info_sentences]

        for i in range(1, len(wiki_raw_text_sentences)):

            reference_urls = wiki_info_sentences[i]["urls"]
            reference_urls = [item for item in reference_urls if not any([bad_domain in item for bad_domain in BAD_DOMAINS])]
            if not reference_urls: continue

            raw_facts.append({
                "fact": i-1,
                "citation_urls": reference_urls,
            })

        if raw_facts:
            to_save = {
                "title": wiki_page_data.get("title"),
                "wiki_url": wiki_page_data.get("wiki_url"),
                "extracted_sentences": extracted_sentences,
                "raw_facts": raw_facts
            }
        else: to_save = {}
        write_to_json(data = to_save, filename = output_file_path)


if __name__ == "__main__":
    main()