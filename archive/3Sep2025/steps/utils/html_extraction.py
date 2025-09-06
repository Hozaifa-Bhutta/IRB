import string, re
from htmldate import find_date
from bs4 import BeautifulSoup
from datetime import datetime
from trafilatura import extract as extract_trafilatura

with open("data/words_alpha.txt") as f:
    WEB2LOWERSET = set()
    for line in f:
        WEB2LOWERSET.add(line.strip().lower())

FEATURE = "html.parser"


def convert_text_to_soup(source: str, feature: str = FEATURE):
    soup = BeautifulSoup(source, feature)

    # Find and remove the header element
    header = soup.find("header")
    if header:
        header.extract()

    # Find and remove the nav element
    nav = soup.find("nav")
    if nav:
        nav.extract()

    # Find and remove the footer element
    footer = soup.find("footer")
    if footer:
        footer.extract()

    for s in soup.select("script"):
        s.extract()

    try:
        res = soup.main
        if res:
            return res
        raise AssertionError
    except Exception as e:
        print("Cannot find <main> element. Use full source")
        return soup
    

def check_chunk(chunk: str, min_chunk_length: int, max_chunk_length: int):
    # return True
    # this is a heuristic function to check if a chunk is good or not

    # firstly, split the chunk into even smaller text segments using newline
    # segments = chunk.split("\n")
    # is_long_segments = all([max_chunk_length >= len(segment.split()) >= min_chunk_length for segment in segments])
    # if not is_long_segments: return False

    if not max_chunk_length >= len(chunk.split()) >= min_chunk_length: return False

    # then, check if they contain mostly english words, let's use a threshold of 50%
    words = [word.strip().strip(string.punctuation) for word in chunk.lower().split()]
    count_english_words = len([word for word in words if word in WEB2LOWERSET])
    is_english = (count_english_words / len(words)) >= 0.5

    if not is_english: return False

    return True
    


def extract_text_from_html_legacy(html_text: str, feature: str = FEATURE, min_chunk_length = 3, max_chunk_length = 500):
    soup = convert_text_to_soup(html_text, feature = feature)

    text = soup.get_text(separator='[SEP]', strip=True)
    text_chunks = [chunk.strip() for chunk in text.split("[SEP]")]

    text_chunks = [chunk for chunk in text_chunks if check_chunk(chunk, min_chunk_length, max_chunk_length)]

    return "\n\n".join(text_chunks)


def extract_text_from_html(html_text: str):
    return extract_trafilatura(
        html_text, 
        target_language="en", 
        favor_precision=True, 
        include_tables = False
    )



def get_publication_date(response):
    try:
        res = find_date(response)
        print(res)
        return res

    except Exception:
        return None