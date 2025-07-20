import string
from bs4 import BeautifulSoup
from english_words import get_english_words_set

WEB2LOWERSET = get_english_words_set(['web2'], lower=True)

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
    # this is a heuristic function to check if a chunk is good or not

    # firstly, split the chunk into even smaller text segments using newline
    segments = chunk.split("\n")
    is_long_segments = all([max_chunk_length >= len(segment.split()) >= min_chunk_length for segment in segments])
    if not is_long_segments: return False

    # then, check if they contain mostly english words, let's use a threshold of 60%
    words = [word.strip().strip(string.punctuation) for word in chunk.lower().split()]
    count_english_words = len([word for word in words if word in WEB2LOWERSET])
    is_english = (count_english_words / len(words)) >= 0.6

    if not is_english: return False

    return True
    


def extract_text_from_html(html_text: str, feature: str = FEATURE, min_chunk_length = 3, max_chunk_length = 500):
    soup = convert_text_to_soup(html_text, feature = feature)

    text = soup.get_text(separator='[SEP]', strip=True)
    text_chunks = [chunk.strip() for chunk in text.split("[SEP]")]

    text_chunks = [chunk for chunk in text_chunks if check_chunk(chunk, min_chunk_length, max_chunk_length)]

    return "\n\n".join(text_chunks)