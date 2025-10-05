import string, re
from htmldate import find_date
from bs4 import BeautifulSoup
from datetime import datetime
from trafilatura import extract as extract_trafilatura


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


def extract_text_from_html(html_text: str):
    return extract_trafilatura(
        html_text,
        include_comments=False,
    )



def get_publication_date(response):
    try:
        res = find_date(response)
        return res

    except Exception:
        return None
    