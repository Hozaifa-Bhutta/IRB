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
    



import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup

JS_SIGNATURES = [
    r'__NEXT_DATA__', r'__NUXT_DATA__', r'data-reactroot', r'data-hydrate',
    r'id="__next"', r'id="__nuxt"', r'id="root"', r'id="app"', r'ng-version',
    r'ember-view', r'data-server-rendered'
]
PAYWALL_KEYWORDS = [
    'subscribe', 'subscriber-only', 'for subscribers', 'log in to continue',
    "you've reached your limit", 'membership required', 'paywall', 'metered'
]
PAYWALL_VENDORS = [
    'tinypass.com', 'tpwidget', 'piano.io', 'zephr', 'cxense', 'chartbeat_paywall',
    'permutive', 'arc/subs', 'newswall', 'meter.js'
]

def text_density_metrics(html: str):
    soup = BeautifulSoup(html, 'lxml')

    # visible text (rough but fast)
    for tag in soup(['script', 'style', 'noscript']):
        tag.extract()
    text = ' '.join(t.strip() for t in soup.stripped_strings)
    words = len(text.split())

    # bytes in scripts vs total
    scripts = soup.find_all('script')
    script_bytes = sum(len(s.get_text() or '') for s in scripts)
    script_srcs = ' '.join(s.get('src') or '' for s in scripts)
    total_bytes = len(html)

    # mount-node style body?
    body = soup.body or soup
    children = [c for c in body.children if getattr(c, 'name', None)]
    single_mount = (len(children) <= 2) and any(
        (c.get('id') or '').lower() in {'root', 'app', '__next', '__nuxt'} for c in children
    )

    # noscript “enable javascript”
    nos = (soup.find('noscript') or {}).get_text(strip=True) if soup.find('noscript') else ''
    needs_js = 'enable javascript' in nos.lower()

    return {
        'words': words,
        'script_ratio': (script_bytes + 1) / (total_bytes + 1),
        'script_srcs': script_srcs.lower(),
        'single_mount': single_mount,
        'needs_js': needs_js,
        'text': text.lower()
    }

def looks_js_rendered(html: str) -> bool:
    m = text_density_metrics(html)
    # Heuristics: very low words and high script ratio, or strong fingerprints
    js_fingerprints = any(re.search(sig, html, re.I) for sig in JS_SIGNATURES)
    low_text_high_script = (m['words'] < 200 and m['script_ratio'] > 0.20)
    return js_fingerprints or m['single_mount'] or m['needs_js'] or low_text_high_script

def looks_paywalled(html: str, final_url: str, redirects: list[str]) -> bool:
    m = text_density_metrics(html)
    url_parts = [final_url] + (redirects or [])
    lower_urls = ' '.join(url_parts).lower()

    # URL patterns that often indicate gating
    gated_url = any(pat in lower_urls for pat in [
        '/subscribe', '/paywall', '/meter', '/membership', '/login', '/account'
    ])

    # Vendor scripts
    vendor_hit = any(v in m['script_srcs'] for v in PAYWALL_VENDORS)

    # On-page language or CSS classes
    keyword_hit = any(k in m['text'] for k in PAYWALL_KEYWORDS) or \
                  ('class="paywall"' in html.lower() or "class='paywall'" in html.lower())

    # Almost no article text but lots of markup can indicate a hard paywall
    starved_text = m['words'] < 150

    return gated_url or vendor_hit or (keyword_hit and starved_text)

def classify_fetch(resp, html: str) -> dict:
    """
    resp: the requests.Response object
    html: resp.text
    Returns a label and reason for filtering.
    """
    # quick status checks
    if resp.status_code in (401, 402, 403, 429, 451):
        return {'ok': False, 'label': 'blocked', 'reason': f'HTTP {resp.status_code}'}

    # redirect destinations (requests stores history)
    redirects = [r.headers.get('Location', '') for r in resp.history if r.is_redirect]
    final_url = str(resp.url)

    if looks_paywalled(html, final_url, redirects):
        return {'ok': False, 'label': 'paywalled', 'reason': 'paywall heuristics matched'}

    if looks_js_rendered(html):
        return {'ok': False, 'label': 'js_rendered', 'reason': 'client-side rendering suspected'}

    # Very small HTML, probably shell/placeholder
    if len(html) < 8000 and len(BeautifulSoup(html, 'lxml').find_all('p')) < 3:
        return {'ok': False, 'label': 'thin_content', 'reason': 'too little visible text'}

    return {'ok': True, 'label': 'good', 'reason': 'sufficient server-rendered text'}