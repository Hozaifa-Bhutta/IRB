import re
import timeout_decorator
from bs4 import BeautifulSoup, Comment

UNWANTED_CODE_TAGS = [
    "code",
    "embed",
    "iframe",
    "kbd",
    "object",
    "pre",
    "samp",
    "script",
    "style",
    "textarea",
    "tt",
    "var",
    "xmp",
]


# @timeout_decorator.timeout(60, timeout_exception=StopIteration)
def extract_text_from_html(html_src: str) -> list[str]:
    def wrap_text_with_p_tags(soup):
        for text in soup.find_all(string=True, recursive=True):
            if text.parent.name not in [
                "style",
                "script",
                "head",
                "title",
                "meta",
                "[document]",
                "p",
                "a",
                "li",
                "td",
                "th",
            ]:
                if text.parent.find_all(recursive=False, string=False):
                    text.wrap(soup.new_tag("p"))

        return soup

    def traverse(tag, path_parts=None, depth=0, max_depth=40):
        if path_parts is None:
            path_parts = []

        if not hasattr(tag, "children"):
            return []

        if depth >= max_depth:
            text = tag.get_text(separator="\n", strip=True)
            return [text] if text else []

        current_path = path_parts + [tag.name]
        results = []

        for content in tag.contents:
            if isinstance(content, str):
                text = content.strip()
                if text:
                    if tag.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                        level = int(tag.name[1])
                        formatted = "#" * level + " " + text
                    elif tag.name in ["b", "strong"]:
                        formatted = f"**{text}**"
                    elif tag.name == "li":
                        formatted = f"+ {text}"
                    elif tag.name == "a":
                        href = tag.get("href", "")
                        formatted = f"[{text}]({href})"
                    elif tag.name == "br":
                        formatted = "\n"
                    elif tag.name == "hr":
                        formatted = "---"
                    else:
                        formatted = text
                    results.append(formatted)

            elif hasattr(content, "name"):
                child_results = traverse(content, current_path, depth + 1)
                results.extend(child_results)

        return results

    soup = BeautifulSoup(html_src, "lxml")

    nav = soup.find("nav")
    if nav:
        nav.extract()

    footer = soup.find("footer")
    if footer:
        footer.extract()

    for s in soup.select("script"):
        s.extract()

    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    soup = wrap_text_with_p_tags(soup)

    # Remove unwanted tags
    for tag_name in UNWANTED_CODE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    results = traverse(soup.body or soup)
    return results

def clean_markdown(md: str) -> str:
    lines = md.strip().splitlines()
    output = []

    for line in lines:
        line = line.strip()

        # Fix spacing between sections
        if re.match(r"^#+\s+", line) and output and output[-1] != "":
            output.append("")

        # Fix bullet point spacing
        if line.startswith("+") and output and not output[-1].startswith("+"):
            output.append("")

        output.append(line)

    cleaned_md = "\n".join(output)

    # Final pass to normalize multiple newlines
    cleaned_md = re.sub(r'\n{3,}', '\n\n', cleaned_md)

    return cleaned_md.strip()


def extract_markdown_from_html(html_src: str, min_word_number=1):
    # try:
    texts = extract_text_from_html(html_src)
    splitted = [
        paragraph
        for paragraph in texts
        if paragraph and len(paragraph.split(" ")) >= min_word_number
    ]
    markdown = "\n".join(splitted)
    try:
        markdown = clean_markdown(markdown)
    except Exception:
        pass
    return markdown
    # except Exception:
    #     return ""
