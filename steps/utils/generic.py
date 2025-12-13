import json, os, spacy, string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from typing import List

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' model. Please wait...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")



def write_to_jsonl(data, filename):
    with open(filename, "a") as f:
        for line in data:
            json.dump(line, f)
            f.write("\n")


def write_to_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def read_json_or_jsonl(filename):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except Exception:
        with open(filename, "r") as f:
            data = []
            for line in f:
                data.append(json.loads(line))

    return data

def maybe_create_folder(folder_path):
    """
    Creates a folder at the specified path if it doesn't already exist.

    Args:
        folder_path (str): The path of the folder to create.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")



def split_sentence_with_newlines(sentences: List[str]) -> List[str]:
    results = []
    for line in sentences:
        splitted_line = line.split("\n")
        splitted_line = [item.strip() for item in splitted_line]
        splitted_line = [item for item in splitted_line if item]

        results.extend(splitted_line)

    return results


def is_proper_sentence(text: str):
    """
    Heuristically checks whether `text` looks like a proper sentence by requiring:
      - a subject (nsubj/csubj/nsubjpass/csubjpass; optionally allow imperatives without an explicit subject)
      - a predicate: either a verbal root (VERB/AUX) or a copular construction
    """
    if not text or not text.strip():
        return False

    doc = nlp(text)

    subject_labels = {"nsubj", "csubj", "nsubjpass", "csubjpass"}
    has_subject = any(tok.dep_ in subject_labels for tok in doc)

    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if not root:
        return False

    has_predicate = root.pos_ in {"VERB", "AUX"}

    if not has_predicate and root.pos_ in {"ADJ", "NOUN", "PROPN"}:
        has_predicate = any(child.dep_ == "cop" and child.pos_ in {"AUX", "VERB"} for child in root.children)

    return has_subject and has_predicate



def sentence_filtering(sentences: List[str]) -> List[str]:

    return [i for i, s in enumerate(sentences) if is_proper_sentence(s)]


SIMPLE_TEXT_SPLITTER = lambda text: [item.strip(string.punctuation).lower() for item in text.replace("_", " ").replace("-", " ").split()]