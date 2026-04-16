import tiktoken
from chonkie import TokenChunker


CHUNKER = {
    "chunker": None
}

def init_chunker(chunk_size = 200, chunk_overlap = 20, tokenizer = None):
    if not CHUNKER["chunker"]:
        print("Initializing Chunker. Currently only allow TokenChunker")
        if not tokenizer:
            print("No tokenizer provided. Default to: 'cl100k_base'")
            tokenizer = tiktoken.get_encoding("cl100k_base")

        chunker = TokenChunker(
            tokenizer = tokenizer,
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap
        )

        CHUNKER["chunker"] = chunker


def text_chunking(text):
    assert CHUNKER["chunker"] is not None

    chunks = CHUNKER["chunker"](text)

    res = []
    # Access chunks
    for chunk in chunks:
        res.append(chunk.text)

    return res