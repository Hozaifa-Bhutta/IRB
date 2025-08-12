from chonkie import TokenChunker


CHUNKER = {
    "chunker": None
}

def init_chunker(chunk_size = 200, chunk_overlap = 20):
    if not CHUNKER["chunker"]:
        print("Initializing Chunker. Currently only allow TokenChunker")

        chunker = TokenChunker(
            tokenizer = "word",
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