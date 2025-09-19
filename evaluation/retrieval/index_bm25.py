import json, os, hydra
from omegaconf import DictConfig
from utils.text_chunking import init_chunker, text_chunking
from utils.allowed_datasets import ALLOWED_DATASETS


@hydra.main(version_base=None, config_path="../../conf/evaluation/", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig):
    dataset_name = cfg.general.dataset
    dataset_date = cfg.general.dataset_date
    work_dir = cfg.general.work_dir
    outfolder = cfg.retrieval.index.outfolder
    chunk_size = cfg.retrieval.index.chunk_size
    chunk_overlap = cfg.retrieval.index.chunk_overlap

    assert dataset_name in ALLOWED_DATASETS

    init_chunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    dataset_name_2_relative_path = {
        dn: os.path.join("benchmarks", dataset_date, dn) if dataset_date is not None else os.path.join("data", dn) \
            for dn in ALLOWED_DATASETS
    }

    # load corpus
    corpus_path = os.path.join(
        work_dir, 
        dataset_name_2_relative_path[dataset_name],
        "corpus.jsonl"    
    )

    corpus = []
    with open(corpus_path) as f:
        for line in f:
            jline = json.loads(line)
            corpus.append(jline)

    outfolder_dataset = os.path.join(outfolder, "collections", f"{dataset_name}__bm25")
    outfile = os.path.join(outfolder_dataset, "chunk.jsonl")

    with open(outfile, "w") as f:
        for line in corpus:
            docid = line.get("_id", line.get("id", None))
            assert docid is not None
            title = line["title"]
            abstract = line["text"]

            content = f"{title}. {abstract}"

            content_chunks = text_chunking(content)

            for chunk_idx, chunk in enumerate(content_chunks):
                to_write = {
                    "id": f"{docid}--__--{chunk_idx}",
                    "contents": chunk
                }

                json.dump(to_write, f)
                f.write("\n")


if __name__ == "__main__":
    main()