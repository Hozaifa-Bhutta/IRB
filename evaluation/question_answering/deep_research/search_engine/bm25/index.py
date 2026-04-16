import json, os, hydra
from omegaconf import DictConfig

from tqdm import tqdm


@hydra.main(version_base=None, config_path="../../conf/", config_name="bm25")
def main(cfg: DictConfig):
    dataset_name = cfg.dataset
    work_dir = cfg.work_dir
    collection_folder = cfg.index.collection_folder
    passage_corpus = cfg.passage_corpus

    # load corpus
    corpus_path = os.path.join(
        work_dir, 
        dataset_name,
        "passage_corpus.jsonl" if passage_corpus else "corpus.jsonl"
    )

    corpus = []
    with open(corpus_path) as f:
        for line in f:
            jline = json.loads(line)
            corpus.append(jline)

    outfile = os.path.join(collection_folder, "chunk.jsonl")

    with open(outfile, "w") as f:
        for line in tqdm(corpus):
            docid = line.get("_id", line.get("id", None))
            assert docid is not None
            title = line["title"]
            text = line["text"]

            to_write = {
                "id": docid,
                "title": title,
                "contents": text
            }

            json.dump(to_write, f)
            f.write("\n")


if __name__ == "__main__":
    main()