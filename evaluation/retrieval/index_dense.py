import os, json, torch, faiss, hydra
import numpy as np
from omegaconf import DictConfig
from argparse import ArgumentParser
from tqdm import tqdm
from evaluation.retrieval.utils.text_embeddings import model_name_2_model_class, \
    model_name_2_tokenizer_class, model_name_2_model_path, model_name_2_prefix, text_embedding_batch, init_model
from evaluation.retrieval.utils.text_chunking import init_chunker, text_chunking
from evaluation.retrieval.utils.allowed_datasets import ALLOWED_DATASETS


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    

def do_indexing(embeddings, 
                texts = None, 
                string_ids=None, 
                index_folder=None):
    """
    Build a Faiss Flat IP index for cosine similarity with string IDs.
    Args:
        embeddings (np.ndarray): shape (num_vectors, dim)
        string_ids (list of str or None): List of string IDs.
        index_folder (str or None): Folder to save the index and mapping.
    Returns:
        index: Faiss index (IndexIDMap)
        id_map: dict mapping int_id -> string_id
    """
    embeddings = embeddings.astype('float32')
    num_vectors, dim = embeddings.shape

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Build Flat IP index
    base_index = faiss.IndexFlatIP(dim)

    # Handle string IDs
    if string_ids is not None:
        assert len(string_ids) == num_vectors
        int_ids = np.arange(num_vectors, dtype=int)
        id_map = dict(zip(int_ids, string_ids))
        index = faiss.IndexIDMap(base_index)
        index.add_with_ids(embeddings, int_ids)
    else:
        id_map = None
        index = base_index
        index.add(embeddings)

    # Optionally save
    if index_folder is not None:
        if not os.path.exists(index_folder):
            os.makedirs(index_folder)
        faiss.write_index(index, os.path.join(index_folder, "faiss_index_flatip.index"))
        if id_map is not None:
            import json
            with open(os.path.join(index_folder, "id_map.json"), "w") as f:
                json.dump({int(k): str(v) for k,v in id_map.items()}, f)

    # save raw data
    if texts is not None:
        id2text = {line["_id"]: {"id": line["_id"], "contents": line["text"], "published_date": line["published_date"]} for line in texts}
        with open(os.path.join(index_folder, "raw.json"), "w") as f:
            json.dump(id2text, f)

    return index, id_map

@hydra.main(version_base=None, config_path="../../conf/evaluation/", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig):
    retrieval_model = cfg.general.retrieval_model
    dataset = cfg.general.dataset
    dataset_date = cfg.general.dataset_date
    work_dir = cfg.general.work_dir
    index_folder = cfg.general.index_folder
    batch_size = cfg.retrieval.index.batch_size
    chunk_size = cfg.retrieval.index.chunk_size
    chunk_overlap = cfg.retrieval.index.chunk_overlap

    dataset_name_2_relative_path = {
        dn: os.path.join("benchmarks", dataset_date, dn) if dataset_date is not None else os.path.join("data", dn) \
            for dn in ALLOWED_DATASETS
    }

    model, tokenizer = init_model(retrieval_model, device = DEVICE)
    init_chunker(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        tokenizer = tokenizer
    )

    prefix_ = model_name_2_prefix.get("model_name")
    if not prefix_:
        prefix = None
    else: prefix = prefix_["doc"]


    # load corpus
    corpus_path = os.path.join(
        work_dir, 
        dataset_name_2_relative_path[dataset],
        "corpus.jsonl")
    assert os.path.exists(corpus_path)

    print("Corpus path: ", corpus_path)

    corpus = []
    with open(corpus_path) as f:
        for line in f:
            jline = json.loads(line)
            corpus.append(jline)
    
    print("Corpus length", len(corpus))


    # perform text chunking here
    texts = []
    for line in tqdm(corpus, desc = "splitting doc into chunks"):
        full_text = f"{line['title']}. {line['text']}"
        text_chunks = text_chunking(full_text)
        docid = line.get("_id", line.get("id", None))
        published_date = line["published_date"]

        for tc_chunk_idx, tc in enumerate(text_chunks):
            texts.append({"_id": f"{docid}--__--{tc_chunk_idx}",
                        "text": tc,
                        "published_date": published_date})
            
    print("Number of text chunks length", len(texts))

    ids = []
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc = "Generating embeddings"):
        batch = texts[i:i+batch_size]
        text_batch = [line["text"] for line in batch]

        batch_embeddings = text_embedding_batch(batch = text_batch, model = model, 
                                                tokenizer = tokenizer, model_name = retrieval_model, 
                                                prefix = prefix, device = DEVICE,
                                                max_length = chunk_size)

        for line, embedding in zip(batch, batch_embeddings):
            line_id = line.get("_id", line.get("id", None))
            list_embedding = embedding.cpu().tolist()
            embeddings.append(list_embedding)
            ids.append(line_id)
    
    assert len(ids) == len(embeddings) == len(texts)
    do_indexing(embeddings = np.array(embeddings), 
                texts = texts,
                string_ids = ids, 
                index_folder = index_folder)


if __name__ == "__main__":
    main()