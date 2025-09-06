NUM_CHUNKS=1
OUT_FOLDER=/scratch/lamdo/beir_splade/
WORK_DIR=/home/lamdo/splade
RETRIEVAL_METADATA_PATH=/scratch/lamdo/IRB/retrieval_metadata/

export CONFIG_NAME="bm25"
export OPENAI_API_KEY="***"
DATASET=irb


DATASET_MODEL_FOLDER_NAME="${DATASET}__bm25"
COLLECTION_FOLDER="$OUT_FOLDER/collections/$DATASET_MODEL_FOLDER_NAME"
INDEX_FOLDER="$OUT_FOLDER/indexes/$DATASET_MODEL_FOLDER_NAME"

rm -r $COLLECTION_FOLDER
rm -r $INDEX_FOLDER

mkdir $COLLECTION_FOLDER
mkdir $INDEX_FOLDER

# create representation
for chunk_idx in $(seq 0 $((${NUM_CHUNKS} - 1))); do
    python evaluation/retrieval/index_bm25.py \
    general.dataset=$DATASET \
    retrieval.index.outfolder=$OUT_FOLDER \
    general.work_dir=$WORK_DIR
done

# do indexing
python -m pyserini.index.lucene \
--collection JsonCollection \
--input $COLLECTION_FOLDER \
--index $INDEX_FOLDER \
--generator DefaultLuceneDocumentGenerator \
--threads 8 --storePositions --storeDocvectors --storeRaw

# evaluate retrieval
python evaluation/retrieval/eval_bm25.py \
retrieval.eval.index_folder="$OUT_FOLDER/indexes/" \
general.dataset=$DATASET \
general.work_dir=$WORK_DIR \
general.retrieval_metadata_path="$RETRIEVAL_METADATA_PATH"

python evaluation/question_answering/eval.py \
    general.work_dir=$WORK_DIR \
    general.dataset=irb \
    qa.outfolder=$OUTFOLDER \
    general.retrieval_model=$RETRIEVAL_MODEL \
    general.retrieval_metadata_path="$RETRIEVAL_METADATA_PATH"