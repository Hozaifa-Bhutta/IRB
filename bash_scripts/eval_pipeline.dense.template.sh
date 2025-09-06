NUM_CHUNKS=1
OUT_FOLDER=/scratch/lamdo/beir_splade/
WORK_DIR=/home/lamdo/splade
RETRIEVAL_METADATA_PATH=/scratch/lamdo/IRB/retrieval_metadata/

CUDA_DEVICE=2
DATASET="irb"


export CONFIG_NAME="dense"
export OPENAI_API_KEY="***"


models=(
    "e5_base"
)

for model in "${models[@]}"; do
    DATASET_MODEL_FOLDER_NAME="${DATASET}__${model}"
    COLLECTION_FOLDER="$OUT_FOLDER/collections/$DATASET_MODEL_FOLDER_NAME"
    INDEX_FOLDER="$OUT_FOLDER/indexes/$DATASET_MODEL_FOLDER_NAME"


    # # create representation and indexing
    # rm -r $COLLECTION_FOLDER
    # rm -r $INDEX_FOLDER
    # mkdir $COLLECTION_FOLDER
    # mkdir $INDEX_FOLDER
    # for chunk_idx in $(seq 0 $((${NUM_CHUNKS} - 1))); do
    #     CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
    #     python evaluation/retrieval/index_dense.py \
    #     general.work_dir=$WORK_DIR \
    #     general.dataset=$DATASET \
    #     general.retrieval_model=$model \
    #     general.index_folder=$INDEX_FOLDER \
    #     retrieval.index.num_chunks=$NUM_CHUNKS \
    #     retrieval.index.chunk_idx=$chunk_idx


    # retrieval evaluation
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python evaluation/retrieval/eval_dense.py \
    general.retrieval_model=$model \
    general.work_dir=$WORK_DIR \
    general.index_folder=$INDEX_FOLDER \
    general.dataset=$DATASET \
    general.retrieval_metadata_path="$RETRIEVAL_METADATA_PATH"

    # qa evaluation
    python evaluation/question_answering/eval.py \
    general.work_dir=$WORK_DIR \
    general.dataset=irb \
    qa.outfolder=$OUTFOLDER \
    general.retrieval_model=$model \
    general.retrieval_metadata_path="$RETRIEVAL_METADATA_PATH"


    done
done