export CUDA_VISIBLE_DEVICES=2

INTERMEDIATE_RESULTS_FOLDER=./dummy_gitig_/
OPENAI_API_KEY=***


mkdir "${INTERMEDIATE_RESULTS_FOLDER}/step0"
python steps/0_wiki_dump_processing.py \
--input_file /scratch/lamdo/wiki_dump/enwiki-20250630-cirrussearch-content.json.gz \
--step0_output_folder "${INTERMEDIATE_RESULTS_FOLDER}/step0" \
--offset 0 \
--max_pages 200



mkdir "${INTERMEDIATE_RESULTS_FOLDER}/step1"
python steps/1_fact_extraction.py \
--step0_output_folder "${INTERMEDIATE_RESULTS_FOLDER}/step0" \
--step1_output_folder "${INTERMEDIATE_RESULTS_FOLDER}/step1"

mkdir "${INTERMEDIATE_RESULTS_FOLDER}/step2_1"
python steps/2_1_url_content_crawling.py \
--step1_output_folder "${INTERMEDIATE_RESULTS_FOLDER}/step1" \
--step2_1_output_folder "${INTERMEDIATE_RESULTS_FOLDER}/step2_1" \
--max_urls_per_page 10


mkdir "${INTERMEDIATE_RESULTS_FOLDER}/step2_2"
python steps/2_2_fact_decontextualization.py \
--step1_output_folder "${INTERMEDIATE_RESULTS_FOLDER}/step1" \
--step2_1_output_folder "${INTERMEDIATE_RESULTS_FOLDER}/step2_1" \
--step2_2_output_folder "${INTERMEDIATE_RESULTS_FOLDER}/step2_2" \
--context_window_size 5 \
--openai_api_key $OPENAI_API_KEY

mkdir "${INTERMEDIATE_RESULTS_FOLDER}/step3"
python steps/3_fact_groundedness_check.py \
--step1_output_folder "${INTERMEDIATE_RESULTS_FOLDER}/step1" \
--step2_1_output_folder "${INTERMEDIATE_RESULTS_FOLDER}/step2_1" \
--step2_2_output_folder "${INTERMEDIATE_RESULTS_FOLDER}/step2_2" \
--step3_output_folder "${INTERMEDIATE_RESULTS_FOLDER}/step3"

mkdir "${INTERMEDIATE_RESULTS_FOLDER}/step4"
python steps/4_question_generation.py \
--step2_2_output_folder "${INTERMEDIATE_RESULTS_FOLDER}/step2_2" \
--step3_output_folder "${INTERMEDIATE_RESULTS_FOLDER}/step3" \
--step4_output_folder "${INTERMEDIATE_RESULTS_FOLDER}/step4" \
--openai_api_key $OPENAI_API_KEY

rm -r "${INTERMEDIATE_RESULTS_FOLDER}/step5"
mkdir "${INTERMEDIATE_RESULTS_FOLDER}/step5"
python steps/5_combine_results.py \
--step1_output_folder "${INTERMEDIATE_RESULTS_FOLDER}/step1" \
--step2_1_output_folder "${INTERMEDIATE_RESULTS_FOLDER}/step2_1" \
--step2_2_output_folder "${INTERMEDIATE_RESULTS_FOLDER}/step2_2" \
--step3_output_folder "${INTERMEDIATE_RESULTS_FOLDER}/step3" \
--step4_output_folder "${INTERMEDIATE_RESULTS_FOLDER}/step4" \
--step5_output_folder "${INTERMEDIATE_RESULTS_FOLDER}/step5"