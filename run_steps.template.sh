INTERMEDIATE_RESULTS_FOLDER=./dummy_gitig_/
OPENAI_API_KEY=***

mkdir "${INTERMEDIATE_RESULTS_FOLDER}/step0"
python steps/0_wiki_dump_processing.py \
--input_file /scratch/lamdo/wiki_dump/enwiki-20250630-cirrussearch-content.json.gz \
--output_folder "${INTERMEDIATE_RESULTS_FOLDER}/step0" \
--offset 10000 \
--max_pages 10

mkdir "${INTERMEDIATE_RESULTS_FOLDER}/step1"
python steps/1_fact_extraction.py \
--input_folder "${INTERMEDIATE_RESULTS_FOLDER}/step0" \
--output_folder "${INTERMEDIATE_RESULTS_FOLDER}/step1"

mkdir "${INTERMEDIATE_RESULTS_FOLDER}/step2_1"
python steps/2_1_url_content_crawling.py \
--input_folder "${INTERMEDIATE_RESULTS_FOLDER}/step1" \
--output_folder "${INTERMEDIATE_RESULTS_FOLDER}/step2_1" \
--max_urls_per_page 10


mkdir "${INTERMEDIATE_RESULTS_FOLDER}/step2_2"
python steps/2_2_fact_decontextualization.py \
--extracted_facts_folder "${INTERMEDIATE_RESULTS_FOLDER}/step1" \
--crawled_url_content_folder "${INTERMEDIATE_RESULTS_FOLDER}/step2_1" \
--output_folder "${INTERMEDIATE_RESULTS_FOLDER}/step2_2" \
--context_window_size 5 \
--openai_api_key $OPENAI_API_KEY

mkdir "${INTERMEDIATE_RESULTS_FOLDER}/step3"
python steps/3_question_generation.py \
--input_folder "${INTERMEDIATE_RESULTS_FOLDER}/step2_2" \
--output_folder "${INTERMEDIATE_RESULTS_FOLDER}/step3" \
--openai_api_key $OPENAI_API_KEY