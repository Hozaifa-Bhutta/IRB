export CUDA_VISIBLE_DEVICES=2
export CONFIG_NAME="run_0_500"

WIKI_DUMP_FILE=/home/hozaifa/code/full-opinions.csv.bz2
INTERMEDIATE_RESULTS_FOLDER=/home/hozaifa/code/IRB/intermediate_results
OPENAI_API_KEY=***


# mkdir "${INTERMEDIATE_RESULTS_FOLDER}/step0"
# python /home/hozaifa/code/IRB/steps/0_wiki_dump_processing.py \
# step0.input_file="${WIKI_DUMP_FILE}" \
# step0.output_folder="${INTERMEDIATE_RESULTS_FOLDER}/step0" 



mkdir "${INTERMEDIATE_RESULTS_FOLDER}/step1"
python /home/hozaifa/code/IRB/steps/1_fact_extraction.py \
step0.output_folder="${INTERMEDIATE_RESULTS_FOLDER}/step0"  \
step1.output_folder="${INTERMEDIATE_RESULTS_FOLDER}/step1"

# mkdir "${INTERMEDIATE_RESULTS_FOLDER}/step2_1"
# python steps/2_1_url_content_crawling.py \
# step1.output_folder="${INTERMEDIATE_RESULTS_FOLDER}/step1" \
# step2_1.output_folder="${INTERMEDIATE_RESULTS_FOLDER}/step2_1"


# mkdir "${INTERMEDIATE_RESULTS_FOLDER}/step2_2"
# python steps/2_2_fact_decontextualization.py \
# step1.output_folder="${INTERMEDIATE_RESULTS_FOLDER}/step1" \
# step2_1.output_folder="${INTERMEDIATE_RESULTS_FOLDER}/step2_1" \
# step2_2.output_folder="${INTERMEDIATE_RESULTS_FOLDER}/step2_2"

# mkdir "${INTERMEDIATE_RESULTS_FOLDER}/step3"
# python steps/3_fact_groundedness_check.py \
# step1.output_folder="${INTERMEDIATE_RESULTS_FOLDER}/step1" \
# step2_1.output_folder="${INTERMEDIATE_RESULTS_FOLDER}/step2_1" \
# step2_2.output_folder="${INTERMEDIATE_RESULTS_FOLDER}/step2_2" \
# step3.output_folder="${INTERMEDIATE_RESULTS_FOLDER}/step3"

# mkdir "${INTERMEDIATE_RESULTS_FOLDER}/step4"
# python steps/4_question_generation.py \
# step2_2.output_folder="${INTERMEDIATE_RESULTS_FOLDER}/step2_2" \
# step3.output_folder="${INTERMEDIATE_RESULTS_FOLDER}/step3" \
# step4.output_folder="${INTERMEDIATE_RESULTS_FOLDER}/step4"

# rm -r "${INTERMEDIATE_RESULTS_FOLDER}/step5"
# mkdir "${INTERMEDIATE_RESULTS_FOLDER}/step5"
# python steps/5_combine_results.py \
# step1.output_folder="${INTERMEDIATE_RESULTS_FOLDER}/step1" \
# step2_1.output_folder="${INTERMEDIATE_RESULTS_FOLDER}/step2_1" \
# step2_2.output_folder="${INTERMEDIATE_RESULTS_FOLDER}/step2_2" \
# step3.output_folder="${INTERMEDIATE_RESULTS_FOLDER}/step3" \
# step4.output_folder="${INTERMEDIATE_RESULTS_FOLDER}/step4" \
# step5.output_folder="${INTERMEDIATE_RESULTS_FOLDER}/step5"