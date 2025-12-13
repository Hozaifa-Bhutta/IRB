import os, json, string
import numpy as np
from tqdm import tqdm
from collections import Counter
from evaluation.question_answering.utils.prompts import LLM_BASED_EVALUATION_PROMPT


def run_llm_based_evaluation_keypoints_(
        eval_metadata_outfile, 
        LLM,
        eval_result_outfile
    ):
    assert os.path.exists(eval_metadata_outfile), "Prediction or outfile file missing"

    with open(eval_metadata_outfile) as f:
        temp = json.load(f)

    query_ids = list(temp["predictions"].keys())

    predictions = [temp["predictions"][qid] for qid in query_ids]
    groundtruths = [temp["groundtruths"][qid] for qid in query_ids]
    queries = [temp["queries"][qid] for qid in query_ids]

    assert len(queries) == len(groundtruths) == len(predictions)

    system_prompt = LLM_BASED_EVALUATION_PROMPT["system"]
    user_prompt_template = LLM_BASED_EVALUATION_PROMPT["user"][:]
    


    eval_result = {}
    for query_id, query, prediction, groundtruth in tqdm(zip(query_ids, queries, predictions, groundtruths)):
        keypoints = groundtruth.split("--__--")
        keypoints = "\n".join(["- " + kp for kp in keypoints])
        num_keypoints = len(groundtruth.split("--__--"))

        print(query, prediction, keypoints)

        try:
            user_prompt = user_prompt_template.replace("[QUESTION]", query)\
                                                .replace("[KEYPOINTS]", keypoints)\
                                                .replace("[GENERATED_ANSWER]", prediction)\
                                                .replace("[NUM_KEYPOINTS]", str(num_keypoints))

            result = LLM.generate(
                system_prompt = system_prompt,
                user_prompt = user_prompt,
                max_output_tokens = 128
            )
            print(result)
        except Exception as e:
            print(e)
            continue


        try:
            json_result = json.loads(result.strip())
            assert len(json_result) == num_keypoints
            print("JSON:", json_result)
        except Exception as e:
            continue
            
        
        result_counter = Counter(json_result)
        total = sum(result_counter.values())

        print(result_counter)

        eval_result[query_id] = {k: v / total for k, v in result_counter.items()}

    
    avg_correct = np.mean([v.get("CORRECT", 0) for k, v in eval_result.items()])
    avg_incorrect = np.mean([v.get("INCORRECT", 0) for k, v in eval_result.items()])
    avg_not_attempted = np.mean([v.get("NOT_ATTEMPTED", 0) for k, v in eval_result.items()])

    formatted_output = f"CORRECT: {avg_correct}\nINCORRECT: {avg_incorrect}\nNOT_ATTEMPTED: {avg_not_attempted}"
    print(formatted_output)    

    if isinstance(eval_result_outfile, str):
        assert eval_result_outfile.endswith(".json")
        with open(eval_result_outfile, "w") as f:
            json.dump(eval_result, f, indent = 4)



def run_llm_based_evaluation_keypoints(
        eval_metadata_outfile, 
        groundtruth_answers,
        LLM,
        LLM2 = None,
        eval_result_outfile = None
    ):
    assert os.path.exists(eval_metadata_outfile), "Prediction or outfile file missing"

    id2gt = {line["_id"]: line for line in groundtruth_answers}

    with open(eval_metadata_outfile) as f:
        temp = json.load(f)

    query_ids = list(temp["predictions"].keys())

    predictions = [temp["predictions"][qid] for qid in query_ids]
    groundtruths_full = ["--__--".join(id2gt[qid]["text"]) for qid in query_ids]
    groundtruths_short = [id2gt[qid]["short"] for qid in query_ids]
    queries = [temp["queries"][qid] for qid in query_ids]

    assert len(queries) == len(groundtruths_full) == len(groundtruths_short) == len(predictions)

    system_prompt = LLM_BASED_EVALUATION_PROMPT["system"]
    user_prompt_template = LLM_BASED_EVALUATION_PROMPT["user"][:]
    


    llm_use_count = 0
    eval_result = {}
    for query_id, query, prediction, gt_full, gt_short in tqdm(zip(query_ids, queries, predictions, groundtruths_full, groundtruths_short)):
        keypoints = gt_full.split("--__--")

        print("GT:", gt_short, "\nPred:", prediction)

        json_result = []
        if gt_short.lower().strip(string.punctuation) == prediction.lower().strip(string.punctuation):
            json_result = ["CORRECT"]
        elif "I don't know" in prediction or "I don’t know" in prediction:
            json_result = ["NOT_ATTEMPTED"]
        elif "false premise question" in prediction.lower():
            if gt_short == "False premise question": json_result = ["CORRECT"]
            else: json_result = ["INCORRECT"]
        else:
            llm_use_count += 1
            print("USE LLM to Eval:", llm_use_count, " times")
            try:
                user_prompt = user_prompt_template.replace("[QUESTION]", query)\
                                                    .replace("[SHORT]", gt_short)\
                                                    .replace("[GENERATED_ANSWER]", prediction)\
                                                    .replace("[ADD_KEYPOINTS_HERE]", "\n".join(keypoints))

                for llm in [LLM, LLM2]:
                    if llm is None: continue
                    result = llm.generate(
                        system_prompt = system_prompt,
                        user_prompt = user_prompt,
                        max_output_tokens = 16,
                        temperature = 0.2
                    )
                    if "A" in result: json_result.append("CORRECT")
                    elif "B" in result: json_result.append("INCORRECT")
                    elif "C" in result: json_result.append("NOT_ATTEMPTED")
                    else: json_result.append("INCORRECT")
            except Exception as e:
                print(e)
                continue

            
        
        result_counter = Counter(json_result)
        total = sum(result_counter.values())

        print(result_counter)

        eval_result[query_id] = {k: v / total for k, v in result_counter.items()}

    
    avg_correct = np.mean([v.get("CORRECT", 0) for k, v in eval_result.items()])
    avg_incorrect = np.mean([v.get("INCORRECT", 0) for k, v in eval_result.items()])
    avg_not_attempted = np.mean([v.get("NOT_ATTEMPTED", 0) for k, v in eval_result.items()])

    formatted_output = f"CORRECT: {avg_correct}\nINCORRECT: {avg_incorrect}\nNOT_ATTEMPTED: {avg_not_attempted}"
    print(formatted_output)    

    if isinstance(eval_result_outfile, str):
        assert eval_result_outfile.endswith(".json")
        with open(eval_result_outfile, "w") as f:
            json.dump(eval_result, f, indent = 4)