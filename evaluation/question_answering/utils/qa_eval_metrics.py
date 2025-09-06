import os, json
import numpy as np
from tqdm import tqdm
from collections import Counter

def run_bertscore_evaluation(outfile_pred, outfile_gt):
    import subprocess
    command = [
        "bert-score",
        "-r", outfile_gt,
        "-c", outfile_pred,
        "--lang", "en",
        "--rescale_with_baseline"
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    print(result.stdout)

    return

MINICHECK = {
    "model": None,
    "model_name": None
}
def init_minicheck(model_name='flan-t5-large', cache_dir='./ckpts'):
    from minicheck.minicheck import MiniCheck
    if MINICHECK["model_name"] != model_name:
        print(f"Initializing Minicheck ({model_name})")
        model = MiniCheck(model_name=model_name, cache_dir=cache_dir)
        MINICHECK["model"] = model


def convert_label(p1, p2):
    # p1: prediction in gt
    # p2: gt in prediction
    # if p1 but not p2 -> missing info -> +0.5
    # if p2 but not p1 -> hallucination, redundant -> -0.5
    # if p1 and p2 -> perfect + 1
    # if not p1 and not p2 -> plain wrong -> -1

    if p1 and not p2: return 0.5
    elif p2 and not p1: return -0.5
    elif p1 and p2: return 1
    else: return -1


def run_minicheck_based_evaluation(outfile_pred, outfile_gt, minicheck_ckpt_path, evaluation_metadata_file = None):
    assert os.path.exists(outfile_pred) and os.path.exists(outfile_gt), "Prediction or outfile file missing"


    init_minicheck(cache_dir=minicheck_ckpt_path)

    predictions = []
    with open(outfile_pred) as f:
        for line in f:
            predictions.append(line)
    

    groundtruths = []
    with open(outfile_gt) as f:
        for line in f:
            groundtruths.append(line)


    pred_label_1, _, _, _ = MINICHECK["model"].score(docs=groundtruths, claims=predictions)
    pred_label_2, _, _, _ = MINICHECK["model"].score(docs=predictions, claims=groundtruths)

    pred_label = [convert_label(p1, p2) for p1, p2 in zip(pred_label_1, pred_label_2)]

    print(np.mean(pred_label))    

    if isinstance(evaluation_metadata_file, str):
        with open(evaluation_metadata_file, "w") as f:
            for p in pred_label:
                f.write(str(p) + "\n")




def run_llm_based_evaluation(queries, outfile_pred, outfile_gt, OPENAI_CLIENT, evaluation_metadata_file = None):
    assert os.path.exists(outfile_pred) and os.path.exists(outfile_gt), "Prediction or outfile file missing"

    predictions = []
    with open(outfile_pred) as f:
        for line in f:
            predictions.append(line)
    

    groundtruths = []
    with open(outfile_gt) as f:
        for line in f:
            groundtruths.append(line)

    assert len(queries) == len(groundtruths) == len(predictions)


    system_prompt = f"""You are an expert QA evaluator. Given a question, a ground-truth answer, and a predicted answer, classify the predicted answer into exactly one of the following categories:

- CORRECT: The prediction fully captures the essential meaning and completeness of the ground truth.
- MISSING: The prediction is partially accurate but misses important information from the ground truth, or the prediction acknowledge that it does not have enough information to answer the question
- INCORRECT: The prediction contains factual errors, contradictions, hallucinations, or irrelevant information.
Output only the category (CORRECT, MISSING, or INCORRECT) without any further text.

For each example below, first explain your reasoning step-by-step in 2-3 sentences, then output only the category (CORRECT, MISSING, or INCORRECT).

Example 1:
Question: What is the capital of France?
Ground-truth answer: Paris
Predicted answer: Paris is the capital city of France.
Explanation: The prediction correctly identifies Paris as the capital and includes minor contextual information that does not alter the factual correctness. It captures the full meaning of the ground truth.
CORRECT

Example 2:
Question: Who wrote "To Kill a Mockingbird"?
Ground-truth answer: Harper Lee
Predicted answer: The book was written by an American author.
Explanation: The prediction is partially correct because it acknowledges an American author wrote the book, but it does not specify the author’s name, which is key information missing from the answer.
MISSING

Example 3:
Question: What year was Marie Curie awarded her second Nobel Prize?
Ground-truth answer: 1911
Predicted answer: I don't have enough information to answer that question.
Explanation: The prediction admits a lack of sufficient information to answer the question and does not provide any answer specific to the ground truth, thereby missing essential information.
MISSING

Example 4:
Question: What is photosynthesis?
Ground-truth answer: Photosynthesis is the process by which green plants use sunlight to synthesize foods from carbon dioxide and water.
Predicted answer: Photosynthesis is the process of cell division in animals.
Explanation: The prediction is factually incorrect as it confuses photosynthesis with cell division, contradicting the ground truth.
INCORRECT"""
    
    user_prompt_template = f"""
Question: [QUESTION]
Ground-truth answer: [GROUNDTRUTH]
Predicted answer: [PREDICTION]"""
    
    pred_label = []
    for query, prediction, groundtruth in tqdm(zip(queries, predictions, groundtruths)):
        user_prompt = user_prompt_template.replace("[QUESTION]", query).replace("[GROUNDTRUTH]", groundtruth).replace("[PREDICTION]", prediction)

        resp = OPENAI_CLIENT["client"].chat.completions.create(
            model=OPENAI_CLIENT["model"],
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            max_tokens = 100,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        result = resp.choices[0].message.content.strip()

        for l in ["INCORRECT", "CORRECT", "MISSING"]:
            if l in result:
                to_append = {"INCORRECT":-1, "CORRECT": 1, "MISSING": 0}[l]
                break

        pred_label.append(to_append)

    print(np.mean(pred_label))    

    if isinstance(evaluation_metadata_file, str):
        with open(evaluation_metadata_file, "w") as f:
            for p in pred_label:
                f.write(str(p) + "\n")



def run_llm_based_evaluation_keypoints(queries, outfile_pred, outfile_gt, OPENAI_CLIENT, evaluation_metadata_file):
    assert os.path.exists(outfile_pred) and os.path.exists(outfile_gt), "Prediction or outfile file missing"

    predictions = []
    with open(outfile_pred) as f:
        for line in f:
            predictions.append(line)
    

    groundtruths = []
    with open(outfile_gt) as f:
        for line in f:
            groundtruths.append(line)

    assert len(queries) == len(groundtruths) == len(predictions)

    system_prompt = f"""In this task , you will receive a question , a generated answer , and multiple key points \
from a standard answer . Please categorize each key point by determining whether it is Relevant , \
Irrelevant , or Wrong based on the generated answer .
- Relevant: indicates that the generated answer contains key information that is related to and
consistent with the key point described in the standard answer .
- Irrelevant: indicates that the generated answer does not contain or involve information related
to the key point in the standard answer .
- Wrong: indicates that the generated answer contains information related to the key point but it
is incorrect or contradicts the standard answer keypoints .


Example 1:
##QUESTION##: What ukulele-based music education program, created by James Hill and Chalmers Doane in 2008, is widely used in Canadian schools?
##GENERATED_ANSWER##: The widely used ukulele-based music education program created by James Hill and Chalmers Doane in 2008 is called Ukulele in the Classroom.
##KEYPOINTS##: 
- "Ukulele in the Classroom", a revised program created by James Hill and Doane in 2008, is a staple of music education in Canada.
##OUTPUT##: ["Relevant"]


Example 2:
##QUESTION##: What historic achievements did Cristiano Ronaldo earn while playing for Manchester United regarding the Ballon d'Or and the FIFA World Player of the Year award?
##GENERATED_ANSWER##: Cristiano Ronaldo made historic achievements while playing for Manchester United by winning the Ballon d'Or in 2008, becoming the club's first Ballon d'Or winner since George Best in 1968. In the same year, he also won the FIFA World Player of the Year award, making him the first Premier League player to receive this prestigious title
##KEYPOINTS##: 
- Cristiano Ronaldo became United's first Ballon d'Or winner since Best in 1968.
- Cristiano Ronaldo was the first Premier League player to be named the FIFA World Player of the Year.
##OUTPUT##: ["Relevant", "Relevant"]


Example 3:
##QUESTION##: What are the primary ingredients and traditional preparation method for Italian carbonara pasta?
##GENERATED_ANSWER##: Italian carbonara pasta is traditionally made with spaghetti, eggs, Pecorino Romano cheese, guanciale, and black pepper. The preparation involves cooking the guanciale until crispy, then mixing it with cooked pasta and a sauce made from beaten eggs and cheese, without using cream.
##KEYPOINTS##:
- Primary ingredients: spaghetti, eggs, Pecorino Romano cheese, guanciale (cured pork cheek), black pepper.
- No cream is used in the authentic recipe; the sauce is created from eggs and cheese emulsified with pasta water.
- Often mistakenly includes pancetta instead of guanciale, but guanciale is traditional.
- The dish originated in Rome during the mid-20th century.
##OUTPUT##: ["Relevant", "Relevant", "Irrelevant", "Irrelevant"]

Example 4:
##QUESTION##: What were Albert Einstein's major contributions to physics, including his famous equation?
##GENERATED_ANSWER##: Albert Einstein's major contributions include the theory of general relativity in 1905, which revolutionized our understanding of gravity, and his famous equation E=mc² from special relativity. He also won the Nobel Prize in Physics in 1921 for his work on the photoelectric effect.
##KEYPOINTS##:
- Developed the theory of special relativity in 1905, including the equation E=mc².
- Developed the theory of general relativity in 1915.
- Won the Nobel Prize in 1921 for the photoelectric effect, not relativity.
- Contributed to quantum theory through his explanation of the photoelectric effect.
##OUTPUT##: ["Wrong", "Wrong", "Relevant", "Irrelevant"]"""
    
    user_prompt_template = f"""User input:
##QUESTION##: [QUESTION]
##GENERATED_ANSWER##: [GENERATED_ANSWER]
##KEYPOINTS##:
[KEYPOINTS]
##OUTPUT##:"""
    


    relevance = []
    irrelevance = []
    wrong = []
    for query, prediction, groundtruth in tqdm(zip(queries, predictions, groundtruths)):
        keypoints = groundtruth.split("--__--")
        keypoints = "\n".join(["- " + kp for kp in keypoints])

        print(prediction, keypoints)

        user_prompt = user_prompt_template.replace("[QUESTION]", query).replace("[KEYPOINTS]", keypoints).replace("[GENERATED_ANSWER]", prediction)

        resp = OPENAI_CLIENT["client"].chat.completions.create(
            model=OPENAI_CLIENT["model"],
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            max_tokens = 100,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        result = resp.choices[0].message.content.strip()
        print(result)
        try:
            json_result = json.loads(result)
        except Exception as e:
            continue
            
        
        result_counter = Counter(json_result)
        total = sum(result_counter.values())

        print(result_counter)

        relevance.append(result_counter["Relevant"] / total)
        irrelevance.append(result_counter["Irrelevant"] / total)
        wrong.append(result_counter["Wrong"] / total)

    
    formatted_output = f"REL: {np.mean(relevance)}\nIRREL: {np.mean(irrelevance)}\nWRONG: {np.mean(wrong)}"
    print(formatted_output)    

    if isinstance(evaluation_metadata_file, str):
        with open(evaluation_metadata_file, "w") as f:
            for r,i,w in zip(relevance, irrelevance, wrong):
                f.write(f"{r},{i},{w}" + "\n")