

# LLM_BASED_EVALUATION_PROMPT = {
#     "system": f"""In this task, you will receive a question, a generated answer, and multiple key points \
# from a standard answer. For each keypoint, please categorize if the generated answer is CORRECT, INCORRECT, or NOT_ATTEMPTED \
# given the keypoint.

# 1. CORRECT: 
#     + Fully contain the important information in the keypoint.
#     + Do not contain any information that contradicts the keypoint.
#     + Only semantic meaning matters; capitalization, punctuation, grammar, and order
# don’t matter.
#     + Hedging and guessing are permissible, provided that the keypoint is fully
# included and the response contains no incorrect information or
# contradictions.

# 2. INCORRECT: 
#     + A factual statement in the answer contradicts the keypoint. Incorrect
# statements that have some hedging (e.g., "it is possible that", "although i’
# m not sure, i think") are also considered incorrect.


# 3. NOT_ATTEMPTED: 
#     + The important information in the gold target is not included in the answer.
#     + No statements in the answer contradict the gold target.


# Example 1:
# ##QUESTION##: What ukulele-based music education program, created by James Hill and Chalmers Doane in 2008, is widely used in Canadian schools?
# ##GENERATED_ANSWER##: The widely used ukulele-based music education program created by James Hill and Chalmers Doane in 2008 is called Ukulele in the Classroom.
# ##NUM_KEYPOINTS##: 1
# ##KEYPOINTS##: 
# - "Ukulele in the Classroom", a revised program created by James Hill and Doane in 2008, is a staple of music education in Canada.
# ##OUTPUT##: ["CORRECT"]

# Example 2:
# ##QUESTION##: What are the names of Barack Obama’s children?
# ##GENERATED_ANSWER##: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
# ##NUM_KEYPOINTS##: 2
# ##KEYPOINTS##: 
# - Malia Ann Obama is Barrack Obama's daughter
# - Sasha Obama is Barrack Obama's daughter
# ##OUTPUT##: ["NOT_ATTEMPTED", "NOT_ATTEMPTED"]


# Example 3:
# ##QUESTION##: What historic achievements did Cristiano Ronaldo earn while playing for Manchester United regarding the Ballon d'Or and the FIFA World Player of the Year award?
# ##GENERATED_ANSWER##: Cristiano Ronaldo made historic achievements while playing for Manchester United by winning the Ballon d'Or in 2008, becoming the club's first Ballon d'Or winner since George Best in 1968. In the same year, he also won the FIFA World Player of the Year award, making him the first Premier League player to receive this prestigious title
# ##NUM_KEYPOINTS##: 2
# ##KEYPOINTS##: 
# - Cristiano Ronaldo became United's first Ballon d'Or winner since Best in 1968.
# - Cristiano Ronaldo was the first Premier League player to be named the FIFA World Player of the Year.
# ##OUTPUT##: ["CORRECT", "CORRECT"]


# Example 4:
# ##QUESTION##: What are the primary ingredients and traditional preparation method for Italian carbonara pasta?
# ##GENERATED_ANSWER##: Italian carbonara pasta is traditionally made with spaghetti, eggs, Pecorino Romano cheese, guanciale, and black pepper. The preparation involves cooking the guanciale until crispy, then mixing it with cooked pasta and a sauce made from beaten eggs and cheese, without using cream.
# ##NUM_KEYPOINTS##: 4
# ##KEYPOINTS##:
# - Primary ingredients: spaghetti, eggs, Pecorino Romano cheese, guanciale (cured pork cheek), black pepper.
# - No cream is used in the authentic recipe; the sauce is created from eggs and cheese emulsified with pasta water.
# - Often mistakenly includes pancetta instead of guanciale, but guanciale is traditional.
# - The dish originated in Rome during the mid-20th century.
# ##OUTPUT##: ["CORRECT", "CORRECT", "INCORRECT", "INCORRECT"]

# Example 5:
# ##QUESTION##: What were Albert Einstein's major contributions to physics, including his famous equation?
# ##GENERATED_ANSWER##: Albert Einstein's major contributions include the theory of general relativity in 1905, which revolutionized our understanding of gravity, and his famous equation E=mc² from special relativity. He also won the Nobel Prize in Physics in 1921 for his work on the photoelectric effect.
# ##NUM_KEYPOINTS##: 4
# ##KEYPOINTS##:
# - Developed the theory of special relativity in 1905, including the equation E=mc².
# - Developed the theory of general relativity in 1915.
# - Won the Nobel Prize in 1921 for the photoelectric effect, not relativity.
# - Contributed to quantum theory through his explanation of the photoelectric effect.
# ##OUTPUT##: ["INCORRECT", "INCORRECT", "CORRECT", "INCORRECT"]""",

#     "user": f"""User input:
# ##QUESTION##: [QUESTION]
# ##GENERATED_ANSWER##: [GENERATED_ANSWER]
# ##NUM_KEYPOINTS##: [NUM_KEYPOINTS]
# ##KEYPOINTS##:
# [KEYPOINTS]
# ##OUTPUT##:"""
# }




LLM_BASED_EVALUATION_PROMPT = {
    "system": f"""Your job is to look at a question, a gold target, and a predicted answer, and then
assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.

The following are examples of CORRECT predicted answers.
‘‘‘
Question: What are the names of Barack Obama’s children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I’m not sure and
would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and
Natasha Marian, but they are commonly referred to as Malia Obama and Sasha
Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
‘‘‘
These predicted answers are all CORRECT because:
- They fully contain the important information in the gold target.
- They do not contain any information that contradicts the gold target.
- Only semantic meaning matters; capitalization, punctuation, grammar, and order
don’t matter.
- Hedging and guessing are permissible, provided that the gold target is fully
included and the response contains no incorrect information or
contradictions.

The following are examples of INCORRECT predicted answers.
‘‘‘
Question: What are the names of Barack Obama’s children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: Malia, Sasha, and Susan.
Predicted answer 2: Barack Obama does not have any children.
Predicted answer 3: I think it’s either Malia and Sasha. Or it could be Malia and
Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don’t know their exact names, I can tell you that Barack
Obama has three children.
Predicted answer 5: It’s possible you may mean Betsy and Olivia. However, you should
clarify further details with updated references if necessary. Is that the
correct answer?
Predicted answer 6: It may be the case that Obama’s child is named James. However,
it’s recommended to confirm the most accurate and updated information since
this could change over time. This model may not always reflect the most current
information.
‘‘‘
These predicted answers are all INCORRECT because:
- A factual statement in the answer contradicts the gold target. Incorrect
statements that have some hedging (e.g., "it is possible that", "although i’
m not sure, i think") are also considered incorrect.

The following are examples of NOT_ATTEMPTED predicted answers.
‘‘‘
Question: What are the names of Barack Obama’s children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: I don’t know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question.
However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia,
but I’m not sure about the other one.
Predicted answer 5: I cannot answer because there are not enough information in the provided context.
‘‘‘
These predicted answers are all NOT_ATTEMPTED because:
- The important information in the gold target is not included in the answer.
- No statements in the answer contradict the gold target.


Also note the following things:
- The gold target may contain more information than the question. In such cases, the
predicted answer only needs to contain the information that is in the question.
- Do not punish predicted answers if they omit information that would be clearly
inferred from the question.
- Do not give credit for an answer if it contains any internal inconsistency.
- If the gold answer is "False premise question" the prediction is only correct when it also states "False premise question".
A prediction saying "I don't know" is not CORRECT, but NOT_ATTEMPTED

Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED.
Don’t apologize or correct yourself if there was a mistake; we are just trying
to grade the answer.""",

    "user": f"""‘‘‘
**Question:** [QUESTION]
**Gold target:** [SHORT]
**Predicted answer:** [GENERATED_ANSWER]
‘‘‘
Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED
Just return the letters "A", "B", or "C", with no text around it."""
}