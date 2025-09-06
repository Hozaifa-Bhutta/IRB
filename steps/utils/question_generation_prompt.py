QUESTION_GENERATION_SYSTEM_PROMPT = f"""You are an AI agent that accurately generate questions. 
Given a molecular fact (a fact that is understandable stand-alone), generate ONE question whose answer IS THE PROVIDED MOLECULAR FACT. 
The generated question should look like a typical query that people use to search on the internet. Directly generate a question without any explanation"""


QUESTION_GENERATION_USER_PROMPT = """##MOLECULAR FACT:##: [ADD_FACT_HERE]"""

#########################

QUESTION_GENERATION_SYSTEM_PROMPT = f"""Determine if all provided keypoints are unambiguous, using the following standard: each keypoint must contain enough \
key elements so that their meaning is clear and not open to multiple reasonable interpretations. If any keypoint is ambiguous, output "ambiguous". \
If all are unambiguous, generate a clear question whose answer would be a claim that can be formed by combining these keypoints.

- Step 1: For each keypoint, assess ambiguity. A keypoint is ambiguous if it is vague, allows conflicting or multiple reasonable interpretations, or lacks enough context\
 (missing essential details).
- Step 2: If any keypoint is ambiguous, immediately output "ambiguous" and do not continue.
- Step 3: If every keypoint is unambiguous:
    - Combine the keypoints into a logical claim.
    - Generate a question that, if answered, would logically yield the combined claim as the answer. The question should be clear, concise, and reference enough key elements to prevent ambiguity.
- Always perform ambiguity analysis (reasoning step) first; only generate the question (conclusion step) if all keypoints are unambiguous.

# Output Format

- If any keypoint is ambiguous: Output should be "ambiguous".
- If all keypoints are unambiguous: Output the generated question as a single, clear, concise interrogative sentence.

# Examples

**Example 1:**  
##KEYPOINTS##:  
- Apples are rich in fiber.  
- Eating fiber can improve digestion.  
##REASONING##: Both facts are clear.
##OUTPUT##: What is a benefit of eating apples based on their fiber content?

**Example 2:**  
##KEYPOINTS##:  
- The meeting took place on March 12.  
- The agenda was not discussed.
##REASONING##: Unclear what "The meeting" refers to (which meeting?).
##OUTPUT##: ambiguous

**Example 3:**   
##KEYPOINTS##:  
- Lionel Messi played five games with the B team in the 2003/2004 season but did not score.
##REASONING##: This fact is about the performance of Messi with the B team in the 2003/2004 season, which is clear.
##OUTPUT##: What noteworthy aspect occurred during the meeting on March 12?"""


QUESTION_GENERATION_USER_PROMPT = f"""User input:
##KEYPOINTS##:
[ADD_KEYPOINTS_HERE]

##REASONING##:"""


#########################


QUESTION_GENERATION_SYSTEM_PROMPT = f"""Combine the provided keypoints into a single, coherent claim. Then, craft one clear, concise interrogative sentence (a question) that, \
if answered, would result in the combined claim being the answer. The question must reference enough specific elements from the keypoints to avoid ambiguity. Ensure your \
question is logical, specific, and directly leads to the combined claim as an answer.

In addition, you are provided with the title of the document from which the keypoints are from

- Output only a single, well-formed question (interrogative sentence) per set of keypoints.
- Do not include the claim or an explanation in your response—output only the generated question.
- For each set of keypoints, reason internally to synthesize them into a claim before drafting your question. Do not state your reasoning in your output; include ONLY the question.
- Never begin examples with the answer/claim; always start reasoning internally.

**Output Format**
- Output must be a single, concise interrogative sentence (not a paragraph, summary, or explanation).
- Do not include code blocks, markdown, or extra formatting beyond the question itself.

**Examples**

Example 1:
##TITLE##: Cristiano Ronaldo
##ORIGINAL_CLAIM##: Ronaldo was named runner-up to Kaká for the 2007 Ballon d'Or, and came third, behind Kaká and Lionel Messi, in the running for the 2007 FIFA World Player of the Year award.
##KEYPOINTS##:
- Ronaldo was named runner-up to Kaká for the 2007 Ballon d'Or.
- Cristiano Ronaldo came third, behind Kaká and Lionel Messi, in the running for the 2007 FIFA World Player of the Year award.
##OUTPUT##: In 2007, how did Cristiano Ronaldo place in the Ballon d'Or voting and the FIFA World Player of the Year award rankings?

Example 2:
##TITLE##: Ukulele
##KEYPOINTS##:
- "Ukulele in the Classroom", a revised program created by James Hill and Doane in 2008, is a staple of music education in Canada
##OUTPUT##: What ukulele-based music education program, created by James Hill and Chalmers Doane in 2008, is widely used in Canadian schools?"""


QUESTION_GENERATION_USER_PROMPT = f"""User input:
##TITLE##: [ADD_TITLE_HERE]
##KEYPOINTS##:
[ADD_KEYPOINTS_HERE]"""