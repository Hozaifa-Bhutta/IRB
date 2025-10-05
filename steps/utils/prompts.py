KEYPOINT_EXTRACTION_PROMPT = {
    "system": """You are an expert at extracting and decontextualizing keypoints from claims in a factual and neutral manner. \
You are also an expert at generating question whose answer contain exclusive the mentioned keypoints. Follow these steps precisely:
1. **Extract the keypoints**: Identify and split the claim into individual keypoints based on the [KP] markers. Each segment ending with [KP] is a separate keypoint. \
  Remove the [KP] tokens from the extracted keypoints. If there is only one [KP], treat the entire claim as a single keypoint.
2. **Decontextualize each keypoint**: For each extracted keypoint, rewrite it to make it standalone and unambiguous by incorporating necessary details from the provided \
  context. Follow these decontextualization criteria strictly:
   - **Ambiguity Criteria (resolve these where present)**:
     - Coreferences (e.g., pronouns like "he" or "it" should be replaced with specific entities).
     - Vague entities or events (e.g., add clarifying details like full names, dates, or locations).
     - Incomplete information (e.g., ensure the claim uniquely specifies all key elements without multiple interpretations).
     - Use only information from the provided context—do not add external knowledge.
     - Ensure the rewritten keypoint includes all key elements (e.g., who, what, when, where, why/how) to form a complete, interpretable fact without multiple possible meanings.
3. **Question generation given keypoints**: Based on the decontextualized keypoints, generate a question whose answer exclusively contain the keypoints.

Example 1:
##CLAIM##: Ronaldo became United's first Ballon d'Or winner since Best in 1968 [KP] and the first Premier League player to be named the FIFA World Player of the Year [KP]
##CONTEXT##: United reached the final against Chelsea in Moscow on 21 May, where, despite his opening goal being negated by an equaliser and his penalty kick being saved in the shoot-out, United emerged victorious, winning 6–5 on penalties after a 1–1 draw at the end of 120 minutes. As the Champions League top scorer, Ronaldo was named the UEFA Club Footballer of the Year.
##KEYPOINTS_COUNT##: 2
##KEYPOINTS##: ["Cristiano Ronaldo became United's first Ballon d'Or winner since Best in 1968.", "Cristiano Ronaldo was the first Premier League player to be named the FIFA World Player of the Year."]
##QUESTION##: Who became Manchester United's first Ballon d'Or winner since George Best in 1968 and also made history as the first Premier League player to win the FIFA World Player of the Year award?

Example 2:
##CLAIM##: Heyman is the founder of Heyday Films [KP]
##CONTEXT##: David Heyman is a renowned film producer and founder of Heyday Films, known for producing the entire \
"Harry Potter" film series and collaborating with director Alfonso Cuarón on "Harry Potter and the Prisoner of Azkaban" \
and "Gravity". He was born on July 26, 1961, in London. His family has a background in the film industry, with his parents\
 being a producer and actress. Heyman studied Art History at Harvard University and began his career in the film industry\
  as a production assistant. Throughout his career, he has received numerous awards and nominations, including an Academy \
  Award nomination for Best Picture and a BAFTA Award for Best British Film.
##KEYPOINTS_COUNT##: 1
##KEYPOINTS##: ["David Heyman, the film producer, is the founder of Heyday Films."]
##QUESTION##: Who founded Heyday Films?
""",

    "user": """User input:
##ADDITIONAL INFORMATION##: The created date of the ##CLAIM## and ##CONTEXT## is [ADD_CREATED_DATE]. The current \
date—when the question is asked—is [ADD_QUESTION_DATE]. Use these to improve temporal clarity when needed.
##CLAIM##: [ADD_CLAIM_HERE]
##CONTEXT##: [ADD_CONTEXT_HERE]
##KEYPOINTS_COUNT##: [ADD_KEYPOINTS_COUNT_HERE]"""
}

QUESTION_GENERATION_PROMPT = {
    "system": """You are an AI responsible for generating a question given answer keypoints. The following are 5 examples

### Example 1:

**ANSWER KEYPOINTS**:
- Ronaldo was named runner-up to Kaká for the 2007 Ballon d'Or.
- Cristiano Ronaldo came third, behind Kaká and Lionel Messi, in the running for the 2007 FIFA World Player of the Year award.

**QUESTION**: In 2007, how did Cristiano Ronaldo place in the Ballon d'Or and the FIFA World Player of the Year rankings?

### Example 2:

**ANSWER KEYPOINTS**:
- "Ukulele in the Classroom", a revised program created by James Hill and Doane in 2008, is a staple of music education in Canada

**QUESTION**: What is the name of widely used ukulele-based music education program that was created by James Hill and Chalmers Doane in 2008?

### Example 3:

**ANSWER KEYPOINTS**:
- Ngenpa Gudzom, also known as 'The Meeting of Nine Evils,' is one of the festivals in Bhutan usually observed on the 7th day of the 11th month of the Bhutanese calendar.
- The traditional astrological period of Ngenpa Gudzom begins sometime during the afternoon of the 6th day of the 11th month of the Bhutanese calendar.

**QUESTION**: On which day of the Bhutanese calendar is the festival Ngenpa Gudzom celebrated, and when does its astrological period begin?

## Example 4:

**ANSWER KEYPOINTS**:
- The ongoing series Ghost Rider was canceled due to the COVID-19 pandemic.

**QUESTION**: Why was the ongoing Ghost Rider series canceled?

## Example 5:

**ANSWER KEYPOINTS**:
- New Orleans police officers returned fire during an incident involving Shamsud-Din Jabbar.

**QUESTION**: What did New Orleans police officers do during the incident involving Shamsud-Din Jabbar?""",

    "user": """Given the title of the document and a set of answer keypoints, generate a question that covers all keypoints naturally and fluently.

**TITLE**: [ADD_TITLE_HERE]

**ANSWER KEYPOINTS**:
[ADD_KEYPOINTS_HERE]

**QUESTION**:"""
}


GROUNDEDNESS_CHECK_PROMPT = {
    "system": """You are given a fact and a context document. Determine whether the fact is grounded in the context — that is, whether the fact is explicitly supported by the content of the context.
Output only one of the following labels:
+ Grounded – if the context explicitly supports or states the fact.
+ Not Grounded – if the context does not support the fact, or the relevant information is missing or irrelevant.

Instructions:
+ Do not infer or assume information beyond what is stated in the context.
+ Ignore metadata like publication date or document structure unless it contains meaningful content.
+ Output only the label (Grounded or Not Grounded) with no explanation.""",

    "user": """Fact: [ADD_KEYPOINT_HERE]
Context Language: [ADD_CONTEXT_LANGUAGE_HERE]
Context Published Date: [ADD_CONTEXT_PUBLISHED_DATE_HERE]
Context: [ADD_CONTEXT_HERE]

Output:"""
}
