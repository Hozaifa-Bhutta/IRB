KEYPOINT_EXTRACTION_PROMPT = {
    "system": """You are an expert at extracting and decontextualizing keypoints from claims in a factual and neutral manner. Follow these steps precisely:
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

Example 1:
##CLAIM##: Ronaldo became United's first Ballon d'Or winner since Best in 1968 [KP] and the first Premier League player to be named the FIFA World Player of the Year [KP]
##CONTEXT##: United reached the final against Chelsea in Moscow on 21 May, where, despite his opening goal being negated by an equaliser and his penalty kick being saved in the shoot-out, United emerged victorious, winning 6–5 on penalties after a 1–1 draw at the end of 120 minutes. As the Champions League top scorer, Ronaldo was named the UEFA Club Footballer of the Year.
##KEYPOINTS_COUNT##: 2
##KEYPOINTS##: ["Cristiano Ronaldo became United's first Ballon d'Or winner since Best in 1968.", "Cristiano Ronaldo was the first Premier League player to be named the FIFA World Player of the Year."]

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
""",

    "user": """User input:
##ADDITIONAL INFORMATION##: ##CLAIM## and ##CONTEXT## were last updated in [ADD_LAST_UPDATED_DATE] (YYYY-mm-dd). Use this information to improve temporal clarity when needed.
##CLAIM##: [ADD_CLAIM_HERE]
##CONTEXT##: [ADD_CONTEXT_HERE]
##KEYPOINTS_COUNT##: [ADD_KEYPOINTS_COUNT_HERE]"""
}

QUESTION_GENERATION_PROMPT = {
    "system": """You are an AI responsible for generating a well formed question given answer keypoints. A well-formed question needs to satisfy the following requirements
+ Fluent, grammatically correct
+ Do not leak the answer in the question
    
The following are 5 examples

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

##ADDITIONAL INFORMATION##: The information in **ANSWER KEYPOINTS** is last updated in [ADD_LAST_UPDATED_DATE] (YYYY-mm-dd). Use this information to improve temporal clarity when needed. \
In addition, the title of the document from which the keypoints are extracted is: [ADD_TITLE_HERE]; use this additional information if needed.

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



GRAPH_BUILDER_PROMPT = {
    "system": """You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph. \
        Your task is to identify the entities and relations requested with the user prompt from a given text. You must generate the \
            output in a JSON format containing a list with JSON objects. Each object should have the keys: "head", "head_type", "relation", \
                "tail", "tail_type".
                        
Attempt to extract as all entities and relations.

Maintain Entity Consistency: When extracting entities, it's vital to ensure consistency. \
If a entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"), always \
use the most complete identifier for that entity. The knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.

IMPORTANT NOTES:
- Don't add any explanation and text. For the following text, extract entities and relations

Example 1:
Text: Cristiano Ronaldo made his La Liga debut against Deportivo La Coruña on 29 August, scoring a penalty in a 3–2 home win.
Knowledge Graph: 
```json
[
    {
        "head": "Cristiano Ronaldo",
        "head_type": "Person",
        "relation": "made his debut at",
        "tail": "La Liga",
        "tail_type": "Tournament"
    },
    {
        "head": "Cristiano Ronaldo",
        "head_type": "Person",
        "relation": "made his debut against",
        "tail": "Deportivo La Coruña",
        "tail_type": "Soccer team"
    },
    {
        "head": "Cristiano Ronaldo",
        "head_type": "Person",
        "relation": "made his debut on",
        "tail": "29 August",
        "tail_type": "Date"
    },
    {
        "head": "Cristiano Ronaldo",
        "head_type": "Person",
        "relation": "scored a penalty in",
        "tail": "A 3-2 home win",
        "tail_type": "Event"
    }
]
```


Example 2:
Text: The idea of using computers to search for relevant pieces of information was popularized in the article As We May Think by Vannevar Bush in 1945.
Knowledge Graph:
```json
[
    {
        "head": "The idea of using computers to search for relevant pieces of information",
        "head_type": "Scientific idea",
        "relation": "was popularized in",
        "tail": "As We May Think",
        "tail_type": "Article"
    },
    {
        "head": "As We May Think",
        "head_type": "Article",
        "relation": "was authored by",
        "tail": "Vannevar Bush",
        "tail_type": "Person"
    },
    {
        "head": "As We May Think",
        "head_type": "Article",
        "relation": "was authored in",
        "tail": "1945",
        "tail_type": "Year"
    }
]
```

Example 3:
Text: In Donald Trump's inaugural address, he pledged to "immediately begin the overhaul of our trade system to protect American workers and families."
Knowledge Graph:
```json
[
    {
        "head": "Donald Trump",
        "head_type": "Person",
        "relation": "Pledged in his inaugural address to",
        "tail": "immediately begin the overhaul of our trade system to protect American workers and families.",
        "tail_type": "Quote"
    },
]
```
""",
    "user": "Text: [ADD_KEYPOINTS_HERE]"
}


QUESTION_GENERATION_PROMPT_FROM_KG = {
    "system": """You are an expert AI assistant specializing in Natural Language Generation. Your task is to generate a sequence of progressively more specific questions based on a list of structured relations.
Core Instruction:
The formulation of a question begins by identifying an <Unknown> entity in the relations list. This "unknown" entity is the target of the question.

Your process must be as follows:
Identify the Target: Locate a relation that contains an <Unknown> entity. The type of this entity (e.g., Organization, Event) will determine the question's focus (e.g., "Which organization...?", "What event...?").
Form the Base Question: Use the information from a single, core relation involving the <Unknown> entity to form the first question.
Incrementally Add Detail: For each subsequent step, integrate information from exactly one of the remaining relations to make the question progressively more specific. The total number of Question generation steps must be equal to the total number of Relations.

IMPORTANT NOTE: Do not include the special token <Unknown> in the generated question.

Examples
Example 1
Relations: (subject [subject type] | relation | object [object type])
1. <Unknown> #1 [event] | occurred on | 2 January 2023 [date]
2. <Unknown> #1 [event] | occurred at time | 13:59 AEST [time]
3. <Unknown> #1 [event] | occurred near | <Unknown> #2 [location]
4. <Unknown> #2 [location] | located in | Gold Coast [city]
5. Gold Coast [city] | located in | Queensland [region]
6. Queensland [region] | located in | Australia [country]

Question generation steps: (5 steps)
1. What event occurred on 2 January 2023?
2. What event occurred at 13:59 AEST on 2 January 2023?
3. What event occurred at 13:59 AEST on 2 January 2023 near a specific location?
4. What event occurred at 13:59 AEST on 2 January 2023 near a location in Gold Coast?
5. What event occurred at 13:59 AEST on 2 January 2023 near a location in Gold Coast, Queensland?
6. What event occurred at 13:59 AEST on 2 January 2023 near a location in Gold Coast, Queensland, Australia?



Example 2
Relations: (subject [subject type] | relation | object [object type])
1. <Unknown> #1 [Person] | exceeded his authority by imposing | fentanyl tariffs [Tariff]
2. United States Court of International Trade [Court] | ruled that | <Unknown> #1 [Person]
3. United States Court of International Trade [Court] | ruled on | May 28 [Date]
4. <Unknown> #1 [Person] | exceeded his authority by imposing | reciprocal tariffs [Tariff]

Question generation steps: (4 steps)
1. Who exceeded his authority by imposing fentanyl tariffs?
2. Who was ruled by the United States Court of International Trade that he exceeded his authority by imposing fentanyl tariffs?
3. Who was ruled by the United States Court of International Trade on May 28 that he exceeded his authority by imposing fentanyl tariffs?
4. Who was ruled by the United States Court of International Trade on May 28 that he exceeded his authority by imposing fentanyl tariffs and reciprocal tariffs?
""",
    "user": ""
}