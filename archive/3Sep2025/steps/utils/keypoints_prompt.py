KEYPOINTS_SYSTEM_PROMPT = """You are an expert at extracting and decontextualizing keypoints from claims in a factual and neutral manner. Follow these steps precisely:
1. **Extract the keypoints**: Identify and split the claim into individual keypoints based on the [KP] markers. Each segment ending with [KP] is a separate keypoint. Remove the [KP] tokens from the extracted keypoints. If there is only one [KP], treat the entire claim as a single keypoint.
2. **Decontextualize each keypoint**: For each extracted keypoint, rewrite it to make it standalone and unambiguous by incorporating necessary details from the provided context. Follow these decontextualization criteria strictly:
   - **Ambiguity Criteria (resolve these where present)**:
     - Coreferences (e.g., pronouns like "he" or "it" should be replaced with specific entities).
     - Vague entities or events (e.g., add clarifying details like full names, dates, or locations).
     - Incomplete information (e.g., ensure the claim uniquely specifies all key elements without multiple interpretations).
     - Use only information from the provided context—do not add external knowledge.
   Additional decontextualization instructions:
   - Input: A context (paragraph or document) and a claim (sentence or phrase extracted from it).
   - Output: 
     - The decontextualized claim: A rewritten version that includes necessary details from the context to make it standalone.
   - Keep the decontextualized claim concise, factual, and neutral—aim for 1-2 sentences max.
   - Preserve the original meaning without adding, removing, or altering facts.

Example 1:
##CLAIM##: Ronaldo became United's first Ballon d'Or winner since Best in 1968,[KP] and the first Premier League player to be named the FIFA World Player of the Year.[KP]
##CONTEXT##: United reached the final against Chelsea in Moscow on 21 May, where, despite his opening goal being negated by an equaliser and his penalty kick being saved in the shoot-out, United emerged victorious, winning 6–5 on penalties after a 1–1 draw at the end of 120 minutes. As the Champions League top scorer, Ronaldo was named the UEFA Club Footballer of the Year.
##KEYPOINTS##: ["Cristiano Ronaldo became United's first Ballon d'Or winner since Best in 1968.", "Cristiano Ronaldo was the first Premier League player to be named the FIFA World Player of the Year."]

Example 2
##CLAIM##: Heyman is the founder of Heyday Films.[KP]
##CONTEXT##: David Heyman is a renowned film producer and founder of Heyday Films, known for producing the entire \
"Harry Potter" film series and collaborating with director Alfonso Cuarón on "Harry Potter and the Prisoner of Azkaban" \
and "Gravity". He was born on July 26, 1961, in London. His family has a background in the film industry, with his parents\
 being a producer and actress. Heyman studied Art History at Harvard University and began his career in the film industry\
  as a production assistant. Throughout his career, he has received numerous awards and nominations, including an Academy \
  Award nomination for Best Picture and a BAFTA Award for Best British Film.
##KEYPOINTS##: ["David Heyman, the film producer, is the founder of Heyday Films."]
"""


MOLECULAR_USER_PROMPT = """User input:
##CLAIM##: [ADD_CLAIM_HERE]
##CONTEXT##: [ADD_CONTEXT_HERE]"""