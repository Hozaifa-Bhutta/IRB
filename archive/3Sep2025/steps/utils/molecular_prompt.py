MOLECULAR_SYSTEM_PROMPT = """\
DECONTEXTUALIZATION CRITERIA: Decontextualization adds the right type of information to a CLAIM to make it standalone \
and contain relevant disambiguating information. This process can modify the original CLAIM in the following manners:
- Substituting pronouns or incomplete names with the specific subject being referred to.
- Incorporating the most important distinguishing details such as location/profession/time period to distinguish the subject from others who might share similar names.
- Should not omit information from the original CLAIM.
- Should only minimally modify the original CLAIM.

Instructions:\
- Substitute any incomplete names or pronouns in the CLAIM.
- If needed, add clarification phrases about the subject to the claim.
- If no disambiguation is necessary, made no change to the CLAIM

Example 1:
##CLAIM##: He is best known for hosting CBS News Sunday Morning.
##CONTEXT##: Charles Osgood, a renowned American radio and television commentator and writer, was born on January 8, 1933, in the Bronx, \
New York City. He is best known for hosting "CBS News Sunday Morning" for over 22 years and "The Osgood File" radio commentaries for over 40 years. \
Osgood also authored several books, including "The Osgood Files", "See You on the Radio", and "Defending Baltimore Against Enemy Attack". He was born \
to Charles Osgood Wood, III and his wife Jean Crafton, and grew up with five siblings. Osgood graduated from Fordham University in 1954 with a bachelor \
of science degree in economics.
##DECONTEXTUALIZED CLAIM##: Charles Osgood, the television commentator, is best known for hosting CBS News Sunday Morning.

Example 2:
##CLAIM##: She served as a Supreme Court justice.
##CONTEXT##: Ruth Bader Ginsburg (March 15, 1933 – September 18, 2020) was an American lawyer and jurist who became an \
associate justice of the Supreme Court of the United States in 1993. Appointed by President Bill Clinton, she served on the \
Court until her death in 2020 and was celebrated for her advocacy for gender equality and civil liberties.
##DECONTEXTUALIZED CLAIM##: Ruth Bader Ginsburg served as a Supreme Court justice.

Example 3:
##CLAIM##: Heyman is the founder of Heyday Films.
##CONTEXT##: David Heyman is a renowned film producer and founder of Heyday Films, known for producing the entire \
"Harry Potter" film series and collaborating with director Alfonso Cuarón on "Harry Potter and the Prisoner of Azkaban" \
and "Gravity". He was born on July 26, 1961, in London. His family has a background in the film industry, with his parents\
 being a producer and actress. Heyman studied Art History at Harvard University and began his career in the film industry\
  as a production assistant. Throughout his career, he has received numerous awards and nominations, including an Academy \
  Award nomination for Best Picture and a BAFTA Award for Best British Film.
##DECONTEXTUALIZED CLAIM##: David Heyman, the film producer, is the founder of Heyday Films.

Now generate a DECONTEXTUALIZED CLAIM for the following. Ensure the DECONTEXTUALIZED \
CLAIM adds minimal information to resolve ambiguity, such as adjusting pronouns or including clarifying details \
about the SUBJECT. The DECONTEXTUALIZED CLAIM should be coherent and must avoid repetition. \
It should retain the same structural format as the original CLAIM."
"""

MOLECULAR_SYSTEM_PROMPT="""You are an expert in decontextualizing claims for fact-checking and information retrieval tasks. Your goal is to rewrite a given claim so it is fully self-contained, unambiguous, and understandable without relying on external context beyond what's provided.

Ambiguity Criteria (resolve these where present):
- Coreferences (e.g., pronouns like "he" or "it" should be replaced with specific entities).
- Vague entities or events (e.g., add clarifying details like full names, dates, or locations).
- Incomplete information (e.g., ensure the claim uniquely specifies all key elements without multiple interpretations).
- Use only information from the provided context—do not add external knowledge.

Instructions:
- Input: A context (paragraph or document) and a claim (sentence or phrase extracted from it).
- Output: 
  - The decontextualized claim: A rewritten version that includes necessary details from the context to make it standalone.
- Keep the decontextualized claim concise, factual, and neutral—aim for 1-2 sentences max.
- Preserve the original meaning without adding, removing, or altering facts.


Example 1:
##CLAIM##: Bird is scrapping thousands of e-scooters in the Middle East.
##CONTEXT##: California scooter sharing start-up Bird is facing financial difficulties. Bird is scrapping thousands of e-scooters in the Middle East due to regulatory issues.
##DECONTEXTUALIZED CLAIM##: California scooter sharing start-up Bird is scrapping thousands of e-scooters in the Middle East due to regulatory issues.


Example 2:
##CLAIM##: This mission carried astronauts to the International Space Station for the first time from U.S. soil since 2011.
##CONTEXT##: Elon Musk's company SpaceX launched the Falcon 9 rocket successfully on May 30, 2020. This mission carried astronauts to the International Space Station for the first time from U.S. soil since 2011.
##DECONTEXTUALIZED CLAIM##: Elon Musk's company SpaceX's Falcon 9 rocket mission on May 30, 2020, carried astronauts to the International Space Station for the first time from U.S. soil since 2011.


Example 3:
##CLAIM##: It reported over 10,000 square kilometers lost that year.
##CONTEXT##: The Amazon rainforest experienced record deforestation in 2022 due to illegal logging and agriculture expansion. Brazil, which contains the largest portion of the rainforest, reported over 10,000 square kilometers lost that year.
##DECONTEXTUALIZED CLAIM##: Brazil reported over 10,000 square kilometers of Amazon rainforest lost in 2022 due to illegal logging and agriculture expansion.
"""


MOLECULAR_USER_PROMPT = """User input:
##CLAIM##: [ADD_CLAIM_HERE]
##CONTEXT##: [ADD_CONTEXT_HERE]"""