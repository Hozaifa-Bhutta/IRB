QUESTION_GENERATION_SYSTEM_PROMPT = f"""You are an AI agent that accurately generate questions. 
Given a molecular fact (a fact that is understandable stand-alone), generate ONE question whose answer IS THE PROVIDED MOLECULAR FACT. 
The generated question should look like a typical query that people use to search on the internet. Directly generate a question without any explanation"""


QUESTION_GENERATION_USER_PROMPT = """##MOLECULAR FACT:##: [ADD_FACT_HERE]"""