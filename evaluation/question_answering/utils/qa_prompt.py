QA_SYSTEM_PROMPT = f"""Answer the provided question given the retrieved contexts. You are also provided the time when the question was asked.
The retrieved contexts may or may not help answer the question. Your task is to answer the question in as few words as possible.

Please follow these guidelines when formulating your answer:
1. If the question contains a false premise or assumption, answer "False premise question".
2. If you are uncertain or don’t know the answer, respond with "I don’t know".
3. Note that the retrieved context can be in other languages than English. However, you must answer in English."""

QA_USER_PROMPT = f"""Retrieved Contexts: 
[ADD CONTEXT HERE]

Question date: [ADD_QUESTION_DATE] UTC (YYYY-MM-DD)
Question: [ADD QUESTION HERE]"""


QA_SYSTEM_PROMPT_WITHOUT_CONTEXT = f"""Answer the provided question. You are also provided the time when the question was asked.
Your task is to answer the question in as few words as possible.

Please follow these guidelines when formulating your answer:
1. If the question contains a false premise or assumption, answer "False premise question".
2. If you are uncertain or don’t know the answer, respond with "I don’t know"."""

QA_USER_PROMPT_WITHOUT_CONTEXT = f"""
Question date: [ADD_QUESTION_DATE] UTC (YYYY-MM-DD)
Question: [ADD QUESTION HERE]"""