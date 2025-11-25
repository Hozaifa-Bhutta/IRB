QA_SYSTEM_PROMPT = f"""Answer the provided question given the retrieved contexts. You are also provided the time when the question was asked.
Write your answer directly without any explanation. It is important that your answer is in short form.
Note that the retrieved context can be in other languages than English. However, you must answer in English.
If you feel that you cannot answer given the provided information, just say 'I cannot answer because there are not enough information in the provided context'"""

QA_USER_PROMPT = f"""Retrieved Contexts: 
[ADD CONTEXT HERE]

Question date: [ADD_QUESTION_DATE] UTC
Question: [ADD QUESTION HERE]"""


QA_SYSTEM_PROMPT_WITHOUT_CONTEXT = f"""Answer the provided question. You are also provided the time when the question was asked.
Write your answer directly without any explanation. 
It is important that your answer is in short form. 
If you don't think you know the answer, do not guess, just say 'I don't know'"""

QA_USER_PROMPT_WITHOUT_CONTEXT = f"""
Question date: [ADD_QUESTION_DATE] UTC
Question: [ADD QUESTION HERE]"""