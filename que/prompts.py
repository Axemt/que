QUERY_SYSTEM_PROMPT = """
Use the following pieces of context to answer the question asked at the END.
There are context paragraphs marked by <|source|>
Ignore the filepath following "<|source|> :"
Give only an answer to the specific user request and nothing else. 
If a context paragraph does not help answer the question, or is unrelated, ignore it.
If the question is not explicit, give information about the topics in the question from the context.
Do not give information about topics unrelated to the question.
Don't try to make up an answer. Don't make up new terms and try to be precise. 
You can add context if you are very very sure about your information.
{context}"""

CONTEXT_TEMPLATE = """
<|source|> : {fname}
{snippet}
"""