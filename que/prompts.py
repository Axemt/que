QUERY_SYSTEM_PROMPT = """Answer the user query based on the source documents.

The source documents have the following format: "<|source|> : /path/to/source/file
...
"

Here are the source documents: {context}


You should provide your answer as a JSON blob, and also provide all relevant short source snippets from the documents on which you directly based your answer, and a confidence score as a float between 0 and 1. Do not use the file paths as snippets.
The source snippets should be very short, a few words at most, not whole sentences! And they MUST be extracted from the context, with the exact same wording and spelling.

Your answer should be built as follows

{{
  "answer": your_answer,
  "confidence_score": your_confidence_score,
  "found_an_answer": True | False, 
  "source_snippets": {{
    "/path/to/snippet1/file": "snippet1",
    "/path/to/snippet2/file": "snippet2",
    ...
  }}
}}
<|endoftext|>

If your answer satisfies the user's query, set "found_an_answer" to "False". If it does, set it to "True"
Now begin!
"""

FOLLOWUP_SYSTEM_PROMPT = """
We have the opportunity to further add to the user's queries with additional pieces of context.
You may recall or use previously provided pieces of context.
""" + "{context}"


CONTEXT_TEMPLATE = """
<|source|> : {fname}
{snippet}
"""