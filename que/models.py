from typing import List, Dict
from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
from pprint import pprint
from que.prompts import QUERY_SYSTEM_PROMPT

def make_llama(verbose: bool = False, k: int = 5, window_size: int = 100) -> Llama:
    model_id = "TheBloke/Llama-2-7B-Chat-GGUF"

    model = Llama.from_pretrained(
        repo_id=model_id,
        filename="*Q5_K_M.gguf",
        n_ctx= 8 * window_size * k,
        n_gpu_layers=-1,
        chat_format='llama-2',
        verbose=verbose,
        draft_model=LlamaPromptLookupDecoding(num_pred_tokens=10)
    )

    return model

def oneshot_query(llm: Llama, query: str, context: str, is_verbose: bool = False):

    messages = [
        {
          "role": "system", 
          "content": QUERY_SYSTEM_PROMPT.format(context=context)
        },
        {
          "role": "user",
          "content": query
        }
    ]

    if is_verbose:
        print()
        pprint(messages)
    
    llm_response = llm_do_chat(llm, messages, is_verbose=is_verbose)

    print()
    print(llm_response)

def llm_do_chat(llm: Llama, messages: List[Dict[str, str]], is_verbose: bool = False) -> str:
    llm_response = llm.create_chat_completion(
        messages=messages,
        max_tokens=None,
        stop=["Q:", "\n\n", '<|endoftext|>'],
    )

    if is_verbose:
        pprint(llm_response)

    llm_response = llm_response['choices'][0]['message']['content'].strip()
    return llm_response