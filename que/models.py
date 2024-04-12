from typing import List, Dict
from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
from pprint import pprint
from que.prompts import QUERY_SYSTEM_PROMPT, FOLLOWUP_SYSTEM_PROMPT
from que.store import DirectoryStore

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

def oneshot_query(llm: Llama, query: str, context: str, is_verbose: bool = False, continues: bool = False):

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
    
    llm_response = llm_do_chat(llm, messages, is_verbose=is_verbose)

    return llm_response if not continues else (llm_response, messages)


def continue_as_interactive_query(
        llm: Llama,
        db: DirectoryStore,
        messages: List[Dict[str, str]],
        is_verbose: bool = False, 
        dir_scope: str | None = None):

    print()
    print('>>Continuing as chat session. Press Ctrl+C or Ctrl+D to exit')
    print('>>----------------------------------------------------------')
    try:
        while True:
            followup_query = input('>>')

            followup_context = db.query(
                followup_query,
                return_formatted_context=True,
                dir_scope=dir_scope
            )

            messages += [
                {
                    'role': 'system',
                    'content': FOLLOWUP_SYSTEM_PROMPT.format(context=followup_context)
                },
                {
                    'role': 'user',
                    'content': followup_query
                }
            ]

            llm_response = llm_do_chat(llm, messages, is_verbose=is_verbose)
            print()
            print(llm_response)
            print()

    except (KeyboardInterrupt, EOFError):
        if is_verbose: print() # clear for llama-cpp logs
        pass


def llm_do_chat(llm: Llama, messages: List[Dict[str, str]], is_verbose: bool = False) -> str:

    if is_verbose:
        pprint(messages)

    llm_response = llm.create_chat_completion(
        messages=messages,
        max_tokens=None,
        stop=["Q:", "\n\n", '<|endoftext|>'],
    )

    if is_verbose:
        pprint(llm_response)
        print()

    llm_response = llm_response['choices'][0]['message']['content'].strip()
    return llm_response