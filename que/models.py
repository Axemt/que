from typing import List, Dict, Tuple, Callable
from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
from pprint import pprint
from que.prompts import QUERY_SYSTEM_PROMPT, FOLLOWUP_SYSTEM_PROMPT
from que.store import DirectoryStore
from ast import literal_eval

def make_llama(
        is_verbose: bool = False,
        ctx_window_size: int = 16_384,
    ) -> Llama:
    """
    Return an instance of a LLama2 model

    Kwargs:
        is_verbose: Enable the model's verbose logging
        ctx_window_size: The size of the context window

    Returns:
        a LLama instance
    """
    model_id = "cognitivecomputations/dolphin-2.9-llama3-8b-gguf"#"TheBloke/Llama-2-7B-Chat-GGUF"

    model = Llama.from_pretrained(
        repo_id=model_id,
        filename="*q5_K_M.gguf",
        n_ctx=ctx_window_size,
        n_gpu_layers=-1,
        chat_format='chatml',
        verbose=is_verbose,
        draft_model=LlamaPromptLookupDecoding(num_pred_tokens=10)
    )

    return model

def oneshot_query(
    llm: Llama,
    query: str,
    context: str,
    is_verbose: bool = False,
    continues: bool = False
    ) -> str | Tuple[str, List[Dict[str, str]]]:
    """
    Performs a LLM text generation

    Args:
        llm: A LLama2 instance
        query: The user query
        context: The document chunks retrieved

    Kwargs:
        is_verbose: Enable verbose logging
        continues: If `True`, also return the message log given to the LLM for completion
    
    Returns:
        The generated LLM response, and the used message log if `continues=True`
    """

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
        print_hook: None | Callable = None,
        is_verbose: bool = False, 
        dir_scope: str | None = None
    ):
    """
    Continues an existing query as an interactive LLM chat

    Args:
        llm: A LLama2 instance
        db: The DirectoryStore instance, used for further context retrieval
        messages: The message log used for a first llm completion

    Kwargs:
        is_verbose: Enable verbose logging
        dir_scope: Restrict document retrieval to the specified directory
    """

    print()
    print('>> Continuing as chat session. Press Ctrl+C or Ctrl+D to exit')
    print('>> ----------------------------------------------------------')
    try:
        while True:
            followup_query = input('>> ')

            followup_context = db.query(
                followup_query,
                dir_scope=dir_scope
            )

            messages += [
                {
                    'role': 'system',
                    'content': db.format_context(followup_context)
                },
                {
                    'role': 'user',
                    'content': followup_query
                }
            ]

            llm_response = llm_do_chat(llm, messages, is_verbose=is_verbose)

            if print_hook is not None:
                llm_response = print_hook(llm_response, followup_context)

            print(llm_response)
            print()

    except (KeyboardInterrupt, EOFError):
        if is_verbose: print() # clear for llama-cpp logs
        pass


def llm_do_chat(
        llm: Llama, 
        messages: List[Dict[str, str]],
        is_verbose: bool = False
    ) -> str:
    """
    Generate text from the given messages

    Args:
        llm: A LLama2 instance
        messages: The message log for llm completion

    Kwargs:
        is_verbose: Enable verbose logging

    Returns:
        The generated llm text
    """

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