import argparse
from que.store import DirectoryStore
from que.models import make_model, oneshot_session, continue_as_interactive_session
from que.config import QUECONFIG
from json import loads
from typing import Dict
from os import path
from pprint import pprint

def main_query(*args, **kwargs):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'query',
        help='The question to ask your files',
    )

    parser.add_argument(
        '-i',
        '--interactive',
        help='Query as an interactive chat session',
        action='store_true'
    )
    
    parser.add_argument(
        '-k',
        '--doc_chunks_k',
        help='The number of document chunks to use as context',
        default=QUECONFIG['documents']['n_documents_per_query'],
        type=int
    )

    parser.add_argument(
        '-qo',
        '--query_only',
        help='Disable LLM responses and return only the DB query',
        action='store_true'
    )

    parser.add_argument(
        '-l',
        '--local',
        help='Restrict the search to the current folder',
        action='store_true',
    )

    parser.add_argument(
        '-v',
        '--verbose',
        help='Enable verbosity',
        action='store_true',
        default=QUECONFIG['verbose']
    )


    args = parser.parse_args()

    query = args.query
    k = args.doc_chunks_k
    is_verbose = args.verbose
    is_query_only = args.query_only
    is_interactive = args.interactive

    err_msg_both_interactive_and_query_only_flags = f'Query only flag = {is_query_only} and Interactive flag = {is_interactive}, but only one is allowed'
    assert is_query_only ^ is_interactive or (is_query_only == is_interactive and not is_interactive), err_msg_both_interactive_and_query_only_flags
    
    if is_verbose:
        print('Loaded configuration:')
        pprint(QUECONFIG)

    db = DirectoryStore(
        chunk_size=QUECONFIG['documents']['chunk_size'],
        chunk_step=QUECONFIG['documents']['chunk_step'],
        is_verbose=is_verbose,
        st_embedding_model=QUECONFIG['documents']['embedding_model'],
    )

    is_scoped_local_search = args.local
    dir_scope = None if not is_scoped_local_search else '.'

    context = db.query(
        args.query,
        k,
        dir_scope=dir_scope
    )

    if is_query_only:
        print(db.format_context(context, QUECONFIG['prompts']['context_template']).replace( path.expanduser('~'), '~' ))
        exit()

    llm = make_model(
        QUECONFIG['model']['model_id'],
        QUECONFIG['model']['quant'],
        is_verbose=is_verbose
    )

    llm_response, messages = oneshot_session(
        llm=llm,
        query=query,
        query_system_prompt=QUECONFIG['prompts']['system_prompt'],
        context=db.format_context(context, QUECONFIG['prompts']['context_template']),
        is_verbose=is_verbose,
        continues=is_interactive
    )


    print(
        format_context_highlight(
            llm_response,
            context,
        )
    )
    
    if not is_interactive:
        exit()

    continue_as_interactive_session(
        llm,
        db,
        k,
        messages,
        QUECONFIG['prompts']['context_template'],
        is_verbose=is_verbose,
        print_hook=format_context_highlight,
        dir_scope=dir_scope
    )
    
def highlight(s):
    return "\x1b[1;32m" + s + "\x1b[0m"

def format_context_highlight(llm_response: Dict[str, str], context: Dict) -> str:

    llm_response = loads(llm_response)

    res = ''
    res += highlight(llm_response['answer']) + '\n\n'
    res += '-'*10 + f"confidence in answer:{llm_response['confidence_score']}" + '-'*10
    
    if not llm_response['found_an_answer']:
        return res
    res += '\n\n'

    for doc_text, meta in zip(context['documents'][0], context['metadatas'][0]):
        
        for fname, exact_snip in llm_response['source_snippets'].items():
            if fname == meta['source'] and exact_snip in doc_text:
                res += f"{meta['source'].replace( path.expanduser('~'), '~' )}:\n{doc_text}\n\n".replace(exact_snip, highlight(exact_snip))

    return res