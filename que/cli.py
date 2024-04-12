import argparse
from argparse import ArgumentParser
from que.store import DirectoryStore
from que.prompts import QUERY_SYSTEM_PROMPT
from que.models import make_llama, oneshot_query, continue_as_interactive_query


def main_query(*args, **kwargs):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'query',
        help='The question to ask your files',
    )

    parser.add_argument(
        '-i',
        '--interactive',
        help='Query as a chat session',
        action='store_true'
    )
    
    parser.add_argument(
        '-k',
        '--doc_chunks_k',
        help='The number of document chunks to use as context',
        default=3,
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
        action='store_true'
    )

    parser.add_argument(
        '-v',
        '--verbose',
        help='Enable verbosity',
        action='store_true'
    )


    args = parser.parse_args()

    query = args.query
    k = args.doc_chunks_k
    is_verbose = args.verbose
    is_query_only = args.query_only
    is_interactive = args.interactive

    err_msg_both_interactive_and_query_only_flags = f'Query only flag = {is_query_only} and Interactive flag = {is_interactive}, but only one is allowed'
    assert is_query_only ^ is_interactive or (is_query_only == is_interactive and not is_interactive), err_msg_both_interactive_and_query_only_flags
    
    db = DirectoryStore(k=k, is_verbose=is_verbose)

    is_scoped_local_search = args.local
    dir_scope = None if not is_scoped_local_search else '.'

    context = db.query(
        args.query,
        return_formatted_context=True,
        dir_scope=dir_scope
    )
    

    if is_query_only:
        print(context)
        exit()

    llm = make_llama(is_verbose=is_verbose, k=db.k, window_size=db.window_size)

    llm_response = oneshot_query(
        llm=llm,
        query=query,
        context=context,
        is_verbose=is_verbose,
        continues=is_interactive
    )

    print()
    print(llm_response[0] if is_interactive else llm_response)
    
    if not is_interactive:
        exit()

    _llm_txt_response, messages = llm_response
    continue_as_interactive_query(llm, db, messages, is_verbose=is_verbose, dir_scope=dir_scope)
    
    
