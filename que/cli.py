import argparse
from que.store import DirectoryStore
from que.prompts import QUERY_SYSTEM_PROMPT
from que.models import make_llama, oneshot_query


def main_query(*args, **kwargs):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'query',
        help='The question to ask your files',
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
    is_scoped_local_search = args.local

    db = DirectoryStore(k=k, is_verbose=is_verbose)


    context = db.query(
        query,
        return_formatted_context=True,
        dir_scope=None if not is_scoped_local_search else '.'    
    )
    
    if is_query_only:
        print(context)
        exit()

    
    llm = make_llama(verbose=is_verbose, k=db.k, window_size=db.window_size)

    llm_response = oneshot_query(
        llm=llm,
        query=query,
        context=context,
        is_verbose=is_verbose
    )

    print()
    print(llm_response)