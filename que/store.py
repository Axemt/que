from typing import List, Tuple, Dict, Callable
import chromadb
import glob
import os
import hashlib
import chromadb.utils
import chromadb.utils.embedding_functions
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as read_pdf_file
from pdfminer.pdfdocument import PDFTextExtractionNotAllowed
from pdfminer.psparser import PSSyntaxError
from warnings import warn
import docx
import torch

class DirectoryStore:


    def __init__(
            self,
            chunk_size: int,
            chunk_step: int,
            st_embedding_model: str,
            is_verbose: bool = False,
        ) -> None:
        
        self.config_folder =  os.path.expanduser('~') + '/.config/que'
        self.findex_name = f'{self.config_folder}/index.chroma'

        self.v = is_verbose

        self.chunk_step = chunk_step
        self.chunk_size = chunk_size
        self.embedding_model = st_embedding_model

        device = 'cuda' if torch.cuda.is_available() else (
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )
        self.collection = chromadb.PersistentClient(path=self.findex_name).get_or_create_collection(
            name='tomes',
            embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model,
                device=device
            )
        )

        # update with new data
        if self.v: print('Updating fmap db with current directory...')
        fmap_in_current_dir = self.explore_current_dir()

        # purge from non-existent/non-updated files and their tomes
        if self.v: print('Updating db to current filesystem state...')
        self.db_update_to_current_files(files_in_dir=fmap_in_current_dir)


    def db_update_to_current_files(self, files_in_dir: List[str] = []):
        """
        Updates the DB to the current state of the file system

        The method deletes DB entries from files that no longer exist or have been modified since last reindexing,
        then adds entries with the current file as source

        Kwargs:
            files_in_dir: A list of new files found to be added to the DB if they were not present already
        """

        metadatas = self.collection.get(include=['metadatas'])['metadatas']

        file_no_longer_exists_fnames = []
        fingerprint_changed_fnames = []
        fingerprint_changed_fprints = []


        for doc_meta in metadatas:

            abs_fname = doc_meta['source']

            if abs_fname in file_no_longer_exists_fnames or abs_fname in fingerprint_changed_fnames:
                # already processed
                continue


            if os.path.exists(abs_fname):
                current_fingerprint = self.get_file_fingerprint(abs_fname, self.embedding_model)
                stored_fingerprint = doc_meta['fingerprint']

                if current_fingerprint != stored_fingerprint:
                    if self.v: 
                        print(f'\tFile fingerprint changed: {abs_fname}')
                        print(f'\t\tExpected fprint {stored_fingerprint}. Got {current_fingerprint}')
                    fingerprint_changed_fnames.append(abs_fname)
                    fingerprint_changed_fprints.append(current_fingerprint)


            else:
                if self.v: print(f'\tFile no longer exists: {abs_fname}')
                file_no_longer_exists_fnames.append(abs_fname)

        # 1. Purge non existent entries - Delete if the file does not exist
        #    Also delete fprint changed. Instead of an upsert which may leave chunks that have
        #     been removed from the document
        deletes = file_no_longer_exists_fnames + fingerprint_changed_fnames

        if len(deletes) > 0:
            if self.v: print(f'\nRemoving {len(deletes)} altered/deleted files. This might take a while...')            
            self.collection.delete(
                where={
                    'source': {
                        '$in': file_no_longer_exists_fnames + fingerprint_changed_fnames
                    }
                }
            )


        # 2. Update if the file exists but the fingerprint changed
        ids = []
        documents = []
        metadatas = []

        for abs_fname, fingerprint in zip(fingerprint_changed_fnames, fingerprint_changed_fprints):
            
            entries_or_err = self.try_prepare_entry(abs_fname, fingerprint=fingerprint)

            if entries_or_err == False:
                continue
            
            if self.v: print(f'\tUpdating changed file: {abs_fname}')
            file_doc_ids, file_metadatas, file_txt_chunks = entries_or_err


            ids += file_doc_ids
            metadatas += file_metadatas
            documents += file_txt_chunks

        # add additional files in current dir, if any


        indexed_files = set([ meta['source'] for meta in self.collection.get(include=['metadatas'])['metadatas'] ])
        


        new_files = set(filter(
            lambda fname: fname not in indexed_files and not fname in fingerprint_changed_fnames,
            files_in_dir
        ))

        for abs_fname in new_files:
            entries_or_err = self.try_prepare_entry(abs_fname)

            if entries_or_err == False:
                continue
            
            if self.v: print(f'\tAdding new file: {abs_fname}')
            file_doc_ids, file_metadatas, file_txt_chunks = entries_or_err


            ids += file_doc_ids
            metadatas += file_metadatas
            documents += file_txt_chunks


        if len(ids) > 0:
            if self.v or len(ids) > 1_000: print(f'\nAdding {len(ids)} text snippets to DB. This might take a while...')
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )



    def explore_current_dir(self, recursive: bool = True) -> List[str]:
        """
        Explore the current directory for indexable files

        Kwargs:
            recursive: Enable recursive traversal

        Returns:
            A list of the discovered indexable files in the current directory
        """
        fmap = []

        files = [
            glob.iglob('**/*.md', recursive=recursive) ,
            glob.iglob('**/*.txt', recursive=recursive),
            glob.iglob('**/*.pdf', recursive=recursive),
            glob.iglob('**/*.docx', recursive=recursive),
            glob.iglob('**/*.epub', recursive=recursive),
        ]
        
        if self.v: print(f'Exploring directory {os.path.abspath(".")}')
        for file_group in files:
            for fname in file_group:
                abs_fname = os.path.abspath(fname)
                if self.v: print('\tFound', abs_fname)
                fmap.append(abs_fname)

        return fmap
    
    def try_prepare_entry(
        self,
          abs_fname: str | os.PathLike, 
          fingerprint: str | None = None,
        ) -> bool | Tuple[List[str], List[Dict[str, str]], List[str]]:
        """
        Prepare a file to insert in the DB store

        Args:
            abs_fname: The absolute name of the file to insert

        Kwargs:
            fingerprint: If previously computed, bypass obtaining the fingerprint and use `fingerprint` instead

        Returns:
            A tuple of the form `ids, metadatas, documents` containing an entry for every document chunk in the file or `False` if the file was empty or could not be read
        """

        txt = read_file(abs_fname).strip()
        if txt == '': return False

        txt_chunks = self.chunkify(txt)

        ids = [ f'{abs_fname}-chk-{i}' for i in range(len(txt_chunks))]

        if fingerprint is None: fingerprint = self.get_file_fingerprint(abs_fname, self.embedding_model)
        metadatas = [ 
            {
                'source': abs_fname,
                'fingerprint': fingerprint,
            } 
        ] * len(txt_chunks)
        
        documents = txt_chunks

        return ids, metadatas, documents


    def chunkify(
        self,
        document_txt: str, 
        ) -> List[str]:
        """
        Convert a string into text chunks

        Args:
            document_txt: The string to chunkify

        Returns
            A list with the text chunks
        """

        text_tokens = document_txt.split()


        sentences = []
        for i in range(0 , len(text_tokens) , self.chunk_step):
            sentence = text_tokens[i : i + self.chunk_size]
            
            if (len(sentence) < self.chunk_size):
                sentences.append(sentence)
                break

            sentences.append(sentence)

        paragraphs = [" ".join(s) for s in sentences]
        return paragraphs
    
    def query(
        self,
        query_txt: str,
        k: int,
        dir_scope: str = None
        ) -> str | Dict:
        """
        Query the DB for related documents to the query

        Args:
            query_txt: The query, in text form
            k: The number of documents to retrieve

        Kwargs: 
            dir_scope: Restrict the query to documents within the `dir_scope` folder.
        Returns:
            Raw DB query results
        """
        if self.v:
            print(f'Querying DB with {self.collection.count()} text snippets...')

        if dir_scope is not None:
            dir_scope = os.path.abspath(dir_scope)
            if self.v: print(f'Scoped search enabled: restricting to {dir_scope}')

            # Unfortunately chroma does not support $contains where queries in metadata fields, only in contents

            files_in_scope = [
                meta['source']
                for meta in filter(
                    # filter cannot modify the collection or apply anything :(
                    lambda meta: dir_scope in meta['source'],
                    # and since `metadatas` is a dict, I cannot get just the source of all docs
                    self.collection.get(include=['metadatas'])['metadatas']
                )
            ]

            dir_scope = {
                "source": {
                    "$in": files_in_scope
                }
            }

        results = self.collection.query(
            query_texts=query_txt,
            n_results=k,
            include=['documents', 'metadatas'],
            where=dir_scope
        )

        return results

    def format_context(self, context: Dict[str, str], context_template: str):
        """
        Format `CONTEXT_TEMPLATE` using the provided snippet and file name

        Args:
            context: The context documents
            context_template: A format string with fields `fname` and `snippet`

        Returns:
            The formatted `context_template` string
        """

            
        res = ''
        for snippet, meta in zip(context['documents'][0], context['metadatas'][0]):
            
            res += context_template.format(fname=meta['source'], snippet=snippet.strip())

        return res
    
    @staticmethod
    def get_file_fingerprint(
        abs_fname: str | os.PathLike,
        built_with: str,
        hard_digest: bool = False
        ) -> str:
        """
        Calculate a fingerprint of the file contents

        Args:
            abs_fname: The absolute path of the file to fingerprint

        Kwargs:
            hard_digest: Hash the file instead of the file metadata
        
        Returns:
            The file fingerprint
        """

        if hard_digest:
            with open(abs_fname, 'rb') as f:
                f_info = f.read()
        else:
            fstat = os.stat(abs_fname)

            f_info = f'{abs_fname}<//>{fstat.st_size}-{fstat.st_mtime}-{fstat.st_ctime}'.encode('utf-8')
        
        return hashlib.md5(f_info).hexdigest()

        
def read_file(abs_fname: str | os.PathLike) -> str:
    """
    Read a file using an aprorpiate file reader

    Args:
        abs_fname: The absolute path of the file to read

    Returns:
        The text content of the file
    """

    doctype = abs_fname[ abs_fname.rfind('.')+1: ]

    # FIXME: Make some sort of supported decoder API that has the types of files accepted, their decoder, etc
    if doctype in ['txt', 'md']:
        return read_raw_text_file(abs_fname)
    elif doctype == 'pdf':
        try:
            return read_pdf_file(abs_fname)
        except PDFTextExtractionNotAllowed:
            warn(f'The pdf file {abs_fname} is locked for reading')
        except PSSyntaxError as e:
            warn(f'The pdf file {abs_fname} has invalid or wonky encoding: {str(e)}')
    elif doctype == 'docx':
        return read_docx_file(abs_fname)
    elif doctype == 'epub':
        return read_epub_file(abs_fname)
    else:
        warn(f'doctype did not match: doctype={doctype}')
    return ''
    
    
def read_raw_text_file(abs_fname: str | os.PathLike) -> str:
    """
    Read a file as raw text

    Args:
        abs_fname: The absolute path of the file to read

    Returns:
        The text content of the file
    """
    with open(abs_fname, 'r') as f:
        try:
            raw_txt = f.read()
        except UnicodeDecodeError as e:
            warn(f'File {abs_fname} has invalid or wonky encoding: {str(e)}')
            return ''
    return raw_txt

def read_docx_file(abs_fname: str | os.PathLike) -> str:
    """
    Read a file as a .docx document

    Args:
        abs_fname: The absolute path of the file to read

    Returns:
        The text content of the file
    """
    doc = docx.Document(abs_fname)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def read_epub_file(abs_fname: str | os.PathLike) -> str:


    book = epub.read_epub(abs_fname)
    content = ''
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            body_content = item.get_body_content().decode()
            content += BeautifulSoup(body_content).get_text().strip()

    return content