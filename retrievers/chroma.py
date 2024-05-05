
import logging
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_core.documents import Document

from config.constants import (
    chromadb_default_collection,
    embedding_model,
    llm_model_name,
)
from config.project_paths import chromadb_dir
from utilities import print_long_text
from utilities.document_utils import filter_for_matching_header_metadata

log = logging.getLogger(__name__)


class ExtendedChromaMarkdownRetriever:
    def __init__(self) -> None:
        self.persistent_client = chromadb.PersistentClient(
            path=str(chromadb_dir),
            settings=Settings(anonymized_telemetry=False))

        self.embedding_function = SentenceTransformerEmbeddings(
            model_name=embedding_model)

        self.langchain_chroma = Chroma(
            collection_name=chromadb_default_collection,
            client=self.persistent_client,
            embedding_function=self.embedding_function,
        )

        log.info(
            f'loaded local Chroma db from {chromadb_dir}, collection:{chromadb_default_collection}')

    def _get_relevant_documents(self, query: str, max_chunk_size: int = 1500, **kwargs):
        return self.get_all_related_chunks(query, max_size=max_chunk_size, **kwargs)

    def get_documents_by_filename(self, filename: str, headers: Optional[tuple | dict | List[tuple | dict]] = None):
        if isinstance(headers, tuple):
            _headers = {headers[0]: headers[1]}
        elif isinstance(headers, list) and headers and isinstance(headers[0], tuple):
            _headers = {k: v for k, v in headers}
        else:
            _headers: dict = headers  # type: ignore

        docs = self._get_documents(metadata_filter={'filename': filename})
        if headers:
            docs = filter_for_matching_header_metadata(docs, _headers)
        return docs

    def _search(self, query, **kwargs):
        # default to max marginal relevance search
        kwargs['search_type'] = kwargs.get('search_type', 'mmr')

        return self.langchain_chroma.search(query=query, **kwargs)

    def _get_documents(self, query='', metadata_filter=None, filter=None):
        docs_dict = self.langchain_chroma.get(
            ids=query, where=metadata_filter, where_document=filter)

        docs: list[Document] = []
        for page_content, metadata in zip(docs_dict['documents'], docs_dict['metadatas']):
            docs.append(Document(page_content=page_content, metadata=metadata))
        return docs

    def get_all_related_chunks(self, query, max_size: int = 1500, **kwargs):
        """search for matching documents for a header
        for each matching doc, find the relative chunks (as defined by highest-parent relationship)
        add chunks above and below the initial chunk until we reach max size"""
        LOOP_LIMIT = 100
        docs = self._search(query, **kwargs)

        for doc in docs:
            largest_header = self.get_biggest_header(doc)
            if not largest_header:
                continue

            largest_header_type, largest_header_value = largest_header
            related_docs = self._get_documents(metadata_filter={
                '$and': [
                    {largest_header_type: largest_header_value},
                    {'filename': doc.metadata.get('filename', -1)}
                ]
            }
            )

            if not related_docs:
                continue

            related_docs_ordered = sorted(related_docs,
                                          key=lambda x: x.metadata.get('document_index', -1))

            # find the index of the matched doc
            doc_index = doc.metadata.get(related_docs_ordered.index(doc))

            # TODO: error handling
            if not doc_index:
                continue

            above_ptr, below_ptr = doc_index - 1, doc_index + 1

            added_chunks = set()

            counter = 0
            # add related docs above and below the retrieved chunk until total size is too big
            while above_ptr > 0 or below_ptr < len(related_docs_ordered):
                # infinite loop sometimes
                # e.g (above_ptr=0, below_ptw=5),(above_ptr=0, below_ptw=5),(above_ptr=0, below_ptw=5)
                if counter > LOOP_LIMIT:
                    break
                counter += 1

                if len(added_chunks) >= len(related_docs_ordered):
                    break

                # add chunks that are above the retrieved chunk in the document
                if above_ptr > 0:

                    log.info(
                        f'{doc_index}, {above_ptr}, {len(related_docs_ordered)}')
                    above_chunk = related_docs_ordered[above_ptr]

                    if above_chunk.page_content not in added_chunks:
                        new_page_content = self.add_two_docs_page_content(
                            above_chunk, doc)

                        if len(new_page_content) > max_size:
                            break

                        above_ptr -= 1
                        added_chunks.add(above_chunk.page_content)
                        doc.page_content = new_page_content

                # add chunks that are below the retrieved chunk in the document
                if below_ptr < len(related_docs_ordered):
                    below_chunk = related_docs_ordered[below_ptr]

                    if below_chunk.page_content not in added_chunks:
                        new_page_content = self.add_two_docs_page_content(
                            doc, below_chunk)

                        if len(new_page_content) > max_size:
                            break

                        doc.page_content = new_page_content
                        added_chunks.add(below_chunk.page_content)
                        below_ptr += 1

        return docs

    def add_documents(self, docs, **kwargs):
        self.langchain_chroma.add_documents(docs, **kwargs)

    def __len__(self):
        return self.langchain_chroma._collection.count()

    @classmethod
    def get_document_headers(cls, doc: Document):
        headers = [(k, v)
                   for k, v in doc.metadata.items() if 'header' in k.lower()]
        return sorted(headers, key=lambda x: x[0].lower().replace('header ', ''))

    @classmethod
    def get_biggest_header(cls, doc: Document):
        headers = cls.get_document_headers(doc)
        return None if not headers else headers[0]

    @classmethod
    def get_smallest_header(cls, doc: Document):
        headers = cls.get_document_headers(doc)
        return None if not headers else headers[-1]

    @classmethod
    def add_two_docs_page_content(cls, main_doc: Document, other_doc: Document):
        header = cls.get_smallest_header(other_doc)

        header_value = ''
        if header:
            _, header_value = header

        text = f'{main_doc.page_content}\n\n'
        if header_value:
            text += f'**{header_value}**\n'

        text += other_doc.page_content
        return text


if __name__ == "__main__":
    import logging

    from tqdm import tqdm

    from chunkers.markdown_chunker import MarkdownChunker
    from config.project_paths import project_root, source_document_directory

    logging.basicConfig(
        format='%(name)s-%(levelname)s|%(lineno)d:  %(message)s', level=logging.INFO)
    log = logging.getLogger(__name__)

    documents = list(source_document_directory.glob('*.md'))
    log.info(
        f'ingesting {len(documents)} documents from source documents: {source_document_directory}')

    retriever = ExtendedChromaMarkdownRetriever()
    for i in tqdm(documents):
        try:
            docs = MarkdownChunker.split(i)
            retriever.add_documents(docs)
        except Exception as e:
            log.error(f'Could not add document {i.name}. Exception: {e}')
