from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter


class MarkdownChunker:
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),

    ]

    @classmethod
    def split(cls, markdown_text: str | Path):
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=cls.headers_to_split_on)

        metadata = {
            'filename': '',
        }
        if isinstance(markdown_text, Path):
            metadata['filename'] = markdown_text.name
            markdown_text = markdown_text.read_text()

        docs = markdown_splitter.split_text(markdown_text)

        for index, doc in enumerate(docs, start=1):
            doc.metadata = {**doc.metadata, **
                            metadata, 'document_index': index}
        return docs
