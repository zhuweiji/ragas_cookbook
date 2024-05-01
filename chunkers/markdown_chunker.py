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

        table_of_contents = cls.get_table_of_contents(markdown_text)
        metadata = {
            'filename': '',
            'table_of_contents': table_of_contents
        }

        if isinstance(markdown_text, Path):
            metadata['filename'] = markdown_text.name
            markdown_text = markdown_text.read_text()

        docs = markdown_splitter.split_text(markdown_text)

        for index, doc in enumerate(docs, start=1):
            doc.metadata = {**doc.metadata, **
                            metadata, 'document_index': index}
        return docs

    @classmethod
    def get_table_of_contents(cls, markdown_text: str | Path):
        """Get a table of contents from a markdown file by extracting all headers (#, ##, ...)."""

        headers = []
        if isinstance(markdown_text, Path):
            markdown_text = markdown_text.read_text()

        lines = markdown_text.splitlines()
        for line in lines:
            if line.startswith("#"):
                headers.append(line.strip())

        headers = '\n'.join(headers)
        return headers
