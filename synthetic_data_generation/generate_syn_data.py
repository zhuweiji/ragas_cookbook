import csv
import logging
import random
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from chunkers.markdown_chunker import MarkdownChunker
from config.project_paths import project_root, source_document_directory
from retrievers.chroma import ExtendedChromaMarkdownRetriever
from synthetic_data_generation.rag_data import DataGenerator

log = logging.getLogger(__name__)


def generate_synthetic_test_data(md_file: Path, num_questions=5, output_dir: Path = project_root / 'synthetic_data_generation' / 'data', **kwargs):
    try:
        text = md_file.read_text(errors='ignore')
    except Exception as e:
        log.error(e)
        return

    toc = MarkdownChunker.get_table_of_contents(md_file)

    chunk_size = kwargs.get('chunk_size', 6000)
    chunk_overlap = kwargs.get('chunk_overlap', 400)

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    docs = text_splitter.create_documents([text])

    generator = DataGenerator()

    results = []
    for i in tqdm(docs, desc='Generating Questions from Documents:'):
        q_a = generator.generate_questions_from_document(i, num_questions)
        print(q_a)
        results.append(q_a)

    filepath = output_dir / f'{md_file.name}__raw.txt'
    filepath.parent.mkdir(parents=True, exist_ok=True)

    t = ''
    for i in results:
        t += i.content
        t += '\n\n'

    filepath.write_text(t)

    qa_tuples = generator.extract_qa_tuples(t)
    filepath = output_dir / f'{md_file.stem}.csv'

    list_to_csv(qa_tuples, filepath)


def list_to_csv(data, filename):
    """
    Writes a list of tuples to a CSV file.

    Args:
        data: A list of tuples, where each tuple represents a row in the CSV file.
        filename: The name of the CSV file to write to.
    """
    with open(filename, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


def csv_to_list(filename, has_header=True):
    """
    Reads a CSV file into a list of lists.

    Args:
        filename: The path to the CSV file.
        has_header (bool, optional): Specifies if the CSV file has a header row. Defaults to True.

    Returns:
        list: A list of lists, where each inner list represents a row in the CSV file.
    """
    data = []
    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        # Skip header row if it exists
        if has_header:
            next(reader)
        for row in reader:
            data.append(row)
    return data


if __name__ == "__main__":
    file_dir = source_document_directory

    sub_dirs = list(file_dir.glob('*'))
    for d in sub_dirs:
        print(d)
        files = list(d.glob('*.md'))
        files = random.sample(files, 10)

        output_dir = project_root / 'synthetic_data_generation' / d.name
        for file in tqdm(files, leave=False, desc='Generating Synthetic Data from files:'):
            generate_synthetic_test_data(md_file=file, output_dir=output_dir)
