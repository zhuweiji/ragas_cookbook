from pathlib import Path

project_root = Path(__file__).parents[1]

source_document_directory = project_root / 'sample_documents'
chromadb_dir = project_root / 'db'
synthetic_data_dir = project_root / 'synthetic_data_generation' / 'data'
