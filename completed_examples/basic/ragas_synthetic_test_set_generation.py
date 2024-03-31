import os
from pathlib import Path

from datasets import Dataset
from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from ragas import evaluate
from ragas.metrics import answer_correctness, faithfulness
from ragas.testset.evolutions import multi_context, reasoning, simple
from ragas.testset.generator import TestsetGenerator

from utilities.ollama import verify_ollama_model_present

MODEL_NAME = "openchat"
# https://ollama.com/library/openchat

generator_llm = Ollama(model=MODEL_NAME)
embeddings = OllamaEmbeddings(model=MODEL_NAME)


# make sure that the ollama is available to use locally (might need to ollama pull)
verify_ollama_model_present("openchat")


loader = DirectoryLoader(Path.cwd() / 'source_docs' / 'only_textract')
documents = loader.load()


generator = TestsetGenerator.from_langchain(
    generator_llm,
    generator_llm,
    embeddings
)

# generate testset
testset = generator.generate_with_langchain_docs(documents,
                                                 test_size=10, distributions={
                                                     simple: 0.5, reasoning: 0.25, multi_context: 0.25})

df = testset.to_pandas()
df.to_excel('synthetic_testset.xlsx')
print(df)
