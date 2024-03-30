import os
import sys
from pathlib import Path

from datasets import Dataset
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.callbacks import StdOutCallbackHandler
from ragas import evaluate
from ragas.metrics import answer_correctness, faithfulness

# https://python.langchain.com/docs/modules/callbacks/
from utilities.langchain_callback_handler import UnimplementedCallbackHandler
from utilities.ollama import verify_ollama_model_present

# https://ollama.com/library/openchat
MODEL_NAME = "openchat"

generator_llm = Ollama(model=MODEL_NAME)
embeddings = OllamaEmbeddings(model=MODEL_NAME)

data_samples = {
    'question': ['When was the first super bowl?', 'Who won the most super bowls?'],
    'answer': ['The first superbowl was held on Jan 15, 1967', 'The most super bowls have been won by The New England Patriots'],
    'contexts': [['The First AFL-NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,'],
                 ['The Green Bay Packers...Green Bay, Wisconsin.', 'The Packers compete...Football Conference']],
    'ground_truth': ['The first superbowl was held on January 15, 1967', 'The New England Patriots have won the Super Bowl a record six times']
}

handler = UnimplementedCallbackHandler()

dataset = Dataset.from_dict(data_samples)

score = evaluate(dataset,
                 metrics=[faithfulness, answer_correctness],
                 llm=generator_llm,
                 embeddings=embeddings,
                 callbacks=[handler]
                 )
score.to_pandas()

print(score)
