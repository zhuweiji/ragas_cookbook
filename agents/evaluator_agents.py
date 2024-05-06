import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import ragas
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_correctness, faithfulness

from agents.basic_agents import BasicAgent
from agents.rag_agents import RagAgent, RAGResponse


@dataclass
class EvaluationResult:
    """An instance of a evaluation of an RAG agent, which should be storable. relevant Ragas metrics should be added as fields to the class"""
    question: str
    ground_truth: str
    answer_evaluation: str
    generated_answer: str
    docs: list

    def to_json(self, filepath: Path):
        with open(filepath, 'w') as fp:
            json.dump(dataclasses.asdict(self), fp)

    @classmethod
    def from_json(cls, filepath: Path):
        with open(filepath, 'r') as fp:
            o = json.load(fp)

        return cls(**o)


class EvaluationAgent(RagAgent):
    def evaluate_rag(self, rag_response: RAGResponse, ground_truth: str):
        data = {
            'question': rag_response.question,
            'answer': rag_response.answer,
            'contexts': [doc.page_content for doc in rag_response.documents],
            'ground_truth': ground_truth
        }
        dataset = Dataset.from_dict(data)

        score = evaluate(dataset,
                         metrics=[faithfulness, answer_correctness],
                         llm=self.llm,
                         embeddings=self.retriever.embedding_function,
                         )
        return score.to_pandas()
