import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import ragas
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_correctness, faithfulness

from agents.basic_agents import BasicAgent
from agents.rag_agents import RagAgent


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
    def evaluate_rag(self, question, answer):
        response = self.answer_with_rag(question)
