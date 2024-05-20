import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import ragas
from datasets import Dataset
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from ragas import evaluate
from ragas.metrics import answer_correctness, faithfulness

from agents.basic_agents import BaseAgent
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


class ReflectionAgent(BaseAgent):
    def evaluate_answer(self, question, answer):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Evaluate a response to a given question. Do not add any preamble or explanation. Use the following metrics:
                    
                    Score (1-5):
                    Answered: (Yes/No).
                    
                    Example:

                    Question: What is the capital of France?
                    Answer: France's capital city.

                    Evaluation:

                    Score (1-5): 2
                    Answered: No.
                    """.replace('\t', '')
                ),
                MessagesPlaceholder(variable_name="question"),
            ]
        )

        generate = prompt | self.llm

        request = HumanMessage(
            content=f"""Question: {question}
            Answer: {answer}""".replace('\t', '')
        )

        return generate.invoke({'question': [request]}, config=self.langchain_config)

    def regenerate_new_answer_from_evaluation(self, question, answer, evaluation):
        """Reflect on an inadequate answer to try to produce a better answer

        This might involve a chain including tools and SQR? 
        But for now just use evaluation to produce a new answer
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Given the following answer for a question, and its critique, respond with a revised answer.
                    If information/context is missing, or no good answer can be generated, then answer in the following templates:
                    Information is missing. Details:
                    No good answer can be generated. Details:
                    """.replace('\t', '')
                ),
                MessagesPlaceholder(variable_name="question"),
            ]
        )

        generate = prompt | self.llm

        request = HumanMessage(
            content=f"""Question: {question}
                Answer: {answer}
                Evaluation: {evaluation}""".replace('\t', '')
        )

        return generate.invoke({'question': [request]}, config=self.langchain_config)


class RagasAgent(RagAgent):
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
