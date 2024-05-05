import re
from dataclasses import dataclass
from typing import Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agents.basic_agents import BaseAgent, BasicAgent


@dataclass
class RAGResponse:
    documents: list[Document]
    answer: BaseMessage
    evaluation: BaseMessage
    evaluation_score: Optional[str]


class RagAgent(BasicAgent):
    def answer_with_rag(self, question: str):
        documents = self.retriever._get_relevant_documents(question)
        answer = self.answer_question(question, documents)
        evaluation = self.evaluate_answer(question, answer)

        assert isinstance(evaluation.content, str)

        match = re.search(r"Score \(1-5\): (\d+)", evaluation.content)
        evaluation_score = None

        if match:
            evaluation_score = match.group(1)

        return RAGResponse(documents, answer, evaluation, evaluation_score)
