import re
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agents.basic_agents import BaseAgent
from agents.evaluator_agents import ReflectionAgent


@dataclass
class RAGResponse:
    question: str
    documents: list[Document]
    answer: BaseMessage
    evaluation: BaseMessage
    evaluation_score: Optional[str]


def join_docs(docs: list[Document]):
    return '\n\n'.join([i.page_content for i in docs])


class RagAgent(ReflectionAgent):
    def answer_with_rag(self, question: str):
        documents = self.retriever._get_relevant_documents(question)
        answer = self.answer_question(question, documents)
        evaluation = self.evaluate_answer(question, answer)

        assert isinstance(evaluation.content, str)

        match = re.search(r"Score \(1-5\): (\d+)", evaluation.content)
        evaluation_score = None

        if match:
            evaluation_score = match.group(1)

        return RAGResponse(question, documents, answer, evaluation, evaluation_score)

    def answer_question(self, query, docs: List[Document]):
        # prompt = """Context: [Insert context here]
        # Question: [Insert question here]
        # Answer: Concise answer based on context: | "Context is irrelevant to the question." | "I do not know the answer."
        # """

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Answer the question based on the context below. Do not add any preamble, keep the answer concise. 
                    If you cannot answer because of any of the following reasons, make sure to give the response corresponding to the reasons below.
                    If all the context provided is irrelevant, respond "Context is irrelevant to the question.\nReason:"
                    If you cannot answer the question confidently, respond "I do not know the answer\nReason:"
                    """.replace('\t', '')
                ),
                MessagesPlaceholder(variable_name="question"),
            ]
        )

        generate = prompt | self.llm

        context = join_docs(docs)

        request = HumanMessage(
            content=f"""Question: {query}
            Context: {context}""".replace('\t', '')
        )

        return generate.invoke({'question': [request]}, config=self.langchain_config)

    def evaluate_document_relevance(self, question, document: Document):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Evaluate a document's relevance for a question. Do not add any preamble or explanation. Follow this template:
                    Answered: Yes|No.
                    Reason:

                    Example:
                    Answered: No.
                    Reason: The context provided does not give information about Python.

                    Example:
                    Answered: Yes.
                    Reason: The context provided gives source code for implementing a Flask server.
                    """.replace('\t', '')
                ),
                MessagesPlaceholder(variable_name="question"),
            ]
        )

        generate = prompt | self.llm

        request = HumanMessage(
            content=f"""Question: {question}
            Document: {document.page_content}""".replace('\t', '')
        )

        return generate.invoke({'question': [request]}, config=self.langchain_config)

    def try_document_toc_irrelevant_document(self, question, evaluation, document: Document):
        """Try to salvage an irrelevant document by either 
        1. searching within the same document for more relevant chunks 
        2. modifying the search query"""

        table_of_contents = document.metadata.get('table_of_contents', None)

        if not table_of_contents:
            return None

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """A document has been evaluated to be irrelevant to a question.
                    Use the evaluation of a generated answer and the table of contents of the document,
                    to determine if the question can be answered by other relevant chunks in the document or not.
                    
                    Answer in the template of these examples:
                    Yes, using the following headers: 3: ##Prerequisites, 5: ### Examples
                    Yes, using the following headers: 1: #Introduction
                    No.
                        """.replace('\t', '')
                ),
                MessagesPlaceholder(variable_name="question"),
            ]
        )

        generate = prompt | self.llm

        request = HumanMessage(
            content=f"""Question: {question}
            Evaluation: {evaluation}
            Table of Contents: {table_of_contents}""".replace('\t', '')
        )

        return generate.invoke({'question': [request]}, config=self.langchain_config)
