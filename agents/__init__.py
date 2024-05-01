
import logging
import sys
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.config import RunnableConfig

# from llms.ollama import llm
from llms.mistral_api import llm
from retrievers.chroma import ExtendedChromaMarkdownRetriever

log = logging.getLogger(__name__)


class Agent:
    def __init__(self, retriever=ExtendedChromaMarkdownRetriever(),
                 config: Optional[RunnableConfig | str] = None) -> None:
        self.llm = llm
        self.retriever = retriever

        if config == 'stdout':
            handler = StdOutCallbackHandler()
            config = RunnableConfig({
                'callbacks': [handler]
            })

        self.langchain_config: Optional[RunnableConfig]
        self.langchain_config = config

        # self.llm_embeddings = OllamaEmbeddings(model=llm_model_name)

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
                    If you cannot answer the question confidently, respond "I do not know the answer.nReason:"
                    """.replace('\t', '')
                ),
                MessagesPlaceholder(variable_name="question"),
            ]
        )

        generate = prompt | self.llm

        request = HumanMessage(
            content=f"""Question: {query}
            Context: {docs}""".replace('\t', '')
        )

        return generate.invoke({'question': [request]}, config=self.langchain_config)

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
            Document: {document}""".replace('\t', '')
        )

        return generate.invoke({'question': [request]}, config=self.langchain_config)

    def try_document_toc_irrelevant_document(self, question, evaluation, document: Document):
        """Try to salvage an irrelevant document by either 
        1. searching within the same document for more relevant chunks 
        2. modifying the search query"""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """A document has been evaluated to be irrelevant to a question.
                    Use the evaluation and a table of contents of the document, to determine if the question can be answered by other relevant chunks in the document or not.
                    
                    Answer in the template of these examples:
                    Yes, using the following headers: 3. ##Prerequesites, 5. ### Example
                    No.
                        """.replace('\t', '')
                ),
                MessagesPlaceholder(variable_name="question"),
            ]
        )

        generate = prompt | self.llm

        request = HumanMessage(
            content=f"""Question: {question}
                Document: {document}""".replace('\t', '')
        )

        return generate.invoke({'question': [request]}, config=self.langchain_config)
