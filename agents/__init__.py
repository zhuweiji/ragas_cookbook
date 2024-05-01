
import logging
import sys
from pathlib import Path
from typing import List

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
    def __init__(self, retriever=ExtendedChromaMarkdownRetriever()) -> None:
        self.llm = llm
        self.retriever = retriever
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

        handler = StdOutCallbackHandler()

        config = RunnableConfig({
            'callbacks': [handler]
        })
        generate = prompt | self.llm

        request = HumanMessage(
            content=f"""Question: {query}
            Context: {docs}""".replace('\t', '')
        )

        return generate.invoke({'question': [request]}, config=config)

    def evaluate_answer(self, question, answer):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Evaluate a response to a given question. Do not add any preamble or explanation. Use the following metrics:
                    
                    Score (1-5):
                    1: Completely wrong or irrelevant answer.
                    2: Somewhat off-topic or inaccurate answer.
                    3: Lacks details or clarity.
                    4: Mostly correct answer.
                    5: Excellent answer.
                    
                    Answered: (Yes/No) - Did the answer address the core aspects of the question?
                    
                    Example:

                    Question: What is the capital of France?
                    Answer: France's capital city.

                    Evaluation:

                    Score (1-5): 4
                    Answered: Yes
                    """.replace('\t', '')
                ),
                MessagesPlaceholder(variable_name="question"),
            ]
        )

        handler = StdOutCallbackHandler()

        config = RunnableConfig({
            'callbacks': [handler]
        })
        generate = prompt | self.llm

        request = HumanMessage(
            content=f"""Question: {question}
            Answer: {answer}""".replace('\t', '')
        )

        return generate.invoke({'question': [request]}, config=config)

    def evaluate_document_relevance(self, question, document: Document):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Evaluate a document's relevance for a question. Do not add any preamble or explanation. Use the following metrics:
                    Answered: (Yes/No) - Did the answer address the core aspects of the question?
                    Reason:
                    """.replace('\t', '')
                ),
                MessagesPlaceholder(variable_name="question"),
            ]
        )

        handler = StdOutCallbackHandler()

        config = RunnableConfig({
            'callbacks': [handler]
        })
        generate = prompt | self.llm

        request = HumanMessage(
            content=f"""Question: {question}
            Document: {document}""".replace('\t', '')
        )

        return generate.invoke({'question': [request]}, config=config)
