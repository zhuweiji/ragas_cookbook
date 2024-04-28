
import logging
import sys
from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config.constants import llm_model_name
from retrievers.chroma import ExtendedChromaMarkdownRetriever
from utilities.ollama import verify_ollama_model_present

log = logging.getLogger(__name__)


class Agent:
    def __init__(self, retriever=ExtendedChromaMarkdownRetriever()) -> None:
        self.llm = Ollama(model=llm_model_name)
        self.retriever = retriever
        # self.llm_embeddings = OllamaEmbeddings(model=llm_model_name)

        # make sure that the ollama is available to use locally (might need to ollama pull)
        verify_ollama_model_present(llm_model_name)

    def answer_question(self, query, docs: List[Document]):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Answer the question based on the context below. Do not include any pre-amble, keep the answer concise. Respond "Unsure about answer" if not sure about the answer."""
                ),
                MessagesPlaceholder(variable_name="question"),
            ]
        )

        generate = prompt | self.llm

        request = HumanMessage(
            content=f"""Question: {query}
            Context: {docs}"""
        )

        result = ''
        for chunk in generate.stream({"question": [request]}):
            print(chunk, end='')
            result += chunk
        return result
