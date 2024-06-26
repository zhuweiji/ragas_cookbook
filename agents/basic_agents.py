
import logging
import sys
from pathlib import Path
from typing import List, Optional

from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.config import RunnableConfig

# from llms.ollama import llm
from llms.mistral_api import llm
from retrievers.chroma import ExtendedChromaMarkdownRetriever

log = logging.getLogger(__name__)


class BaseAgent:
    def __init__(self, retriever=ExtendedChromaMarkdownRetriever(),
                 config: Optional[RunnableConfig | str] = None) -> None:
        log.info(f'loading {self.__class__}')
        self.llm = llm
        self.retriever = retriever

        if config == 'stdout':
            handler = StdOutCallbackHandler()
            config = RunnableConfig({
                'callbacks': [handler]
            })

        self.langchain_config: Optional[RunnableConfig]
        self.langchain_config = config  # type: ignore

        log.info(f'loaded {self.__class__}')

        # self.llm_embeddings = OllamaEmbeddings(model=llm_model_name)

    prompt_force_json = "Your output should only be json."
