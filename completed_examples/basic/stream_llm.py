import sys
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from utilities.ollama import verify_ollama_model_present

sys.path.append(
    str(Path.cwd().parents[1])
)


MODEL_NAME = "openchat"
# https://ollama.com/library/openchat

generator_llm = Ollama(model=MODEL_NAME)
embeddings = OllamaEmbeddings(model=MODEL_NAME)


# make sure that the ollama is available to use locally (might need to ollama pull)
verify_ollama_model_present("openchat")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an essay assistant tasked with writing excellent 5-paragraph essays.
             Generate the best essay possible for the user's request.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generate = prompt | generator_llm

request = HumanMessage(
    content="Write an essay on the threat of supply chain attacks."
)
for chunk in generate.stream({"messages": [request]}):
    print(chunk, end='')
