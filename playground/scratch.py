import logging

from agents.basic_agents import BasicAgent
from retrievers.chroma import ExtendedChromaMarkdownRetriever

logging.basicConfig(
    format='%(name)s-%(levelname)s|%(lineno)d:  %(message)s', level=logging.INFO)
log = logging.getLogger(__name__)

retriever = ExtendedChromaMarkdownRetriever()
agent = BasicAgent()

question = 'what is AWS Chalice?'
answer = agent.answer_question(
    question, retriever._get_relevant_documents(question))

print(answer)
