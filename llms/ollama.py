import logging

from langchain_community.llms import Ollama

from config.constants import llm_model_name
from utilities.ollama import verify_ollama_model_present

log = logging.getLogger(__name__)

llm = Ollama(model=llm_model_name)
verify_ollama_model_present(llm_model_name)
