import logging

from langchain_mistralai.chat_models import ChatMistralAI

from config.project_secrets import MISTAL_API_KEY

log = logging.getLogger(__name__)

llm = ChatMistralAI(
    api_key=MISTAL_API_KEY  # type: ignore
)
