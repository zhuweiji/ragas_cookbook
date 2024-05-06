import logging
import re

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agents.basic_agents import BasicAgent

log = logging.getLogger(__name__)


class DataGenerator(BasicAgent):
    def generate_questions_from_document(self, document: Document, num_questions=3):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Your goal is to generate questions and answer pairs to quiz students of their knowledge of a document chunk. Generate up to" + f' {num_questions} ' + """questions from the given document and provide the answer to each one.
                    End each question with a '?' character. Write the answer to that question using only the context provided.
                    Each question must start with "Question:"
                    Each answer must start with "Answer:"
                    
                    The generated questions must satisfy the rules given below:
                    1. The question must be understandable by itself, without the given context.
                    2. The question should be fully answered from the given context.
                    3. The question should be framed from a part of context that contains 
                    4. Do not use phrases like 'provided context',etc in the question
                    5. Avoid framing question using word "and" that can be decomposed into more than one question.
                    6. The question should not contain more than 20 words. Make of use of abbreviation wherever possible.
                    Follow the template below when generating the question answer pairs.
                    
                    Example:
                    Context: Photosynthesis in plants involves converting light energy into chemical energy, using chlorophyll and other pigments to absorb light. This process is crucial for plant growth and the production of oxygen.
                    Question: "What is the role of photosynthesis in plant growth?"
                    Answer: "Photosynthesis provides the food and energy for plants to grow."\n\n
                    """.replace('\t', '')
                ),
                MessagesPlaceholder(variable_name="question"),
            ]
        )

        generate = prompt | self.llm

        request = HumanMessage(
            content=f"""Document Name: {document.metadata.get('filename', 'unknown')}
            Document: {document.page_content}""".replace('\t', '')
        )

        return generate.invoke({'question': [request]}, config=self.langchain_config)

    def generate_questions_from_document__search_engine_style_query(self, document: Document, num_questions=3):
        """Generate questions, mimicking how users familiar with search engines
        might rephrase their question using a combination of important keywords.

        Example: For the question
        What is the role of photosynthesis in plant growth?
        -> photosynthesis plant growth
        """
        raise NotImplementedError

    def generate_questions_from_document__simple_style_query(self, document: Document, num_questions=3):
        """Generate questions, focusing on how someone might ask the question 
        when they don't know the technical keywords/jargon.

        Example: For the question
        What is the role of photosynthesis in plant growth?
        -> why sunlight plant grow"""
        raise NotImplementedError

    @classmethod
    def extract_qa_tuples(cls, text):
        """
        Extracts question and answer groups into tuples from a given text.

        Args:
            text: The text containing questions and answers in the specified format.

        Returns:
            A list of tuples, where each tuple contains the question and its corresponding answer(s).
        """
        pattern = r"(Question: (?P<question>.+?)\nAnswer: (?P<answer>.+?)(?=\n\n|$))"
        matches = re.findall(pattern, text, re.DOTALL)

        return [(match[1], match[2]) for match in matches]
