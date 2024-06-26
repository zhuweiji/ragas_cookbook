import logging
import re

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agents.basic_agents import BaseAgent

log = logging.getLogger(__name__)


class DataGenerator(BaseAgent):
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
                    2. The question should be fully answerable from the given context.
                    3. Do not use phrases like 'provided context', etc in the question
                    4. Questions should not be able to be decomposed into smaller subquestions.
                    5. The question should not contain more than 20 words. Make of use of abbreviation wherever possible.
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

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Your goal is to generate questions and answer pairs with questions mimicking how users search for information on Google.
                    Users condense questions into short form queries, typically employing domain specific keywords.
                    Examples: aws run container, connect to remote ec2, k8s cluster LB.
                    The idea is distilled, removing extraneous information and ambiguous phrases are removed or replaced with shorter words."""

                    "Generate up to" + f' {num_questions} ' + """questions from the given document and provide the answer to each one.
                    Each question must start with "Question:"
                    Each answer must start with "Answer:"
                    
                    The generated questions must satisfy the rules given below:
                    1. The question should be fully answerable from the given document.
                    2. Do not use phrases like 'provided context', 'provided document', etc in the question
                    3. The question must be shorter than 6 words.
                    Follow the template below when generating the question answer pairs.
                    
                    Example:
                    Context: Photosynthesis in plants involves converting light energy into chemical energy (carbohydrates), using chlorophyll and other pigments to absorb light. This process is crucial for plant growth and the production of oxygen.
                    Question: photosynthesis plant growth
                    Answer: Carbohydrates produced from photosynthesis provide energy for all plant growth and maintenance.
                    
                    Example:
                    Context: The **Visual Studio Code Dev Containers** extension lets you use a container as a full-featured development environment. It allows you to open any folder inside (or mounted into) a container and take advantage of Visual Studio Code's full feature set. A [devcontainer.json file](#create-a-devcontainerjson-file) in your project tells VS Code how to access (or create) a **development container** with a well-defined tool and runtime stack. This container can be used to run an application or to separate tools, libraries, or runtimes needed for working with a codebase. Workspace files are mounted from the local file system or copied or cloned into the container. Extensions are installed and run inside the container, where they have full access to the tools, platform, and file system. This means that you can seamlessly switch your entire development environment just by connecting to a different container.
                    Question: vscode container dev env
                    Answer: The Visual Studio Code Dev Containers extension lets you use a container as a full-featured development environment. \n\n
                    """.replace('\t', '')
                ),
                MessagesPlaceholder(variable_name="document"),
            ]
        )

        generate = prompt | self.llm

        request = HumanMessage(
            content=f"""Document Name: {document.metadata.get('filename', 'unknown')}
            Document: {document.page_content}""".replace('\t', '')
        )

        return generate.invoke({'document': [request]}, config=self.langchain_config)

    def evaluate_generated_question(self, document: Document, question, answer):
        """evaluate if the generated question is usable
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Evaluate if the following generated question and answer for a given document is usable as synthetic test data.
                    Give your answer as a single word, either 'yes' or 'no.' Do not provide additional output besides 'yes' or 'no'.
                    
                    Answer 'no' if the following criteria are not met:
                    1. If the question is understandable by itself
                    2. If the answer is relevant to the question
                    3. If it is plausible that the question can be answered using the document and/or table of contents given.
                    """.replace('\t', '')
                ),
                MessagesPlaceholder(variable_name="document"),
            ]
        )

        generate = prompt | self.llm

        request = HumanMessage(
            content=f"""Document Name: {document.metadata.get('filename', 'unknown')}
            Table of Contents: {document.metadata.get('table_of_contents')}
            Question: {question}
            Answer: {answer}
            """.replace('\t', '')
        )
        return generate.invoke({'document': [request]}, config=self.langchain_config)

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
