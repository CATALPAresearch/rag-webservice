import logging
import os
from ollama import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# Set protobuf environment variable to avoid error messages
# This might cause some issues with latency but it's a tradeoff
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class LLM_Manager:

    def __init__(self):
        self.base_url = "localhost:11434"
        # self.base_url = "http://catalpa-llm.fernuni-hagen.de:11434"
        self.selected_model = 'llama3.1:latest'  # 'deepseek-r1:latest'
        self.llm = None
        self.load_LLM()
        
    
    def load_LLM(self, model=None, temperatur=0.1, top_k=25, top_p=0.3, repeat_penalty=1.1):
        """
        """
        if self.selected_model == None and model == None:
            self.selected_model = self.get_model_names()[0]
            logger.warning(f"Set LLM model to {self.selected_model}")

        if  self.selected_model not in self.get_model_names():
            self.selected_model = self.get_model_names()[0]
            logger.warning(f"Set LLM model to {self.selected_model}")
            
        self.llm = ChatOllama(
            model=self.selected_model, 
            base_url=self.base_url,
            temperature=temperatur,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
        )


    def get_model_names(self) -> list:
        """
        """
        client = Client(host=self.base_url)
        names = []
        for model in client.list()['models']:
            m = dict(model)
            names.append(m['model'])
        return names


    def process_simple_question(self, question: str) -> str:
        """
        Process a user question using the selected language model.

        Args:
            question (str): The user's question.
            
        Returns:
            str: The generated response to the user's question.
        """
        logger.info(f"Processing simple question: {question} using model: {self.selected_model}")
            
    
        template = """Answer the question:
        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke({"input": question })
        logger.info("Question processed and response generated")
        return response


