import logging
import os
from typing import Any, Dict
import warnings
import asyncio
from ollama import Client
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever

from VectorDB import VectorDB
from LLM_Manager import LLM_Manager

# Suppress torch warning
warnings.filterwarnings('ignore', category=Warning, message='.*torch.classes.*')


# Set protobuf environment variable to avoid error messages
# This might cause some issues with latency but it's a tradeoff
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class RAG_Manager(LLM_Manager):
    """
    ...
    """
    
    def __init__(self):
        # super.selected_model = # FixMe
        super().__init__() # load LLM
        self.rt = VectorDB()
        self.vector_db = self.rt.vector_db
        

    async def process_rag_question(self, question: str, model='llama3.1:latest', document_ids=[], filter=[]) -> str:
        """
        Process a user question using the vector database and selected language model.

        Args:
            question (str): The user's question.
            document_index: indexed documents

        Returns:
            str: The generated response to the user's question.
        """
        logger.info(f"Processing question: {question} using model: {self.selected_model}")

        # update model if necessary
        if model != self.selected_model and model in self.get_model_names():
            self.selected_model = model
            self.load_LLM(model=model)
            logger.info(f"Model changed to: {self.selected_model}")
            
        query_prompt = self._build_query_prompt_from_template(question)
        logger.info('query_prompt')
        logger.info(query_prompt)

        retriever = self._setup_retriever(query_prompt, filter)

        #doc_retriever = self.vector_db.as_retriever(search_type='mmr', search_kwargs={"k": 1})
        doc_retriever = self.vector_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.2, "k" : 2})
        #bm25_retriever = BM25Retriever.from_documents(docs)
        #matched_docs = bm25_retriever.get_relevant_documents('Musk')
        matched_docs = doc_retriever.get_relevant_documents(query=question)


        #TODO
        #requested_documents = self.vector_db.get_by_ids(document_ids)
        

        template = """
        Answer the question based ONLY on the following context. Do not make any assumption beyond this context:
        {context}
        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)
        chain = self._setup_chain(prompt, retriever)
        response = chain.invoke(question)


        logger.info("Question processed and response generated")
        # TODO: return the document where the answer was found: https://python.langchain.com/v0.2/docs/how_to/qa_sources/
        return matched_docs, response
    

    def _build_query_prompt_from_template(self, question:str)->PromptTemplate:
        """
        ...

        Args:
            question (str): Question posed by the user
        
        Returns:
            PromptTemplate: Custome prompt template including the users' question
        """
        template = """You are an AI language model assistant. Your task is to generate 2
            different versions of the given user question to retrieve relevant information from the providedd context. 
            By generating multiple perspectives on the user question, your
            goal is to help the user overcome content limitations. Provide these alternative questions separated by newlines.
            Original question: {question}"""
        
        return PromptTemplate(
            input_variables=["question"],
            template=template,
        )


    def _setup_retriever(self, query_prompt, filter=[]) -> MultiQueryRetriever:
        """
        Set up the multi-query retriever.
        TODO: search by similarity and get similarity score
        """

        #if isinstance(filter, list) and len(filter) > 0 and isinstance(filter[0], dict):
        #    filter = filter[0]
        logger.info('filter rag')
        logger.info(filter)
        
        filter_keys = {
            "system": "system",
            "courses": "course_id",
            "activity_name": "activity_name",
            "activity_pdf": "activity_pdf",
            "activity_longpage": "activity_longpage",
            "activity_page": "activity_page",
            "activity_forum": "activity_forum",
            "activity_wiki": "activity_wiki",
        }

        combined_filter = {
            "$and": [
                {db_key: {"$in": filter[user_key]}}
                for user_key, db_key in filter_keys.items()
                if filter.get(user_key)  # Only include if list is not empty or None
            ]
        }

        logger.info('combined filter')
        logger.info(combined_filter)
        
        filters = {'k': 10, 'filter': combined_filter }
        filters = {'k': 10 }
        try:
            return MultiQueryRetriever.from_llm(
                retriever=self.vector_db.as_retriever(
                    search_kwargs=filters
                ), 
                llm=self.llm,
                prompt=query_prompt
            )
        except Exception as e:
            logger.error(f"Error setting up retriever: {e}")
            raise


    def _setup_chain(self, prompt, retriever) -> Any:
        """Set up the RAG chain."""
        try:
            return (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
        except Exception as e:
            logger.error(f"Error setting up chain: {e}")
            raise
    

if __name__ == "__main__":
    rag_manager = RAG_Manager()
    
    documents = [
        {
            "file": 'data/pdfs/1884/2022-Kurs1884-KE1.pdf',
            "system": 'aple-demo-moodle',
            "course_id": 0,
            "activity_longpage": 1,
            "activity_name": 'KE 1',
            
        },
         {
            "file": 'data/pdfs/1884/2022-Kurs1884-KE7.pdf',
            "system": 'aple-demo-moodle',
            "course_id": 0,
            "activity_longpage": 7,
            "activity_name": 'KE 7',
        }
    ]
    
    ids = asyncio.run(rag_manager.rt.add_documents(documents))
    logger.info('document ids')
    logger.info(ids)

    prompt = 'Was ist ein koopertaives System?'
    #response = asyncio.run(pp.process_simple_question(prompt))
    filter = {
            'system': ['aple-demo-moodle'],
            'course_id': [0],
            'activity_longpage': [1,7],
        }
    docs, response = asyncio.run(rag_manager.process_rag_question(
        question=prompt, # 
        model='deepseek-r1:latest',
        filter=filter)
        )
    logger.info('docs')
    #logger.info(docs)
    logger.info('response')
    logger.info(response)

    


# LangChainDeprecationWarning: The class `Chroma` was deprecated in LangChain 0.2.9 and will be removed in 1.0. An updated version of the class exists in the :class:`~langchain-chroma package and should be used instead. To use it run `pip install -U :class:`~langchain-chroma` and import as `from :class:`~langchain_chroma import Chroma``.
#self.vector_db = Chroma(