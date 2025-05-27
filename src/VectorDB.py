import logging
import os
import math
import tempfile
import shutil
import warnings
from uuid import uuid4

# Suppress torch warning
warnings.filterwarnings('ignore', category=Warning, message='.*torch.classes.*')

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from utils.document_model import RAG_Document

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



class VectorDB:
    """
    
    """

    def __init__(self):
        self.vector_db = None
        self.collection_name="Moodle-RAG-collection"
        self.index_embedding = None
        self.embed_model_name = None
        self.set_index_embedding_model() 
        self.vector_db_path = "./data/chroma_langchain_db"
        self.create_vector_db()


    def set_index_embedding_model(self, model_provider="local", model_name="all-MiniLM-L6-v2"):
        match model_provider:
            case 'local':
                #self.index_embedding = SentenceTransformerEmbeddings(model_name=model_name)
                self.index_embedding = HuggingFaceEmbeddings(model_name=model_name)
            case 'HuggingFace':
                self.index_embedding = HuggingFaceEmbeddings(model_name=model_name)
            
            #TODO: add futher model provider
    

    def create_vector_db(self):
        """
        Create an empty vector store including a new collection.
        """
        logger.info(f"Create empty vector store with collection nameed {self.collection_name}")
        self.vector_db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.index_embedding,
            persist_directory=self.vector_db_path  
        )
        logger.info(f"Collection {self.collection_name} in a new vector store was created")


    # TODO
    def handleFile(self, file_upload):
        """
        Stores uploaded file temporally on disc
        """
        temp_dir = tempfile.mkdtemp()

        path = os.path.join(temp_dir, file_upload.name + math.random()*10000)
        with open(path, "wb") as f:
            f.write(file_upload.getvalue())
            logger.info(f"File saved to temporary path: {path}")
            loader = UnstructuredPDFLoader(path)
            data = loader.load()
        return data, temp_dir, path


    async def handlePDF(self, file_path):
        loader = PyPDFLoader(file_path)
        pages = []
        async for page in loader.alazy_load():
            #print('new page')
            pages.append(page)
        return pages


    def getChunks(self, data, splitter='langchain_text_splitters'):
        """
        ...
        """
        #print(data)
        chunks = None
        match splitter:
            case 'langchain_text_splitters':
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
                logger.info("Document split into chunks")
                chunks = text_splitter.split_text(str(data))
        return chunks
        

    async def add_documents(self, document_list) -> Chroma:
        """
        Adds documents to the vector store after preprocessing

        Params 
          file_list_ is list of file paths as string
          collection_name: Name of the collection in which the documents should be stored. If no collection name is provided documents will be stored in individual collections.
        """
        logger.info(f"Add documents to vector DB from file upload: {len(document_list)}")
        documents = []
        uuids = []
        i = 0
        for doc in document_list:
            i = i+1
            #print('########## ' + doc["file"])
            # Step 1: Load and preprocess document
            #data, temp_dir, path = self.handleSampleFile(doc["file"])
            pages = await self.handlePDF(doc["file"])

            # data, temp_dir, path = self.handleFile(doc["file"])

            # Step 2: Chunk the document content
            #chunks = self.getChunks(data)

            # Step 3: Add documents including meta data
            page_num = 0
            for page in pages:
                page_num = page_num + 1
                logger.info('page: ' + str(page_num))
                metadata = doc
                metadata['page'] = page_num
                
                documents.append(
                    Document(
                        page_content=page.page_content,
                        metadata=metadata,
                        id=i, # FixMe: ID must be unique accross courses and activities
                    )
                )
            
            # Step 4: Cleanup
            #if temp_dir != '':
            #    shutil.rmtree(temp_dir)
            #logger.info(f"Temporary directory {temp_dir} removed")
        
        # Step 5: Store the vector representation in the vector database
        if self.vector_db == None:
            self.vector_db = self.create_vector_db()

        uuids = [str(uuid4()) for _ in range(len(documents))]
        document_ids = self.vector_db.add_documents(documents=documents, ids=uuids)

        logger.info('document_ids')
        logger.info(document_ids)
        return document_ids
    
    
    def get_documents_by_course(self, system=None, course_id=None):
        """
        ...
        """
        # Perform filtered search via metadata
        # results = self.vector_db.get(include=["metadatas", "documents"])
        results = self.vector_db.get()
        
        documents = []
        # filter documents by system and course_id
        for doc, meta in zip(results['documents'], results['metadatas']):
            if meta.get('system') == system and str(meta.get('course_id')) == str(course_id):
                doc = self.get_RAG_document(meta)
                documents.append( doc.dict(exclude_none=True))
            elif meta.get('system') == system and course_id==None:
                doc = self.get_RAG_document(meta)
                documents.append( doc.dict(exclude_none=True))
            elif system == None and course_id == None:
                doc = self.get_RAG_document(meta)
                documents.append(doc.dict(exclude_none=True))

        return documents
    

    def get_RAG_document(self, meta) -> RAG_Document:
        """
        """
        return RAG_Document(
            system=meta.get("system"),
            course_id=meta.get("course_id", 0),
            activity_type=meta.get("activity_type", None),
            activity_name=meta.get("activity_name", None),
            activity_longpage=meta.get("activity_longpage", None),        
            activity_pdf=meta.get("activity_pdf", None),
            activity_assign=meta.get("activity_assign", None),
            activity_wiki=meta.get("activity_wiki", None),
            activity_quiz=meta.get("activity_quiz", None),
            activity_forum=meta.get("activity_forum", None),
            file=meta.get("file", None),
            page=meta.get("page", None),
        )
        

    def get_index(self, system, course_id, activity_type, activity_id) -> None:
        """
        Get a document from the document store
        """
        results = self.vector_db.get()
        documents = []
        for doc, meta in zip(results['documents'], results['metadatas']):
            if meta.get('system') == system and str(meta.get('course_id')) == str(course_id) and str(meta.get('activity_type')) == activity_type and str(meta.get('activity_id')) == activity_id:
                print('got' + str(doc.id))
                documents.append(doc)
        return documents


    def delete_document(self, system, course_id, activity_type, activity_id) -> None:
        """
        Deletes a document from the document store
        """
        results = self.vector_db.get()
        for doc, meta in zip(results['documents'], results['metadatas']):
            if meta.get('system') == system and str(meta.get('course_id')) == str(course_id) and str(meta.get('activity_type')) == activity_type and str(meta.get('activity_id')) == activity_id:
                print('delete' + str(doc.id))
                self.vector_db.delete(ids=[doc.id])


    def update_document(self) -> None:
        """
        Updates a document from the document store. Currently not implemented.
        """
        pass



    def delete_vector_db(self) -> None:
        """
        Delete the vector database and clear related session state.
        """
        logger.info("Deleting vector DB")
        if self.vector_db is not None:
            try:
                self.vector_db.delete_collection()
                logger.info("Vector DB and related session state cleared")
            except Exception as e:
                logger.error(f"Error deleting collection: {e}")
        else:
            logger.warning("Attempted to delete vector DB, but none was found")


    # TODO: remove
    def load_sample_pdf(self, sample_path = "data/pdfs/sample/pobscinf.pdf"):
        if os.path.exists(sample_path):
            if self.vector_db is None:
                loader = UnstructuredPDFLoader(file_path=sample_path)
                data = loader.load()
                self.vector_db = Chroma.from_documents(
                    documents=self.getChunks(data),
                    embedding=self.index_embedding,
                    persist_directory=self.vector_db_path,
                    collection_name="sample_pdf"
                )
        else:
            print("Sample PDF file not found in the current directory.")
    
    
    # TODO remove function
    def handleSampleFile(self, file):
        """
        Stores uploaded file temporally on disc
        """
        logger.info(f"File {file} content extracted")
        loader = UnstructuredPDFLoader(file)
        data = loader.load()
        return data, '', ''
    