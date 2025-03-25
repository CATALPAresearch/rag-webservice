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
        self.vector_db_path = "./chroma_langchain_db"
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


    def handleSampleFile(self, file):
        """
        Stores uploaded file temporally on disc
        """
        logger.info(f"File {file} content extracted")
        loader = UnstructuredPDFLoader(file)
        data = loader.load()
        return data, '', ''
    
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
                #metadata = {
                #            "system": doc["system"],
                #            "file": doc["file"],
                #            "course_id": doc["course_id"],
                #            "page": page_num,
                            #"activity_id": doc["activity_id"],
                            #"activity_name": doc["activity_name"],
                            #"activity_type": doc["activity_type"],
                #        }
                #metadata[doc["activity_type"]] = doc["activity_id"]
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
    

    def get_documents_by_course(self, system, course_id):
        """
        ...
        """
        # Perform filtered search via metadata
        results = self.vector_db.get(include=["metadatas", "documents"])

        documents = []
        # filter documents by system and course_id
        for doc, meta in zip(results['documents'], results['metadatas']):
            if meta.get('system') == system and str(meta.get('course_id')) == str(course_id):
                documents.append({
                    #"id": meta.get("document_id", ""),
                    #"title": meta.get("title", "Untitled Document"),
                    "file": meta.get("filename", "unknown.pdf"),
                    "system": meta.get("system"),
                    "course_id": meta.get("course_id")
                })
        return documents
    
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
