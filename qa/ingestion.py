from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging
import os
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def data_ingestion(pdf_file_path):
    logger.info(f"Processing file: {pdf_file_path}")
    
    # Load the PDF files
    loader = PyPDFDirectoryLoader(pdf_file_path)
    documents = loader.load()
    
    if not documents:
        logger.error("No documents were loaded from the PDF.")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    logger.info(f"Loaded {len(docs)} document chunks")
    
    return docs

def get_vector_store(docs):
    if not docs:
        logger.error("No documents to process. Cannot create vector store.")
        return None
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    try:
        logger.info("Starting to create vector store...")
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        
        vector_store_faiss = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        vector_store_faiss.save_local("faiss_index")
        logger.info("Vector store created and saved successfully")
        return vector_store_faiss
    except Exception as e:
        logger.error(f"Error in creating vector store: {str(e)}")
        return None

def load_vector_store():
    try:
        vector_store = FAISS.load_local("faiss_index", HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
        logger.info("Vector store loaded successfully")
        return vector_store
    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}")
        return None

def process_pdf_to_vector_store(pdf_file_path):
    docs = data_ingestion(pdf_file_path)
    if docs:
        vector_store = get_vector_store(docs)
        if vector_store:
            logger.info("PDF processed and vector store created successfully!")
        else:
            logger.error("Failed to create vector store.")
    else:
        logger.error("No documents to process. Please check the PDF file.")
        
    return vector_store

if __name__ == "__main__":
    pdf_file_path = "data/good.pdf"  # Replace with actual path to your PDF file or directory
    vector_store = process_pdf_to_vector_store(pdf_file_path)
    
    if vector_store:
        logger.info("Vector store creation complete.")
    else:
        logger.error("Vector store creation failed.")
