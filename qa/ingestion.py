from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def data_ingestion():
    # Get the absolute path of the current working directory
    current_dir = os.getcwd()
    # Go up one level to the parent directory
    parent_dir = os.path.dirname(current_dir)
    # Construct the absolute path to the data directory
    data_dir = os.path.join(parent_dir, "data")
    
    logger.info(f"Looking for data in: {data_dir}")
    if not os.path.exists(data_dir):
        logger.error(f"Directory {data_dir} does not exist.")
        return []
    
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    if not pdf_files:
        logger.error(f"No PDF files found in {data_dir}")
        return []
    
    logger.info(f"Found {len(pdf_files)} PDF files in {data_dir}")
    
    loader = PyPDFDirectoryLoader(data_dir)
    documents = loader.load()
    
    if not documents:
        logger.error("No documents were loaded from the PDFs.")
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
        
        # Debug: Print the first few texts
        logger.info(f"First few texts: {texts[:3]}")
        
        # Embed documents in smaller batches
        batch_size = 32
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = embeddings.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            logger.info(f"Embedded batch {i//batch_size + 1}/{len(texts)//batch_size + 1}")
        
        logger.info(f"Created {len(all_embeddings)} embeddings")
        
        vector_store_faiss = FAISS.from_embeddings(all_embeddings, embeddings, metadatas)
        vector_store_faiss.save_local("faiss_index")
        logger.info("Vector store created and saved successfully")
        return vector_store_faiss
    except Exception as e:
        logger.error(f"Error in creating vector store: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        docs = data_ingestion()
        if docs:
            get_vector_store(docs)
        else:
            logger.error("No documents to process. Exiting.")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")