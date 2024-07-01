from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

embeddings=FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

def data_ingestion():
    loader=PyPDFDirectoryLoader("./data")
    documents=loader.load()
    
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs=text_splitter.split_documents(documents)
    
    return docs
    

def get_vector_store(docs):
    vector_store_faiss=FAISS.from_documents(docs,embeddings)
    vector_store_faiss.save_local("faiss_index")
    return vector_store_faiss
    
    
if __name__=="__main__":
    docs=data_ingestion()
    get_vector_store(docs)
    