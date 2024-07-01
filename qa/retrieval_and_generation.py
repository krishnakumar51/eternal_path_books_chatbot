from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.llms import Ollama

embeddings=FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question asked bu the user. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT=PromptTemplate(
    template=prompt_template,input_variables=["context","question"]
)

def get_llama3_llm():
    llm=Ollama(model="llama3")
    return llm

def get_response_llm(llm,vectorstore_faiss,query):
    qa=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k":5}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt":PROMPT}
        
        
    )
    answer=qa.invoke({"query":query})
    return answer["result"]
    
if __name__=='__main__':
    faiss_index=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
    query="What are the risk factors associated with Google and Tesla?"
    llm=get_llama3_llm()
    print(get_response_llm(llm,faiss_index,query))