from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
import os
import streamlit as st

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

prompt_template = """
Use the following pieces of context to provide a concise answer to the question asked by the user. 
Base your answer strictly on the provided context. If the information is not in the context, say "I don't have information about that in my knowledge base."

<context>
{context}
</context>

Question: {question}

Answer:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_groq_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    return ChatGroq(groq_api_key=api_key, model_name="mixtral-8x7b-32768")

def get_response_llm(llm, vectorstore_faiss, query, chat_history):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    result = qa({"question": query, "chat_history": chat_history})
    answer = result["answer"]
    source_docs = result["source_documents"]
    
    ranked_sources = []
    for i, doc in enumerate(source_docs[:3], 1):
        snippet = doc.page_content[:150].replace("\n", " ").strip() + "..."
        source = os.path.basename(doc.metadata.get('source', 'Unknown'))
        ranked_sources.append(f"{i}. **{source}**\n   {snippet}")
    
    sources_text = "\n\n".join(ranked_sources)
    
    return {
        "answer": answer,
        "sources": sources_text
    }