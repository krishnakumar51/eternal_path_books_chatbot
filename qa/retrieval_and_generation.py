from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Initialize HuggingFaceEmbeddings with a suitable model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Prompt template for the ConversationalRetrievalChain
prompt_template = """
Use the following pieces of context to provide a concise answer to the question asked by the user. 
Base your answer strictly on the provided context. If the information is not in the context, say "I don't have information about that in my knowledge base."

<context>
{context}
</context>

Question: {question}

Answer:"""

# Create a PromptTemplate instance
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
        output_key="answer"  # Explicitly set the output key
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
    
    sources = set([doc.metadata.get('source', 'Unknown') for doc in source_docs])
    source_attribution = "\n\nSources:\n" + "\n".join(sources)
    
    return answer + source_attribution

def get_keyword_suggestions(query, vectorstore_faiss, top_n=5):
    tfidf = TfidfVectorizer(stop_words='english')
    query_tfidf = tfidf.fit_transform([query])
    
    feature_names = np.array(tfidf.get_feature_names_out())
    tfidf_scores = query_tfidf.toarray()[0]
    
    sorted_idx = np.argsort(tfidf_scores)[::-1]
    keywords = feature_names[sorted_idx][:3]
    
    relevant_docs = []
    for keyword in keywords:
        docs = vectorstore_faiss.similarity_search(keyword, k=2)
        relevant_docs.extend(docs)
    
    potential_questions = []
    for doc in relevant_docs:
        sentences = doc.page_content.split('.')
        for sentence in sentences:
            if '?' in sentence:
                potential_questions.append(sentence.strip())
    
    if potential_questions:
        questions_tfidf = tfidf.transform(potential_questions)
        similarities = cosine_similarity(query_tfidf, questions_tfidf)
        
        sorted_idx = np.argsort(similarities[0])[::-1]
        top_questions = [potential_questions[i] for i in sorted_idx[:top_n]]
        return top_questions
    else:
        return []

if __name__ == '__main__':
    # Load FAISS index with embeddings
    faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    query = "What are the risk factors associated with Google and Tesla?"
    llm = get_groq_llm()
    chat_history = []  # Initialize an empty chat history
    print(get_response_llm(llm, faiss_index, query, chat_history))
