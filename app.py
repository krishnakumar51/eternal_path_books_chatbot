import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from qa.retrieval_and_generation import get_response_llm, get_groq_llm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

# Set environment variable for GROQ API key
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

if "GROQ_API_KEY" not in os.environ:
    st.error("Please set the GROQ_API_KEY environment variable.")
    st.stop()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Streamlit app setup
st.set_page_config(page_title="Eternal Path Books Customer Service", page_icon="📚")
st.title("📚 Eternal Path Books Customer Service")

# Initialize session state for conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Initialize the Groq LLM
faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
llm = get_groq_llm()


# Display existing messages
for message in st.session_state.conversation_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("How can I assist you with Eternal Path Books?"):
    st.session_state.conversation_history.append({"role": "human", "content": prompt})
    with st.chat_message("human"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Convert the conversation history to the required format
            chat_history = [
                HumanMessage(content=msg["content"]) if msg["role"] == "human" else AIMessage(content=msg["content"])
                for msg in st.session_state.conversation_history
            ]
            
            # Assume the vector store is already created and available
            vector_store = st.session_state.get("vector_store")
            
            response = get_response_llm(llm, prompt, chat_history, faiss_index)
            
            st.markdown(response)
            st.session_state.conversation_history.append({"role": "assistant", "content": response})

