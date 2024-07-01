import streamlit as st
from langchain.vectorstores import FAISS
from qa.retrieval_and_generation import get_llama3_llm, get_response_llm
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# Initialize FastEmbedEmbeddings model
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Set page configuration first
st.set_page_config(page_title="QA with DOC", page_icon="📚")

st.header("QA with DOC")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input: Placeholder where user will input their prompt
# Chat message: Display all input messages by user and response from assistant
if prompt := st.chat_input("Ask a question from the PDF files"):
    # Add user message to session state
    user_msg = {"role": "user", "content": prompt}
    st.session_state["messages"].append(user_msg)
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            # Load FAISS index and get response
            faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            llm = get_llama3_llm()
            response = get_response_llm(llm, faiss_index, prompt)
            st.markdown(response)
            
            # Add assistant message to session state
            assistant_msg = {"role": "assistant", "content": response}
            st.session_state["messages"].append(assistant_msg)
