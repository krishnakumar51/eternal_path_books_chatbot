import streamlit as st
from langchain.vectorstores import FAISS
from qa.retrieval_and_generation import get_groq_llm, get_response_llm, get_keyword_suggestions
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Ensure GROQ_API_KEY is set
if "GROQ_API_KEY" not in os.environ:
    st.error("Please set the GROQ_API_KEY environment variable.")
    st.stop()

# Initialize HuggingFaceEmbeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Set page configuration
st.set_page_config(page_title="QA with DOC", page_icon="📚")

st.header("QA with DOC")

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = {"messages": [], "chat_history": []}

# Sidebar for conversation management
with st.sidebar:
    st.subheader("Conversations")
    if st.button("Start New Chat"):
        st.session_state.conversation_history.append(st.session_state.current_conversation)
        st.session_state.current_conversation = {"messages": [], "chat_history": []}
        st.experimental_rerun()
    
    for i, conv in enumerate(st.session_state.conversation_history):
        if st.button(f"Load Chat {i+1}"):
            st.session_state.current_conversation = conv
            st.experimental_rerun()

# Main chat interface
messages = st.session_state.current_conversation["messages"]
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load FAISS index and LLM
faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
llm = get_groq_llm()

# Chat input and processing
if prompt := st.chat_input("Ask a question from the PDF files"):
    # Add user message to current conversation
    user_msg = {"role": "user", "content": prompt}
    st.session_state.current_conversation["messages"].append(user_msg)
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get keyword suggestions
    suggestions = get_keyword_suggestions(prompt, faiss_index)
    
    # Display suggestions if available
    if suggestions:
        st.write("Related questions:")
        for i, suggestion in enumerate(suggestions, 1):
            if st.button(f"{i}. {suggestion}", key=f"suggestion_{i}"):
                prompt = suggestion
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            # Convert the chat history to the format expected by get_response_llm
            chat_history = [(msg["content"], msg["content"]) for msg in st.session_state.current_conversation["messages"] if msg["role"] == "assistant"]
            
            response = get_response_llm(llm, faiss_index, prompt, chat_history)
            st.markdown(response)
            
            # Add assistant message to current conversation
            assistant_msg = {"role": "assistant", "content": response}
            st.session_state.current_conversation["messages"].append(assistant_msg)
    
    # Force a rerun to update the display
    st.experimental_rerun()