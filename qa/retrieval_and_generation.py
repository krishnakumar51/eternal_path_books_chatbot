import streamlit as st 
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from qa.ingestion import load_vector_store
import os

# Import functions from your existing file
from qa.ingestion import data_ingestion, get_vector_store

# Set environment variable for GROQ API key
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

if "GROQ_API_KEY" not in os.environ:
    st.error("Please set the GROQ_API_KEY environment variable.")
    st.stop()

# Initialize the Groq LLM
def get_groq_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    return ChatGroq(groq_api_key=api_key, model_name="llama3-8b-8192")

# Function to get response from LLM with chat history
def get_response_llm(llm, query, chat_history, vector_store):
    # Define the retriever prompt
    retriever_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )

    # Create the contextualize question prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", retriever_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Create a history-aware retriever
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Define the system prompt for question answering
    system_prompt = (
   "You are a concise and polite customer service bot, designed to help disciples find the teachings of the Guru and guide them to related books or resources. "
    "Always respond briefly, accurately, and in a composed manner. Keep answers short and to the point, without unnecessary details. "
    "Greet the user with 'Sat Sri Akal Ji' if it's their first message. "
    "If the user asks about a book or Vyakhya, ask them: 'Would you like to buy the book or try a few verses from the digital version?' "
    "1. If the user agrees to buy:\n"
        "   - Provide a WhatsApp link for purchasing the physical book.\n"
        "2. If the user prefers to read the digital version:\n"
        "   - Search the knowledge base and examples given for the link to the digital version.\n"
        "3. If the user declines both options:\n"
        "   - Offer a brief description of the book’s value and its relevance to Sikh spiritual teachings.\n\n"
        "4. If the information is not available:\n"
        "   - Respond with: 'I'm sorry, please visit our customer service on WhatsApp for more details.'\n\n"
        "5. Whenever you are responding about to talk or connect to our customer service\n"
        "   - Include the whatsapp link in the response as [WhatsApp Order Link](https://api.whatsapp.com/send/?phone=9105575000)'\n\n"
        "**System Prompts for Actions:**\n"
        "- **For Buying the Physical Book:**\n"
        "  - Provide a fixed WhatsApp link for purchasing the physical book. Example: 'You can buy [Book Title] for ₹[Price] by clicking this WhatsApp link: [WhatsApp Order Link](https://api.whatsapp.com/send/?phone=9105575000&text=I+want+to+order&type=phone_number&app_absent=0)'\n\n"
        "- **For Reading the Digital Version:**\n"
        "  - Search the knowledge base for the link to the digital version and provide it to the user. Example: 'Our digital version of [Book Title] is available for you to read. Play Store: [Andoird](Fetch the play store link for required book) | Apple Store:[iOS](Fetch the apple store link for required book) '\n\n"
        "**Examples:**\n"
        "- If a user wants to buy 'Sunn Galaa Aakaas Kee':\n"
        "  - 'You can buy 'Sunn Galaa Aakaas Kee' for ₹500 by clicking this WhatsApp link: [WhatsApp Order Link](https://api.whatsapp.com/send/?phone=9105575000&text=I+want+to+order&type=phone_number&app_absent=0)'\n\n"
        "- If a user wants to read the digital version of 'Sunn Galaa Aakaas Kee':\n"
        "  - 'Our digital version of 'Sunn Galaa Aakaas Kee' is available for you to read in both Play Store and Apple Store.\n Play Store: [Andoird](https://play.google.com/store/apps/details?id=com.books.sunn_galaa_aakaas_kee&pli=1) | Apple Store:[iOS](https://apps.apple.com/in/app/sunn-galaa-aakaas-kee/id6448728336) '\n\n"
        "- If a user declines both options:\n"
        "  - 'This book offers a profound understanding of Sikh teachings. If you have any other questions or need more details, feel free to ask.'"
        
        
        "- If a user wants to buy 'Sukhmani Sehaj Gobind Gunn Naam Part 1':\n"
        "  - 'You can buy 'Sukhmani Sehaj Gobind Gunn Naam' for ₹450 per part by clicking this WhatsApp link: [WhatsApp Order Link](https://api.whatsapp.com/send/?phone=9105575000&text=I+want+to+order&type=phone_number&app_absent=0)'\n\n"
        "- If a user wants to read the digital version of 'Sukhmani Sehaj Gobind Gunn Naam Part 1':\n"
        "  - 'Our digital version of 'Sukhmani Sehaj Gobind Gunn Naam Part 1' is available for you to read in both Play Store and Apple Store.\n Play Store: [Andoird](https://play.google.com/store/apps/details?id=com.books.sukhmani_sehaj_gobind_gunname&hl=en-IN) | Apple Store:[iOS](https://apps.apple.com/mx/app/sukhmani-sehaj-part-1/id6503642591?l=en-GB) '\n\n"
        "- If a user declines both options:\n"
        "  - 'This book offers a profound understanding of Sikh teachings. If you have any other questions or need more details, feel free to ask.'"

        "- If a user wants to buy 'Sukhmani Sehaj Gobind Gunn Naam Part 2':\n"
        "  - 'You can buy 'Sukhmani Sehaj Gobind Gunn Naam' for ₹450 per part by clicking this WhatsApp link: [WhatsApp Order Link](https://api.whatsapp.com/send/?phone=9105575000&text=I+want+to+order&type=phone_number&app_absent=0)'\n\n"
        "- If a user wants to read the digital version of 'Sukhmani Sehaj Gobind Gunn Naam Part 1':\n"
        "  - 'Our digital version of 'Sukhmani Sehaj Gobind Gunn Naam Part 1' is available for you to read in both Play Store and Apple Store.\n Play Store: [Andoird](https://play.google.com/store/apps/details?id=com.books.sukhmani_sehaj_gobind_gunname_part_2&hl=en-IN) | Apple Store: Coming soon'\n\n"
        "- If a user declines both options:\n"
        "  - 'This book offers a profound understanding of Sikh teachings. If you have any other questions or need more details, feel free to ask.'"    
        "And all these links are been provided in the knowledge base, just look for it."
    "If user is leaving, reply them that they can always stay connected with the Guru's teachings on Instagram ([https://www.instagram.com/eternalpathbooks/]), Facebook ([https://www.facebook.com/EternalPathBooks]), and with us for more updates and queries."
    "If you don't know the answer, respond with 'I'm sorry, please visit our customer service on WhatsApp for more details.' "
    "\n\n"
    "{context}"
)


    # Create the question answering prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Create the question answering chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create the final RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Invoke the chain with the query and chat history
    response = rag_chain.invoke({
        "input": query,
        "chat_history": chat_history
    })

    return response["answer"]
