# RAG-assignment

A Streamlit-based generative AI project for Retrieval-Augmented Generation (RAG).

## Description

This project implements a RAG system using Streamlit, LangChain, FAISS and various NLP tools. It allows users to interact with an AI model that can retrieve relevant information and generate responses based on given queries.

## Features

- Retrieval-Augmented Generation using LangChain
- Interactive Streamlit interface
- Document processing with PyPDF
- Efficient vector storage with FAISS
- Integration with Groq for language model inference

## Installation

To set up this project, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/krishnakumar51/RAG-assginment.git
cd RAG-assginment
```

2. Create and activate a Conda environment:
```bash
conda create -n venv python=3.9 -y
conda activate venv
```

3. Install the required dependencies:
```bash 
pip install -r requirements.txt
```

4. Set up your environment variables:
Create a `.env` file in the root directory and add your Groq API key:
```bash 
GROQ_API_KEY=your_api_key_here
```


## Usage

To run the Streamlit app locally:
```bash
streamlit run app.py
```
Navigate to the URL provided by Streamlit in your web browser to interact with the application.

## Dependencies

- Python >=3.9
- langchain
- langchainhub
- langchain_community
- sentence-transformers
- transformers
- streamlit
- pypdf
- faiss-cpu
- langchain-groq

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.


## Contact

Krishna Kumar - godkrishnasskal@gmail.com

Project Link: [https://github.com/krishnakumar51/RAG-assginment](https://github.com/krishnakumar51/RAG-assginment)