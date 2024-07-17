from setuptools import setup, find_packages

setup(
    name="RAG-assignment",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchainhub",
        "langchain_community",
        "sentence-transformers",
        "transformers",
        "streamlit",
        "pypdf",
        "faiss-cpu",
        "langchain-groq",
    ],
    python_requires=">=3.9",
    description="A Streamlit-based generative AI project",
    author="Krishna Kumar",
    author_email="godkrishnasskal@gmail.com",
    url="https://github.com/krishnakumar51/RAG-assginment.git",
)