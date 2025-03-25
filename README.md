# Retrieval-Augmented-Generation-Chatbot


Run:ai Chatbot
A Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about Run:ai cluster setup and installation using scraped documentation and NVIDIA AI models.


Scrapes content from the Run:ai documentation .
Uses a vector database (FAISS) to store and retrieve relevant information.
Leverages NVIDIA's nvidia/neva-22b language model to generate human-like responses to user queries.
The chatbot is built using:

Streamlit : For the interactive UI.
LangChain : For integrating embeddings, retrieval, and language models.
NVIDIA AI Endpoints : For generating embeddings and responses.
FAISS : For efficient similarity search in the knowledge base.
