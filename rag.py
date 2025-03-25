import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import streamlit as st

# Set the NVIDIA API key
os.environ["NVIDIA_API_KEY"] = "nvapi-xxx"

def truncate_input(input_text, max_tokens=3072):
    # Truncate the input to fit within the model's token limit
    words = input_text.split()
    truncated_words = words[:max_tokens]
    return " ".join(truncated_words)

def main():
    st.title("Run.ai Chatbot")

    # Load the vectorstore
    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")
    if os.path.exists("vectorstore.faiss"):
        vectorstore = FAISS.load_local("vectorstore.faiss", embeddings, allow_dangerous_deserialization=True)
    else:
        st.error("Vectorstore not found! Run data_prep.py first.")
        return

    # Session state for chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for message in st.session_state["messages"]:
        st.write(message)

    # User input
    user_input = st.text_input(
        "Ask a question about Run.ai:", 
        placeholder="e.g., What are the steps for cluster setup?"
    )
    if user_input:
        with st.spinner("Generating response..."):
            # Retrieve relevant documents
            docs = vectorstore.similarity_search(user_input)
            context = " ".join([doc.page_content for doc in docs])

            # Combine user input and context
            full_input = f"Answer the following question about Run.ai:\nQuestion: {user_input}\nContext: {context}"

            # Truncate input to fit the model's token limit
            truncated_input = truncate_input(full_input)

            # Generate response
            llm = ChatNVIDIA(model="nvidia/neva-22b")
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
            response = qa_chain(truncated_input)

            # Save and display the response
            st.session_state["messages"].append(f"**User**: {user_input}")
            st.session_state["messages"].append(f"**Bot**: {response['result']}")
            st.write(response['result'])

if __name__ == "__main__":
    main()
