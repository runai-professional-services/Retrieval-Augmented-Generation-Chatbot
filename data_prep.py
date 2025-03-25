import os
import requests
os.environ["NVIDIA_API_KEY"] = "nvapi-xxx"
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document

def load_data_from_directory(directory):
    text = ""
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r") as file:
                text += file.read()
    return text

def scrape_web_page(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise errors for bad status codes
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Extract ONLY the main content (adjust the selector based on the page structure)
        main_content = soup.find("div", class_="content")  # Inspect the page to confirm the class
        if main_content:
            return main_content.get_text(separator="\n", strip=True)
        else:
            return "Failed to find main content."
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"

def main():
    # Load local text files
    directory = "/home/local/data"
    local_text = load_data_from_directory(directory)
    
    # Scrape the Run.ai documentation
    url = "https://docs.run.ai/v2.20/admin/runai-setup/cluster-setup/cluster-setup-intro/"
    web_text = scrape_web_page(url)
    
    # Combine local and web text
    combined_text = local_text + "\n" + web_text
    
    # Split and process
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text_chunks = text_splitter.split_text(combined_text)
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    
    # Generate vectorstore
    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("vectorstore.faiss")

if __name__ == "__main__":
    main()
