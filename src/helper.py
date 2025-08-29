from huggingface_hub import InferenceClient
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document

# Extract text from PDF files
def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(Document(page_content=doc.page_content, metadata={"source": src}))
    return minimal_docs

# Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# --- NEW: Hugging Face Inference API client ---
_hf_client = None

def download_hugging_face_embeddings():
    """Return an embedding function using Hugging Face Inference API"""
    global _hf_client
    if _hf_client is None:
        hf_token = os.getenv("HF_API_TOKEN")
        _hf_client = InferenceClient("sentence-transformers/paraphrase-MiniLM-L3-v2", token=hf_token)
    return _hf_client.feature_extraction
