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

from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings
import os

class HuggingFaceAPIEmbeddings(Embeddings):
    def __init__(self, model="sentence-transformers/paraphrase-MiniLM-L6-v2", token=None):
        self.client = InferenceClient(token=token)
        self.model = model   # store model name

    def embed_query(self, text: str):
        result = self.client.post(
            model=self.model,
            task="feature-extraction",
            json={"inputs": text}
        )
        return result[0] if isinstance(result[0], list) else result

    def embed_documents(self, texts):
        embeddings = []
        for t in texts:
            result = self.client.post(
                model=self.model,
                task="feature-extraction",
                json={"inputs": t}
            )
            embeddings.append(result[0] if isinstance(result[0], list) else result)
        return embeddings

def download_hugging_face_embeddings():
    hf_token = os.getenv("HF_API_TOKEN")
    return HuggingFaceAPIEmbeddings(token=hf_token)
