import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from google import genai

# -------------------------
# PDF utilities
# -------------------------
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

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# -------------------------
# Google Gemini Embeddings
# -------------------------
class GoogleGeminiEmbeddings(Embeddings):
    def __init__(self, model="models/embedding-001", api_key=None):
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY in environment variables")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def embed_query(self, text: str):
        response = self.client.models.embed_content(
            model=self.model,
            contents=text
        )
        return response.embeddings[0].values

    def embed_documents(self, texts):
        vectors = []
        for t in texts:
            response = self.client.models.embed_content(
                model=self.model,
                contents=t
            )
            vectors.append(response.embeddings[0].values)
        return vectors


def download_google_embeddings():
    """Return a LangChain-compatible embeddings object using Google Gemini API"""
    return GoogleGeminiEmbeddings()
