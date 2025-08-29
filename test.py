import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings

if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    raise ValueError("❌ Missing HUGGINGFACEHUB_API_TOKEN")

emb = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

vec = emb.embed_query("Hello world")
print("Embedding length:", len(vec))
print("First 5 numbers:", vec[:5])
