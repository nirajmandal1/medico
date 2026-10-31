from dotenv import load_dotenv
import os
import sys
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

if not PINECONE_API_KEY or PINECONE_API_KEY.startswith("xxxx"):
    print("[ERROR] PINECONE_API_KEY is not set or contains placeholder value in .env file.")
    print("Please add your valid Pinecone API key to .env:")
    print("PINECONE_API_KEY=your_key_here")
    sys.exit(1)

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

print("Loading PDF files from data/ directory...")
extracted_data = load_pdf_file(data='data/')
print(f"Extracted {len(extracted_data)} pages from PDF documents.")

print("Filtering documents to essential metadata...")
filter_data = filter_to_minimal_docs(extracted_data)

print("Splitting text into chunks...")
text_chunks = text_split(filter_data)
print(f"Created {len(text_chunks)} text chunks.")

print("Initializing HuggingFace embeddings model (sentence-transformers/all-MiniLM-L6-v2)...")
embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = os.environ.get("PINECONE_INDEX_NAME", "medical-chatbot")

try:
    existing_index_names = [idx.name for idx in pc.list_indexes()]
except Exception:
    try:
        existing_index_names = [idx["name"] for idx in pc.list_indexes()]
    except Exception:
        existing_index_names = []

if index_name not in existing_index_names:
    print(f"Index '{index_name}' not found. Creating serverless index...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Index '{index_name}' created successfully!")
else:
    print(f"Found existing Pinecone index: '{index_name}'.")

index = pc.Index(index_name)
docsearch = PineconeVectorStore(index=index, embedding=embeddings)

batch_size = 100
total_chunks = len(text_chunks)
print(f"Upserting embeddings for {total_chunks} chunks in batches of {batch_size}...")

for i in range(0, total_chunks, batch_size):
    batch = text_chunks[i:i + batch_size]
    docsearch.add_documents(batch)
    print(f"Progress: [{min(i + batch_size, total_chunks)}/{total_chunks}] chunks uploaded to Pinecone.")

print("\n[SUCCESS] All documents successfully indexed and stored in Pinecone!")
