import os
import glob
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from dotenv import load_dotenv

load_dotenv()

# Check if Google API key is set
if not os.getenv("GOOGLE_API_KEY"):
    print("Warning: Please set your Google API key in the .env file")
    print("You can get one from: https://makersuite.google.com/app/apikey")
    print("For now, using a mock embedding function...")
    # Create a mock embedding function for testing
    class MockEmbeddings:
        def embed_documents(self, texts):
            return [[0.1] * 768 for _ in texts]
        def embed_query(self, text):
            return [0.1] * 768
    
    embeddings = MockEmbeddings()
else:
    # Use Gemini embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

# 1. Load the data
docs = []
for file in glob.glob("data/*.txt"):
    loader = TextLoader(file, encoding="utf-8")
    docs.extend(loader.load())

print(f"Loaded {len(docs)} documents.")

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Filter complex metadata from chunks
filtered_chunks = filter_complex_metadata(chunks)

print(f"Split into {len(chunks)} chunks.")
print(f"Filtered metadata for {len(filtered_chunks)} chunks.")

# 3. Create embeddings and store in ChromaDB
# Create or get the ChromaDB collection
vectorstore = Chroma.from_documents(
    documents=filtered_chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Persist the vectorstore
# vectorstore.persist()  # No longer needed in ChromaDB 0.4.x+

print(f"Successfully stored {len(chunks)} chunks in ChromaDB")
print("Vector database is ready for querying!")
