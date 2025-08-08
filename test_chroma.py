import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize embeddings
if os.getenv("GOOGLE_API_KEY"):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
else:
    # Mock embeddings for testing
    class MockEmbeddings:
        def embed_documents(self, texts):
            return [[0.1] * 768 for _ in texts]
        def embed_query(self, text):
            return [0.1] * 768
    
    embeddings = MockEmbeddings()

# Initialize ChromaDB using LangChain
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="documents"
)

# Test getting all documents
print("Testing ChromaDB connection...")
try:
    all_docs = vectorstore.get()
    print(f"Total documents: {len(all_docs['documents']) if all_docs and 'documents' in all_docs else 0}")
    
    if all_docs and 'documents' in all_docs and all_docs['documents']:
        print("First few documents:")
        for i, doc in enumerate(all_docs['documents'][:3]):
            print(f"Document {i+1}: {doc[:100]}...")
    else:
        print("No documents found in the database")
        
    # Test similarity search
    print("\nTesting similarity search...")
    results = vectorstore.similarity_search_with_score("Sri Lanka climate", k=3)
    print(f"Found {len(results)} results")
    
    for i, (doc, score) in enumerate(results):
        print(f"Result {i+1} (score: {score}): {doc.page_content[:100]}...")
        
except Exception as e:
    print(f"Error: {e}") 