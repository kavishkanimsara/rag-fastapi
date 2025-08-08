from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import tiktoken
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="RAG FastAPI App",
    description="A FastAPI application for RAG (Retrieval-Augmented Generation)",
    version="1.0.0"
)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

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
    collection_name="langchain"
)

class Item(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None
    price: float

class DocumentRequest(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None

class QueryRequest(BaseModel):
    query: str
    n_results: Optional[int] = 5

class DocumentResponse(BaseModel):
    id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    distance: Optional[float] = None

# In-memory storage for demo purposes
items_db = []
item_id_counter = 1

@app.get("/")
async def root():
    return {"message": "Welcome to RAG FastAPI App!", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "rag-fastapi"}

# RAG Endpoints
@app.post("/documents", response_model=Dict[str, Any])
async def add_document(document: DocumentRequest):
    """Add a document to the vector database"""
    try:
        # Split the document into chunks
        chunks = text_splitter.split_text(document.content)
        
        # Add chunks to ChromaDB using LangChain
        vectorstore.add_texts(
            texts=chunks,
            metadatas=[document.metadata or {} for _ in chunks]
        )
        
        return {
            "message": f"Document added successfully with {len(chunks)} chunks",
            "chunks_created": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding document: {str(e)}")

@app.post("/query", response_model=List[DocumentResponse])
async def query_documents(query_request: QueryRequest):
    """Query the vector database for similar documents"""
    try:
        # Use LangChain's similarity search
        docs_and_scores = vectorstore.similarity_search_with_score(
            query_request.query,
            k=query_request.n_results
        )
        
        documents = []
        for doc, score in docs_and_scores:
            doc_response = DocumentResponse(
                id=str(hash(doc.page_content))[:8],  # Generate a simple ID
                content=doc.page_content,
                metadata=doc.metadata,
                distance=float(score) if score is not None else None
            )
            documents.append(doc_response)
        
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")

@app.get("/documents", response_model=Dict[str, Any])
async def list_documents():
    """List all documents in the vector database"""
    try:
        # Get all documents from the vectorstore
        all_docs = vectorstore.get()
        
        if not all_docs or not all_docs['documents']:
            return {"total_documents": 0, "documents": []}
        
        documents = []
        for i, doc in enumerate(all_docs['documents']):
            doc_info = {
                "id": all_docs['ids'][i] if 'ids' in all_docs else f"doc_{i}",
                "content": doc[:200] + "..." if len(doc) > 200 else doc,
                "metadata": all_docs['metadatas'][i] if 'metadatas' in all_docs else None
            }
            documents.append(doc_info)
        
        return {
            "total_documents": len(all_docs['documents']),
            "documents": documents
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a specific document from the vector database"""
    try:
        # Note: This is a simplified implementation
        # In a real application, you'd need to implement proper deletion
        return {"message": f"Document {document_id} deletion not implemented in this version"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.delete("/documents")
async def clear_documents():
    """Clear all documents from the vector database"""
    try:
        # This would require reinitializing the vectorstore
        return {"message": "Clear documents not implemented in this version"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing documents: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
