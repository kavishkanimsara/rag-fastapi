from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import tiktoken
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="RAG FastAPI App",
    description="A FastAPI application for RAG (Retrieval-Augmented Generation) with enhanced prompts and context-aware responses",
    version="1.0.0"
)

# Initialize text splitter with optimized settings
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Smaller chunks for better precision
    chunk_overlap=150,  # Increased overlap for better context
    length_function=len,
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]  # Better separators
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

# Simplified prompt templates
CONTEXT_ANALYSIS_PROMPT = """
Answer this question using ONLY the information provided below:

Question: {query}

Information:
{context}

Instructions:
- Use only the provided information
- Give a clear, direct answer (1-2 sentences)
- If no relevant information is found, say "I couldn't find information about this in the provided data"
- Don't add external knowledge or explanations

Answer:"""

QUERY_CLASSIFICATION_PROMPT = """
Classify this query into one category:
- FACTUAL: what, when, where, who, how many
- COMPARATIVE: compare, difference, versus, similar
- ANALYTICAL: why, explain, analyze, interpret
- PROCEDURAL: how to, steps, process, method
- OPINION: opinion, think, believe, feel

Query: {query}
Category:"""


class Item(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None
    price: float

class DocumentRequest(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None
    title: Optional[str] = None
    category: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    n_results: Optional[int] = 5
    include_analysis: Optional[bool] = False

class DocumentResponse(BaseModel):
    id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    distance: Optional[float] = None
    relevance_score: Optional[float] = None

class EnhancedQueryResponse(BaseModel):
    original_query: str
    enhanced_query: Optional[str] = None
    query_category: Optional[str] = None
    documents: List[DocumentResponse]
    analysis: Optional[str] = None
    total_results: int
    search_metadata: Dict[str, Any]

# In-memory storage for demo purposes
items_db = []
item_id_counter = 1

def enhance_query(query: str) -> str:
    """Enhance the user query for better search results"""
    # Simple query enhancement logic
    enhanced_terms = []
    
    # Add common synonyms and related terms for any type of content
    synonyms = {
        "climate": ["weather", "temperature", "rainfall", "monsoon", "seasonal", "atmospheric"],
        "food": ["cuisine", "dishes", "cooking", "recipes", "traditional food", "culinary"],
        "culture": ["traditions", "customs", "heritage", "festivals", "religion", "society"],
        "economy": ["business", "trade", "industries", "exports", "economic", "financial"],
        "geography": ["location", "terrain", "landscape", "mountains", "coast", "physical"],
        "history": ["historical", "ancient", "past", "heritage", "civilization", "events"],
        "language": ["languages", "speaking", "communication", "dialects", "linguistic"],
        "wildlife": ["animals", "nature", "biodiversity", "parks", "conservation", "fauna"],
        "technology": ["tech", "digital", "software", "hardware", "innovation", "computing"],
        "science": ["scientific", "research", "experiments", "discoveries", "methodology"],
        "health": ["medical", "healthcare", "wellness", "medicine", "treatment", "disease"],
        "education": ["learning", "teaching", "academic", "school", "university", "training"],
        "politics": ["government", "political", "policy", "elections", "democracy", "leadership"],
        "sports": ["athletics", "games", "competition", "fitness", "recreation", "olympics"],
        "art": ["artistic", "creative", "painting", "sculpture", "music", "literature"],
        "architecture": ["buildings", "design", "construction", "structures", "urban planning"],
        "transportation": ["travel", "vehicles", "infrastructure", "mobility", "logistics"],
        "environment": ["ecological", "sustainability", "pollution", "conservation", "climate change"],
        "social": ["society", "community", "relationships", "social issues", "demographics"],
        "business": ["corporate", "commerce", "enterprise", "management", "marketing"]
    }
    
    query_lower = query.lower()
    for term, related in synonyms.items():
        if term in query_lower:
            enhanced_terms.extend(related[:2])  # Add top 2 related terms
    
    if enhanced_terms:
        enhanced_query = f"{query} {' '.join(enhanced_terms)}"
        return enhanced_query
    
    return query

def classify_query(query: str) -> str:
    """Classify the type of query"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["what", "when", "where", "who", "how many", "how much"]):
        return "FACTUAL"
    elif any(word in query_lower for word in ["compare", "difference", "versus", "vs", "better", "similar"]):
        return "COMPARATIVE"
    elif any(word in query_lower for word in ["why", "explain", "analyze", "interpret", "cause"]):
        return "ANALYTICAL"
    elif any(word in query_lower for word in ["how to", "steps", "process", "procedure", "method"]):
        return "PROCEDURAL"
    elif any(word in query_lower for word in ["opinion", "think", "believe", "feel", "view"]):
        return "OPINION"
    else:
        return "FACTUAL"

def calculate_relevance_score(distance: float) -> float:
    """Convert distance to relevance score (0-1, higher is better)"""
    if distance is None:
        return 0.5
    # Convert distance to relevance score (inverse relationship)
    return max(0, 1 - distance)

def analyze_context(query: str, documents: List[DocumentResponse]) -> str:
    """Provide simple context analysis"""
    if not documents:
        return "No relevant documents found."
    
    # Extract key information from documents
    content_summary = []
    for doc in documents[:2]:  # Use top 2 documents
        content_summary.append(f"- {doc.content[:150]}...")
    
    context_text = "\n".join(content_summary)
    
    analysis = f"""
Query: "{query}"
Found {len(documents)} relevant documents.

Key information:
{context_text}
"""
    
    return analysis

@app.get("/")
async def root():
    return {
        "message": "Welcome to Enhanced RAG FastAPI App!", 
        "status": "running",
        "features": [
            "Enhanced query processing",
            "Context-aware responses", 
            "Query classification",
            "Relevance scoring",
            "Document analysis",
            "Universal document support"
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "rag-fastapi", "version": "2.0.0"}

# RAG Endpoints
@app.post("/documents", response_model=Dict[str, Any])
async def add_document(document: DocumentRequest):
    """Add a document to the vector database with enhanced metadata"""
    try:
        # Split the document into chunks
        chunks = text_splitter.split_text(document.content)
        
        # Enhanced metadata
        enhanced_metadata = document.metadata or {}
        enhanced_metadata.update({
            "title": document.title or "Untitled",
            "category": document.category or "general",
            "chunk_count": len(chunks),
            "total_length": len(document.content)
        })
        
        # Add chunks to ChromaDB using LangChain
        vectorstore.add_texts(
            texts=chunks,
            metadatas=[enhanced_metadata for _ in chunks]
        )
        
        return {
            "message": f"Document added successfully with {len(chunks)} chunks",
            "chunks_created": len(chunks),
            "metadata": enhanced_metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding document: {str(e)}")

@app.post("/query", response_model=EnhancedQueryResponse)
async def query_documents(query_request: QueryRequest):
    """Simple query endpoint that returns clear, direct answers"""
    try:
        # Classify query
        query_category = classify_query(query_request.query)
        
        # Use LangChain's similarity search with original query
        docs_and_scores = vectorstore.similarity_search_with_score(
            query_request.query,
            k=query_request.n_results
        )
        
        documents = []
        for doc, score in docs_and_scores:
            relevance_score = calculate_relevance_score(score)
            doc_response = DocumentResponse(
                id=str(hash(doc.page_content))[:8],
                content=doc.page_content,
                metadata=doc.metadata,
                distance=float(score) if score is not None else None,
                relevance_score=relevance_score
            )
            documents.append(doc_response)
        
        # Generate simple analysis if requested
        analysis = None
        if query_request.include_analysis:
            analysis = analyze_context(query_request.query, documents)
        
        return EnhancedQueryResponse(
            original_query=query_request.query,
            enhanced_query=None,  # No enhancement
            query_category=query_category,
            documents=documents,
            analysis=analysis,
            total_results=len(documents),
            search_metadata={
                "query_enhanced": False,
                "analysis_included": query_request.include_analysis,
                "search_strategy": "similarity_search_with_score"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")

@app.get("/documents", response_model=Dict[str, Any])
async def list_documents():
    """List all documents in the vector database with enhanced information"""
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
                "metadata": all_docs['metadatas'][i] if 'metadatas' in all_docs else None,
                "length": len(doc)
            }
            documents.append(doc_info)
        
        # Calculate statistics
        total_length = sum(len(doc) for doc in all_docs['documents'])
        avg_length = total_length / len(all_docs['documents']) if all_docs['documents'] else 0
        
        return {
            "total_documents": len(all_docs['documents']),
            "documents": documents,
            "statistics": {
                "total_content_length": total_length,
                "average_document_length": round(avg_length, 2),
                "collection_size_mb": round(total_length * 0.000001, 2)
            }
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
