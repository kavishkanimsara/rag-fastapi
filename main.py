from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import TextLoader, PyPDFLoader
import tiktoken
from dotenv import load_dotenv
import tempfile
import shutil

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

# RAG Endpoints
@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document (PDF or text) to the vector database"""
    try:
        # Check file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ['.txt', '.pdf']:
            raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        try:
            # Load document based on type
            if file_ext == '.pdf':
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
            else:  # .txt
                loader = TextLoader(temp_path, encoding="utf-8")
                docs = loader.load()
            
            # Split into chunks
            chunks = text_splitter.split_documents(docs)
            
            # Filter complex metadata
            filtered_chunks = filter_complex_metadata(chunks)
            
            # Add to vectorstore
            vectorstore.add_documents(filtered_chunks)
            
            return {
                "message": f"Document '{file.filename}' processed successfully",
                "chunks_created": len(chunks),
                "file_type": file_ext,
                "file_size": len(chunks) * 500  # Approximate size
            }
            
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.post("/query")
async def simple_query(query_request: QueryRequest):
    """Simple query endpoint that uses Gemini LLM to enhance responses"""
    try:
        # Use LangChain's similarity search with more results to find relevant chunks
        docs_and_scores = vectorstore.similarity_search_with_score(
            query_request.query,
            k=10  # Get more results to find the right chunk
        )
        
        if not docs_and_scores:
            return {"answer": "I couldn't find any information about that in my knowledge base."}
        
        # Get the most relevant documents
        relevant_docs = []
        for doc, score in docs_and_scores[:3]:  # Top 3 most relevant
            relevant_docs.append({
                "content": doc.page_content,
                "relevance_score": calculate_relevance_score(score)
            })
        
        # Use Gemini LLM to generate enhanced response
        try:
            if os.getenv("GOOGLE_API_KEY"):
                # Initialize Gemini LLM
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                    temperature=0.3
                )
                
                # Create context from relevant documents
                context = "\n\n".join([f"Document {i+1} (Relevance: {doc['relevance_score']:.2f}):\n{doc['content']}" 
                                     for i, doc in enumerate(relevant_docs)])
                
                # Create prompt for Gemini
                prompt = f"""
                Based on the following documents, provide a clear and accurate answer to the user's question.
                
                User Question: {query_request.query}
                
                Relevant Documents:
                {context}
                
                Instructions:
                - Answer the question using ONLY the information from the provided documents
                - Be concise and short answers
                - If the documents don't contain enough information to answer the question, say so
                - Focus on the most relevant information from the documents
                - Don't add external knowledge or assumptions
                
                Answer:"""
                
                # Generate response using Gemini
                response = llm.invoke(prompt)
                enhanced_answer = response.content.strip()
                
                return {"answer": enhanced_answer}
                
            else:
                # Fallback to simple response if no API key
                best_doc = relevant_docs[0]
                content = best_doc["content"]
                
                if len(content) > 200:
                    answer = content[:200].strip()
                    if not answer.endswith(('.', '!', '?')):
                        answer += "..."
                else:
                    answer = content.strip()
                
                return {"answer": answer}
                
        except Exception as llm_error:
            # Fallback if LLM fails
            print(f"LLM error: {llm_error}")
            best_doc = relevant_docs[0]
            content = best_doc["content"]
            
            if len(content) > 200:
                answer = content[:200].strip()
                if not answer.endswith(('.', '!', '?')):
                    answer += "..."
            else:
                answer = content.strip()
            
            return {"answer": answer}
        
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
            metadata = all_docs['metadatas'][i] if 'metadatas' in all_docs else {}
            source_type = metadata.get('source_type', 'unknown')
            
            doc_info = {
                "id": all_docs['ids'][i] if 'ids' in all_docs else f"doc_{i}",
                "content": doc[:200] + "..." if len(doc) > 200 else doc,
                "metadata": metadata,
                "length": len(doc),
                "source_type": source_type
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
