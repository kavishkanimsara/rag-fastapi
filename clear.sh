#!/bin/bash

echo "🧹 Clearing RAG FastAPI data..."

# Stop any running uvicorn processes
echo "Stopping any running servers..."
pkill -f uvicorn 2>/dev/null || true

# Clear the ChromaDB directory
echo "Clearing ChromaDB data..."
rm -rf chroma_db

# Clear Python cache
echo "Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Ingest documents
echo "📚 Ingesting documents..."
python ingest.py

# Start the FastAPI application
echo "🚀 Starting FastAPI application..."
uvicorn main:app --host 0.0.0.0 --port 8000 --reload 