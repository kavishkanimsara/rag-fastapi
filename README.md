# RAG FastAPI App

A FastAPI application for RAG (Retrieval-Augmented Generation) with document ingestion, vector storage, and similarity search capabilities using Google's Gemini AI.

## Features

- FastAPI framework with automatic API documentation
- RAG (Retrieval-Augmented Generation) functionality
- Document ingestion and chunking
- Vector storage with ChromaDB
- Semantic similarity search using Gemini embeddings
- RESTful API endpoints for both items and documents
- Interactive API documentation (Swagger UI)
- Free tier support with Google Gemini API

## Prerequisites

- Python 3.8+
- Google API key (for Gemini embeddings) - Free tier available

## Installation

1. Clone the repository and navigate to the project directory:
```bash
cd rag-fastapi
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables:
   - Copy the `.env.example` file to `.env`
   - Add your Google API key to the `.env` file:
```bash
GOOGLE_API_KEY=your_actual_google_api_key_here
```

## Getting Your Google API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key
5. Add it to your `.env` file

**Note**: Google Gemini API offers a generous free tier, making it perfect for development and testing!

## Running the Application

### Option 1: Using uvicorn directly
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 2: Using Python
```bash
python main.py
```

## Document Ingestion

Before using the RAG functionality, you need to ingest documents into the vector database:

1. Place your text documents in the `data/` directory (e.g., `data/sample.txt`, `data/country_knowledge.txt`)

2. Run the ingestion script:
```bash
python ingest.py
```

This will:
- Load all `.txt` files from the `data/` directory
- Split them into chunks using RecursiveCharacterTextSplitter
- Create embeddings using Google Gemini
- Store them in ChromaDB

## API Endpoints

### RAG Endpoints

- `POST /documents` - Add a document to the vector database
  ```bash
  curl -X POST "http://localhost:8000/documents" \
       -H "Content-Type: application/json" \
       -d '{"content": "Your document content here", "metadata": {"source": "example"}}'
  ```

- `POST /query` - Query documents for similarity
  ```bash
  curl -X POST "http://localhost:8000/query" \
       -H "Content-Type: application/json" \
       -d '{"query": "What is Sri Lanka known for?", "n_results": 3}'
  ```

- `GET /documents` - List all documents in the database
  ```bash
  curl "http://localhost:8000/documents"
  ```

- `DELETE /documents/{document_id}` - Delete a specific document
  ```bash
  curl -X DELETE "http://localhost:8000/documents/doc_1"
  ```

- `DELETE /documents` - Clear all documents
  ```bash
  curl -X DELETE "http://localhost:8000/documents"
  ```

### Basic Endpoints

- `GET /` - Welcome message
- `GET /health` - Health check

## API Documentation

Once the application is running, you can access:
- Interactive API documentation: http://localhost:8000/docs
- Alternative API documentation: http://localhost:8000/redoc

## Example Usage

### 1. Ingest Documents
```bash
# Make sure you have documents in the data/ directory
python ingest.py
```

### 2. Add a Document via API
```bash
curl -X POST "http://localhost:8000/documents" \
     -H "Content-Type: application/json" \
     -d '{
       "content": "FastAPI is a modern, fast web framework for building APIs with Python based on standard Python type hints.",
       "metadata": {"source": "documentation", "category": "framework"}
     }'
```

### 3. Query Documents
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What is Sri Lanka known for?",
       "n_results": 3
     }'
```

### 4. List All Documents
```bash
curl "http://localhost:8000/documents"
```

## Project Structure

```
rag-fastapi/
├── main.py              # FastAPI application with RAG endpoints
├── ingest.py            # Document ingestion script
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create this)
├── data/                # Directory for text documents
│   ├── sample.txt       # Sample fictional story
│   └── country_knowledge.txt  # Country domain knowledge
├── chroma_db/           # ChromaDB storage (created automatically)
└── README.md           # This file
```

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Google API Configuration
GOOGLE_API_KEY=your_google_api_key_here

# Optional: ChromaDB configuration
CHROMA_DB_PATH=./chroma_db
```

### ChromaDB Settings

The application uses ChromaDB for vector storage with the following default settings:
- Persistence: Enabled (stored in `./chroma_db`)
- Distance metric: Cosine similarity
- Collection name: "documents"

## Troubleshooting

### Common Issues

1. **Google API Key Error**
   - Make sure you have set your Google API key in the `.env` file
   - Get your API key from: https://makersuite.google.com/app/apikey
   - The app will use mock embeddings if no API key is provided

2. **ChromaDB Issues**
   - If ChromaDB fails to start, try deleting the `chroma_db` directory and restarting
   - Make sure you have write permissions in the project directory

3. **Import Errors**
   - Make sure you're using the virtual environment: `source venv/bin/activate`
   - Reinstall dependencies: `pip install -r requirements.txt`

## Development

### Adding New Features

1. **New RAG Endpoints**: Add them to `main.py` in the RAG Endpoints section
2. **New Document Types**: Modify `ingest.py` to support different file formats
3. **Custom Embeddings**: Update the embeddings configuration in both files

### Testing

Test the API endpoints using the interactive documentation at http://localhost:8000/docs

## License

This project is open source and available under the MIT License. 