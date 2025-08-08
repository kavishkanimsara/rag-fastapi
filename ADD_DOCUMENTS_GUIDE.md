# How to Add New Text Files with Content

This guide shows you how to add new text files with content to your RAG FastAPI application.

## Method 1: Manual File Creation

### Step 1: Create a New Text File
Create a new `.txt` file in the `data/` directory:

```bash
# Navigate to the data directory
cd data

# Create a new text file (you can use any text editor)
nano my_new_document.txt
# or
vim my_new_document.txt
# or use your preferred text editor
```

### Step 2: Add Content
Add your content to the file. For example:

```text
Your Document Title

This is the content of your document. You can add any text here - articles, 
research papers, documentation, stories, or any other textual content.

You can structure your content with:
- Bullet points
- Numbered lists
- Headers and subheaders
- Paragraphs

The more structured and well-organized your content is, the better the RAG system 
will be able to chunk and retrieve relevant information.
```

### Step 3: Save the File
Save the file in the `data/` directory with a `.txt` extension.

## Method 2: Using the API Endpoint

### Step 1: Prepare Your Content
Prepare your content as a JSON payload:

```json
{
  "content": "Your document content here. This can be any text content you want to add to the knowledge base.",
  "title": "My Document Title",
  "category": "technology",
  "metadata": {
    "source": "manual_input",
    "author": "Your Name",
    "date": "2024-01-01"
  }
}
```

### Step 2: Send the Request
Use curl or any HTTP client to send the content:

```bash
curl -X POST "http://localhost:8000/documents" \
     -H "Content-Type: application/json" \
     -d '{
       "content": "Your document content here...",
       "title": "My Document Title",
       "category": "technology",
       "metadata": {
         "source": "manual_input",
         "author": "Your Name"
       }
     }'
```

## Method 3: Using Python Script

### Step 1: Create a Python Script
Create a script to add documents programmatically:

```python
# add_document.py
import requests
import json

def add_document(content, title=None, category=None, metadata=None):
    url = "http://localhost:8000/documents"
    
    payload = {
        "content": content,
        "title": title,
        "category": category,
        "metadata": metadata or {}
    }
    
    response = requests.post(url, json=payload)
    return response.json()

# Example usage
if __name__ == "__main__":
    content = """
    This is my new document content.
    
    It can contain multiple paragraphs and structured information.
    
    The RAG system will automatically chunk this content and make it searchable.
    """
    
    result = add_document(
        content=content,
        title="My New Document",
        category="general",
        metadata={"source": "python_script", "author": "User"}
    )
    
    print(f"Document added: {result}")
```

### Step 2: Run the Script
```bash
python add_document.py
```

## Method 4: Bulk Import from Directory

### Step 1: Create a Bulk Import Script
```python
# bulk_import.py
import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def bulk_import_from_directory(directory_path="./data"):
    # Initialize components
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="langchain"
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    
    # Load all text files
    docs = []
    for file_path in glob.glob(f"{directory_path}/*.txt"):
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            docs.extend(loader.load())
            print(f"Loaded: {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not docs:
        print("No documents found to import.")
        return
    
    # Split documents into chunks
    chunks = text_splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    
    # Add to vectorstore
    vectorstore.add_documents(chunks)
    print(f"Successfully imported {len(chunks)} chunks")

if __name__ == "__main__":
    bulk_import_from_directory()
```

### Step 2: Run Bulk Import
```bash
python bulk_import.py
```

## Method 5: Using the ingest.py Script

### Step 1: Add Your Files
Simply place your `.txt` files in the `data/` directory.

### Step 2: Run the Ingest Script
```bash
python ingest.py
```

This will automatically load all `.txt` files from the `data/` directory and ingest them into the vector database.

## Best Practices

### 1. File Naming
- Use descriptive names: `ai_ml_guide.txt`, `company_policies.txt`
- Avoid spaces: use underscores or hyphens
- Use lowercase for consistency

### 2. Content Structure
- Use clear headings and subheadings
- Organize content with bullet points and numbered lists
- Include relevant keywords naturally in the text
- Keep paragraphs reasonably sized (not too long or too short)

### 3. Metadata
- Include relevant metadata when possible
- Use consistent categories and tags
- Add source information and dates

### 4. Content Quality
- Ensure content is accurate and up-to-date
- Use clear, well-written text
- Avoid excessive formatting or special characters
- Consider the target audience and use cases

## Example: Adding a Technical Document

Here's an example of adding a technical document:

```bash
# Create the file
echo "# Python Programming Guide

Python is a high-level, interpreted programming language known for its simplicity and readability.

## Key Features:
- Easy to learn and use
- Extensive standard library
- Cross-platform compatibility
- Large community and ecosystem

## Common Use Cases:
- Web development
- Data science and machine learning
- Automation and scripting
- Scientific computing

This guide covers the fundamentals of Python programming..." > data/python_guide.txt
```

## Verification

After adding documents, you can verify they were ingested:

```bash
# Check all documents
curl "http://localhost:8000/documents"

# Test a query
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "Python programming", "n_results": 3}'
```

## Troubleshooting

### Common Issues:
1. **File not found**: Ensure the file is in the `data/` directory
2. **Encoding issues**: Use UTF-8 encoding for text files
3. **Empty results**: Check if the ingest script ran successfully
4. **API errors**: Ensure the FastAPI server is running

### Solutions:
- Check file permissions
- Verify file encoding
- Restart the ingest process
- Check server logs for errors 