# Retrieval Augmented Generation Example

This repository demonstrates a simple Retrieval Augmented Generation (RAG) pipeline. It enables uploading documents and querying them, supporting text, Markdown, and PDF files, all of which are converted into plain text. This project was developed as a learning tool to understand the various components and how they integrate.

## Setup

### 1. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Create Environment Variables
Create a `.env` file in the `Rag/` directory with the following content:
```env
GROQ_API_KEY=your-groq-api-key
```
Replace `your-groq-api-key` with your actual API key.

### 4. Run the Flask Application
```bash
python controller.py
```

The application will start, and you can interact with it via HTTP requests.

## Endpoints

### 1. Upload Document
- **Endpoint**: `/upload_doc`
- **Method**: POST
- **Description**: Upload a document to the pipeline.

Example using `curl`:
```bash
curl -X POST http://localhost:5000/upload_doc \
     -F "file=@test.txt" \
     -F "doc_name=test.txt"
```

### 2. Query the Pipeline
- **Endpoint**: `/query`
- **Method**: POST
- **Description**: Send a query to the pipeline and receive a response.

Example using `curl`:
```bash
curl -X POST http://localhost:5000/query \
     -H "Content-Type: application/json" \
     -d '{"prompt": "What is the capital of France?"}'
```

## Author

Nicholas Lombard
