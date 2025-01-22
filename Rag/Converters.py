# This script helps convert PDFs, Markdown files, and plain text files into a unified format for chunking as well as generates the embedings
from PyPDF2 import PdfReader
import re
from sentence_transformers import SentenceTransformer
import os


class Converter:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')


    def convert_to_chunks(self, file_path):
        """
        Convert the file into a list of chunks with embeddings and document name,
        depending on the file type (PDF, Markdown, or Text).
        """
        file_extension = os.path.splitext(file_path)[1][1:]

        if file_extension == "pdf":
            return self.convert_pdf(file_path)
        elif file_extension == "txt":
            return self.convert_text_file(file_path)
        elif file_extension in {"md", "markdown"}:
            return self.convert_markdown_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension} for file {file_path}")

        
        
    def convert_pdf(self, doc):
        """
        Convert a PDF document into a list of chunks with embeddings and document name.
        """
        reader = PdfReader(doc)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        return self._process_text(text, doc)

    def convert_text_file(self, doc):
        """
        Convert a plain text file into a list of chunks with embeddings and document name.
        """
        with open(doc, 'r', encoding='utf-8') as file:
            text = file.read()

        return self._process_text(text, doc)

    def convert_markdown_file(self, doc):
        """
        Convert a Markdown file into a list of chunks with embeddings and document name.
        """
        with open(doc, 'r', encoding='utf-8') as file:
            text = file.read()

        # Clean Markdown-specific formatting
        text = re.sub(r'#+\s*', '', text)  # Remove headers
        text = re.sub(r'[*_~`]+', '', text)  # Remove emphasis and inline code markers
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove links
        return self._process_text(text, doc)

    def _process_text(self, text, doc_name):
        """
        Clean, chunk, and embed text, returning a list of chunk data with embeddings.
        """
        cleaned_text = self.clean_text(text)
        chunks = self.chunk_text(cleaned_text, chunk_size=200)
        embeddings = self.model.encode(chunks)

        chunk_data = [
            {
                "chunk_text": chunk, 
                "embedding": embedding, 
                "doc_name": f"{doc_name}_chunk_{i}"
            }
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        return chunk_data

    def clean_text(self, text):
        """
        Remove unnecessary whitespace and newlines.
        """
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
        return text.strip()

    def chunk_text(self, text, chunk_size=500):
        """
        Split text into chunks of specified size.
        """
        words = text.split()
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks
    
