import chromadb
from chromadb.config import Settings
import os

class VectorDatabase:
    def __init__(self, collection_name, dim=384, db_path="./database"):
        """
        Initialize a connection to the Chroma vector database.
        
        Args:
            collection_name (str): Name of the collection.
            dim (int): Dimension of the vector embeddings.
            db_path (str): Path to persist the Chroma database.
        """
        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.dim = dim

    def _ensure_collection(self):
        """
        Ensure the collection exists, and create it if not.
        """
        return self.client.get_or_create_collection(name=self.collection_name)

    def insert(self, chunks):
        """
        Insert chunk data into the collection.
        
        Args:
            chunks (list): List of dictionaries with keys: 'chunk_text', 'embedding', 'doc_name'.
        """
        if not chunks:
            print("No chunks to insert.")
            return

        chunk_texts = [chunk["chunk_text"] for chunk in chunks]
        doc_names = [chunk["doc_name"] for chunk in chunks]
        embeddings = [chunk["embedding"] for chunk in chunks]

        self.collection.upsert(
            documents=chunk_texts,
            ids=doc_names,
            metadatas=[{"chunk_text": chunk["chunk_text"]} for chunk in chunks],
            embeddings=embeddings
        )
        print(f"Inserted {len(chunks)} chunks into the collection.")

    

    def similarity_search(self, query_vector, top_k=5):
        """
        Perform a similarity search.

        Args:
            query_vector (list): The query vector embedding.
            top_k (int): Number of top similar results to retrieve.

        Returns:
            list: Top-k search results.
        """
        print("hip")
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        print(f"Query Results: {results}")

        # Handle edge cases where results may not have expected keys
        documents = results.get("documents", []) if isinstance(results, dict) else []
        distances = results.get("distances", []) if isinstance(results, dict) else []

        if not documents or not distances:
            return [{"doc_name": None, "chunk_text": None, "score": None}]

        return [
            {"doc_name": doc_id, "chunk_text": metadata.get("chunk_text", ""), "score": score}
            for doc_id, metadata, score in zip(results["ids"][0], results["metadatas"][0], results["distances"][0])
        ]

    def delete_chunks_by_doc_name(self, doc_name):
        """
        Delete all chunks associated with a specific doc_name.
        
        Args:
            doc_name (str): The document name to delete.
        
        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        # Query to check if the doc_name exists first
        chunks_to_delete = self.get_chunks_by_doc_name(doc_name)
        
        if not chunks_to_delete:
            print(f"No chunks found with doc_name: {doc_name}.")
            return False
        
        # Perform the deletion by upserting with an empty document list for the doc_name
        self.collection.delete(ids=[doc_name])
        print(f"Deleted chunks associated with doc_name: {doc_name}.")
        return True

    def doc_name_exists(self, doc_name):
        """
        Check if a specific doc_name exists in the database.
        
        Args:
            doc_name (str): The document name to check for existence.
        
        Returns:
            bool: True if doc_name exists, False otherwise.
        """
        chunks = self.get_chunks_by_doc_name(doc_name)
        if chunks:
            print(f"Chunks found for doc_name: {doc_name}.")
            return True
        else:
            print(f"No chunks found for doc_name: {doc_name}.")
            return False
