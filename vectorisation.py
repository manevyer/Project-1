"""
vectorisation.py

An end-to-end script for data loading, text cleaning, deduplication,
vectorization, and database persistence using ChromaDB and sentence-transformers.

Key design decisions:
- Only 'content' (page text) is embedded; 'title' and 'url' are stored as metadata.
- Supports both new format ('content'/'title'/'url') and legacy format ('chatbot_response'/'topic').
- Chunking happens ONLY here (single-point chunking), not in webscrap.py,
  to avoid the double-chunking problem.
- chunk_size=800 keeps chunks within the 512-token limit of multilingual-e5-base.
"""

import os
import re
import json
import glob
import logging
from typing import List, Dict, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions


class DataProcessor:
    """Class to handle data loading, extraction, cleaning, and preparation for vectorization."""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        """
        Initialize the text splitter for handling oversized entries.
        
        Args:
            chunk_size (int): The maximum size of each text chunk.
            chunk_overlap (int): The number of overlapping characters between chunks.
        """
        self.chunk_size = chunk_size
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and normalize text for embedding quality."""
        # Normalize line endings and whitespace
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'\t+', ' ', text)
        
        # Remove form template noise (empty fields, dot/underscore patterns)
        text = re.sub(r'\.{3,}', '', text)
        text = re.sub(r'_{3,}', '', text)
        
        # Collapse excessive whitespace
        text = re.sub(r'[ ]+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        
        return text.strip()

    def load_and_prepare(self, directory: str, specific_files: Optional[List[str]] = None) -> Tuple[List[str], List[Dict]]:
        """
        Load JSON files and prepare documents with metadata for vectorization.
        
        Only the 'content' field (or 'chatbot_response' for legacy data) is used as the
        document text for embedding. The 'title' and 'url' are stored as searchable metadata.
        
        Args:
            directory (str): The folder containing JSON files.
            specific_files (Optional[List[str]]): List of specific filenames to process.
            
        Returns:
            Tuple of (documents, metadatas) ready for ChromaDB.
        """
        documents = []
        metadatas = []
        
        search_pattern = os.path.join(directory, "*.json")
        file_paths = glob.glob(search_pattern)
        
        if specific_files:
            file_paths = [fp for fp in file_paths if os.path.basename(fp) in specific_files]
            
        for file_path in file_paths:
            logging.info(f"Loading data from: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not isinstance(data, list):
                    data = [data]
                    
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    
                    # Extract the primary content for embedding
                    # Supports both new format ("content") and old format ("chatbot_response")
                    text = item.get("content", "") or item.get("chatbot_response", "")
                    if isinstance(text, list):
                        text = " ".join(str(v) for v in text)
                    text = self._clean_text(str(text))
                    
                    # Skip very short or empty entries — these are typically form titles
                    if len(text) < 50:
                        logging.debug(f"Skipping short entry ({len(text)} chars): {text[:50]}")
                        continue
                        
                    # Extract metadata fields (supports both new and old format)
                    topic = str(item.get("title", "") or item.get("topic", "")).strip()
                    url = str(item.get("url", "")).strip()
                    
                    # Prepare metadata dict
                    meta = {
                        "topic": topic,
                        "source_url": url,
                        "source_file": os.path.basename(file_path),
                    }
                    
                    # If text fits within chunk_size, use as-is; otherwise chunk it
                    if len(text) <= self.chunk_size:
                        documents.append(text)
                        metadatas.append(meta)
                    else:
                        # Chunk long texts and distribute metadata
                        chunks = self.splitter.split_text(text)
                        for j, chunk in enumerate(chunks):
                            if chunk.strip() and len(chunk.strip()) >= 50:
                                documents.append(chunk.strip())
                                chunk_meta = meta.copy()
                                chunk_meta["chunk_part"] = j + 1
                                metadatas.append(chunk_meta)
                                
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON in file {file_path}: {e}")
            except Exception as e:
                logging.error(f"Unexpected error reading {file_path}: {e}")
        
        logging.info(f"Prepared {len(documents)} documents from {len(file_paths)} files.")
        return documents, metadatas

    @staticmethod
    def deduplicate(documents: List[str], metadatas: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """Remove duplicate documents based on content similarity (first 200 chars)."""
        seen = set()
        deduped_docs = []
        deduped_meta = []
        
        for doc, meta in zip(documents, metadatas):
            key = doc.strip()[:200]
            if key not in seen:
                seen.add(key)
                deduped_docs.append(doc)
                deduped_meta.append(meta)
        
        removed = len(documents) - len(deduped_docs)
        if removed > 0:
            logging.info(f"Deduplication removed {removed} duplicate entries.")
        
        return deduped_docs, deduped_meta


class VectorDatabaseManager:
    """Class to manage vector embeddings and vector DB operations using ChromaDB."""
    
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "chatbot_data", model_name: str = "intfloat/multilingual-e5-base"):
        """
        Initializes the Vector DB Client and embedding function.
        
        Args:
            persist_directory (str): Path to store the ChromaDB files locally.
            collection_name (str): Name of the collection.
            model_name (str): HuggingFace sentence-transformers model name.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        logging.info(f"Initializing embedding model: {model_name}")
        # Initialize the embedding function using sentence-transformers
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        
        # Initialize the persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)

    def create_and_store_embeddings(self, documents: List[str], metadatas: Optional[List[Dict]] = None):
        """
        Converts text documents into vectors and saves them to the vector database.
        
        Args:
            documents (List[str]): List of text documents to be embedded.
            metadatas (Optional[List[Dict]]): List of metadata dicts corresponding to each document.
        """
        if not documents:
            logging.warning("No documents provided to vectorize.")
            return

        try:
            self.client.delete_collection(name=self.collection_name)
            logging.info(f"Deleted old ChromaDB collection '{self.collection_name}' to start fresh.")
        except Exception:
            pass
            
        # Create collection fresh
        collection = self.client.create_collection(
            name=self.collection_name, 
            embedding_function=self.embedding_fn
        )
        
        # Prepare batch payloads
        ids = [f"doc_{i}" for i in range(len(documents))]
        if metadatas is None:
            metadatas = [{"doc_id": i} for i in range(len(documents))]
        
        logging.info(f"Adding {len(documents)} documents to Vector Database (this may take a while)...")
        # ChromaDB handles batching internally, but we add in chunks for memory safety
        batch_size = 5000
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            collection.add(
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
            logging.info(f"Stored documents {i} to {end_idx - 1}")
            
        logging.info(f"Successfully stored all {len(documents)} documents in ChromaDB at {self.persist_directory}")

    def load_existing_db(self):
        """
        Helper function to load the existing database for querying.
        
        Returns:
            Collection: The loaded ChromaDB collection ready for querying.
        """
        logging.info(f"Loading existing Vector Database from {self.persist_directory}")
        collection = self.client.get_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )
        return collection


def main():
    """Main pipeline execution."""
    
    # Define directories
    DATA_DIR = "./"  # Directory containing JSON files
    DB_DIR = "./vector_db" # Directory to save the vector DB
    
    # Define specific files to target
    target_files = ["metu_ie_chatbot_dataset.json", "custom_faqs.json"]
    
    # 1. Initialize Processor with safe chunk size for multilingual-e5-small (max 512 tokens)
    processor = DataProcessor(chunk_size=800, chunk_overlap=100)
    
    # 2. Load and prepare data with metadata (single-point chunking, no double chunking)
    documents, metadatas = processor.load_and_prepare(directory=DATA_DIR, specific_files=target_files)
    
    if not documents:
        logging.error("No documents extracted. Exiting pipeline.")
        return
    
    # 3. Deduplicate
    documents, metadatas = processor.deduplicate(documents, metadatas)
    
    # 4. Vectorize and save to Database
    # Using 'intfloat/multilingual-e5-small' for Turkish/English compatibility
    db_manager = VectorDatabaseManager(
        persist_directory=DB_DIR, 
        collection_name="metu_chatbot",
        model_name="intfloat/multilingual-e5-base"
    )
    
    db_manager.create_and_store_embeddings(documents, metadatas)
    
    # 5. Demonstration: Load the database and make a test query
    logging.info("Testing the loaded database...")
    loaded_collection = db_manager.load_existing_db()
    results = loaded_collection.query(
        query_texts=["IE 400 stajı için ön koşullar nelerdir?"],
        n_results=3
    )
    
    logging.info("Sample query results:")
    if results and results["documents"]:
        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            logging.info(f"  Result {i+1} (topic: {meta.get('topic', 'N/A')}): {doc[:100]}...")


if __name__ == "__main__":
    main()
