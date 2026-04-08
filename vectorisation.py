"""
vectorisation.py

An end-to-end script for data loading, text extraction, cleaning, chunking,
vectorization, and database persistence using ChromaDB and sentence-transformers.
"""

import os
import json
import glob
import logging
import subprocess
import sys
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def install_requirements():
    """Installs required packages if they are not already installed."""
    try:
        import langchain_text_splitters
        import chromadb
        import sentence_transformers
    except ImportError:
        logging.info("Missing required packages. Installing them now...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            logging.info("Packages installed successfully.")
        except Exception as e:
            logging.error(f"Failed to install packages automatically: {e}")
            logging.info("Please run: pip install -r requirements.txt")

# Call the installation before importing 3rd party libraries
install_requirements()

from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions


class DataProcessor:
    """Class to handle data loading, extraction, and chunking."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the text splitter.
        
        Args:
            chunk_size (int): The maximum size of each text chunk.
            chunk_overlap (int): The number of overlapping characters between chunks.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_text_from_item(self, item: Any) -> str:
        """
        Flexibly extracts text from a JSON object.
        Looks for common keys like 'text', 'question', 'answer', 'chatbot_response', etc.
        """
        if isinstance(item, str):
            return item.strip()
        
        extracted = []
        if isinstance(item, dict):
            # Target keys prioritized for RAG
            target_keys = ['topic', 'question', 'answer', 'text', 'chatbot_response', 'user_query_variations']
            
            found_target = False
            for key in target_keys:
                if key in item:
                    val = item[key]
                    if isinstance(val, list):
                        # Join lists into a single string (useful for query variations)
                        extracted.append(" ".join([str(v) for v in val]))
                    else:
                        extracted.append(str(val))
                    found_target = True
            
            # Fallback if no specific keys are found: extract all string values
            if not found_target:
                for val in item.values():
                    if isinstance(val, str):
                        extracted.append(val)
                    elif isinstance(val, list):
                        extracted.extend([str(v) for v in val if isinstance(v, str)])
                        
        return "\n".join(extracted).strip()

    def load_and_merge_json(self, directory: str, specific_files: Optional[List[str]] = None) -> List[str]:
        """
        Reads JSON files in a directory and extracts text.
        
        Args:
            directory (str): The folder containing JSON files.
            specific_files (Optional[List[str]]): List of specific filenames to process.
            
        Returns:
            List[str]: A list of extracted text strings.
        """
        texts = []
        search_pattern = os.path.join(directory, "*.json")
        file_paths = glob.glob(search_pattern)
        
        if specific_files:
            file_paths = [fp for fp in file_paths if os.path.basename(fp) in specific_files]
            
        for file_path in file_paths:
            logging.info(f"Loading data from: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Handle cases where json contains a list of objects or a single object
                if isinstance(data, list):
                    for item in data:
                        text = self.extract_text_from_item(item)
                        if text:
                            texts.append(text)
                else:
                    text = self.extract_text_from_item(data)
                    if text:
                        texts.append(text)
                        
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON in file {file_path}: {e}")
            except Exception as e:
                logging.error(f"Unexpected error reading {file_path}: {e}")
                
        return texts

    def chunk_texts(self, texts: List[str]) -> List[str]:
        """
        Splits texts into smaller manageable chunks.
        
        Args:
            texts (List[str]): List of extracted string texts.
            
        Returns:
            List[str]: List of chunked texts.
        """
        logging.info("Chunking texts...")
        # create_documents returns Document objects, we just need the text for ChromaDB
        docs = self.splitter.create_documents(texts)
        chunks = [doc.page_content for doc in docs if doc.page_content.strip()]
        logging.info(f"Generated {len(chunks)} text chunks.")
        return chunks


class VectorDatabaseManager:
    """Class to manage vector embeddings and vector DB operations using ChromaDB."""
    
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "chatbot_data", model_name: str = "intfloat/multilingual-e5-small"):
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

    def create_and_store_embeddings(self, chunks: List[str]):
        """
        Converts text chunks into vectors and saves them to the vector database.
        
        Args:
            chunks (List[str]): List of text chunks to be embedded.
        """
        if not chunks:
            logging.warning("No chunks provided to vectorize.")
            return

        # Get or create collection
        collection = self.client.get_or_create_collection(
            name=self.collection_name, 
            embedding_function=self.embedding_fn
        )
        
        # Prepare batch payloads
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"chunk_id": i} for i in range(len(chunks))]
        
        logging.info("Adding chunks to Vector Database (this may take a while depending on chunk size)...")
        # ChromaDB handles batching internally, but we add it in chunks just in case memory issues arise
        batch_size = 5000
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            collection.add(
                documents=chunks[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
            logging.info(f"Stored chunks {i} to {end_idx - 1}")
            
        logging.info(f"Successfully stored all chunks in ChromaDB at {self.persist_directory}")

    def load_existing_db(self):
        """
        Helper function to demonstrate how to load the database later.
        
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
    
    # Define specific files to target if necessary
    target_files = ["metu_ie_chatbot_dataset.json", "custom_faqs.json"]
    
    # 1. Initialize Processor
    processor = DataProcessor(chunk_size=500, chunk_overlap=50)
    
    # 2. Load and merge data flexibly
    # You can set specific_files=None to process all JSONs in DATA_DIR
    extracted_texts = processor.load_and_merge_json(directory=DATA_DIR, specific_files=target_files)
    
    if not extracted_texts:
        logging.error("No texts extracted. Exiting pipeline.")
        return
        
    # 3. Clean and Chunk texts
    chunks = processor.chunk_texts(extracted_texts)
    
    # 4. Vectorize and save to Database
    # Using 'intfloat/multilingual-e5-small' for Turkish/English compatibility
    db_manager = VectorDatabaseManager(
        persist_directory=DB_DIR, 
        collection_name="metu_chatbot",
        model_name="intfloat/multilingual-e5-small"
    )
    
    db_manager.create_and_store_embeddings(chunks)
    
    # 5. Demonstration: Load the database and make a test query
    logging.info("Testing the loaded database...")
    loaded_collection = db_manager.load_existing_db()
    results = loaded_collection.query(
        query_texts=["What are the IE 300 internship requirements?"],
        n_results=1
    )
    
    logging.info("Sample query result:")
    logging.info(results["documents"])


if __name__ == "__main__":
    main()
