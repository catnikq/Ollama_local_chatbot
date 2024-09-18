import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

import ollama

from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.environ.get("DATA_PATH")
CHROMA_PATH = os.environ.get("CHROMA_PATH")

class FileLoader:
    def __init__(self) -> None:
        """
        Initialize the FileLoader with ChromaDB settings and embedding model.
        """
        # Initialize Chromadb
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_PATH
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        # Default collection name (can be changed later)
        self.collection = None

    def add_documents(self, pdf_files, collection):
        """Add new pdf files as documents. Split documents into chunks and store in collection.

        Args:
            pdf_files: A list of PDF file paths.
        """
        # Set an existing collection
        self.collection = collection
        if self.collection is None:
            raise ValueError("Collection is not set. Use 'create_or_get_collection' to set a collection.")
        print("Collection information:", self.collection)
        
        # Load documents
        documents = []
        ids = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)
            loaded_documents = loader.load()  # These are the document objects
            documents.extend(loaded_documents)  # Append the loaded document objects
        
        # Split documents into chunk
        all_splits = self.text_splitter.split_documents(documents)
        ids = [str(i) for i, _ in enumerate(all_splits)] # Generate ids for each chunk
        chunks = [split.page_content for split in all_splits] # All_splits into chunks

        # Ensure IDs and chunks have the same length
        assert len(ids) == len(chunks), "Mismatch between number of IDs and chunks"
        
        # Generate embeddings and prepare the data for upserting
        embeddings = []
        ids = [str(i) for i in range(len(chunks))]  # Generate unique IDs for each chunk

        for chunk in chunks:
            response = ollama.embeddings(model="mxbai-embed-large", prompt=chunk)
            embedding = response["embedding"]
            embeddings.append(embedding)  # Store the embedding for this chunk
        
        # Add new documents into collection. Use upsert to avoid adding duplicates
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks
        )
        
        print(f"Added {len(all_splits)} chunks to the collection.")

    def retrieve(self, query, collection, top_k=5):
        # Set an existing collection
        self.collection = collection
        if self.collection is None:
            raise ValueError("Collection is not set. Use 'create_or_get_collection' to set a collection.")
    
        response = ollama.embeddings(
            prompt=query,
            model="mxbai-embed-large"
        )
        
        result = self.collection.query(
            query_embeddings=[response["embedding"]],
            n_results=1
        )
        
        data = result['documents'][0][0]
        return data
        