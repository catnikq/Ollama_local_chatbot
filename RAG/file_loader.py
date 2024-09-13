import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
from model.embedding_model import EmbeddingModel


class FileLoader:
    def __init__(self, db_path = 'chroma_db') -> None:
        """
        Initialize the FileLoader with ChromaDB settings and embedding model.
        """
        # Initialize Chromadb
        self.chroma_client = chromadb.PersistentClient(
            Settings(
                persist_directory=db_path,
                embedding_function=EmbeddingModel().generate_embedding
            )
        )
    
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        # Default collection name (can be changed later)
        self.collection = None

        
    def add_documents(self, pdf_files):
        """Add new pdf files as documents. Split documents into chunks and store in collection.

        Args:
            pdf_files: A list of PDF file paths.
        """
        if not self.collection:
            raise ValueError("Collection is not set. Use 'create_or_get_collection' to set a collection.")
        
        # Load documents
        documents = []
        ids = []
        for i, pdf_file in enumerate(pdf_files):
            loader = PyPDFLoader(pdf_file)
            loaded_documents = loader.load()
            documents.extend([doc.page_content for doc in loaded_documents])
            ids.extend([f"pdf_{i}_{j}" for j in range(len(loaded_documents))])  # Generate unique IDs
        
        # Split documents into chunk
        all_splits = self.text_splitter.split_documents(documents)
        
        # Add new documents into collection. Use upsert to avoid adding duplicates
        self.collection.upsert(
            documents=all_splits,
            ids=ids
        )
        
        print(f"Added {len(all_splits)} chunks to the collection.")
        
    def retriever(self):
        """
        Create a retriever for similarity searching using cosine similarity.

        Returns:
            chromadb.utils.Retriever: A retriever instance.
        """
        if not self.collection:
            raise ValueError("Collection is not set. Use 'create_or_get_collection' to set a collection.")
    
        # Create retriever
        return self.collection.as_retriever()
        
