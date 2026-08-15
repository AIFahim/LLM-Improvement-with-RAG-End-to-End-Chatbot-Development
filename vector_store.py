"""
Vector store module for managing ChromaDB operations (Fixed version)
"""
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import chromadb
from chromadb.config import Settings
import config
import logging
import shutil
import os

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages vector database operations with ChromaDB"""
    
    def __init__(self, model_name: str = config.OLLAMA_EMBED_MODEL,
                 base_url: str = config.OLLAMA_BASE_URL,
                 batch_size: int = config.EMBED_BATCH_SIZE):
        """
        Initialize the vector store manager

        Args:
            model_name: Name of the embedding model
            base_url: Base URL for Ollama
            batch_size: Number of chunks embedded per request
        """
        self.model_name = model_name
        self.base_url = base_url
        self.batch_size = batch_size
        self.embeddings = OllamaEmbeddings(
            model=model_name,
            base_url=base_url
        )
        self.persist_directory = config.CHROMA_PERSIST_DIR
        self.collection_name = config.COLLECTION_NAME
        self._vector_store = None
        
        # Initialize ChromaDB client with settings
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
    
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """
        Create a new vector store from documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            Chroma vector store instance
        """
        staging_name = f"{self.collection_name}_staging"
        try:
            # Drop leftovers from a previous run that died mid-build
            self._drop_collection(staging_name)

            # Build into staging first, so a failure part way through leaves the
            # existing collection untouched
            staging_store = Chroma(
                client=self.chroma_client,
                collection_name=staging_name,
                embedding_function=self.embeddings
            )
            self._add_in_batches(staging_store, documents)

            # Only now is it safe to replace the previous collection
            self._drop_collection(self.collection_name)
            self.chroma_client.get_collection(name=staging_name).modify(
                name=self.collection_name
            )

            self._vector_store = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
            logger.info(f"Created vector store with {len(documents)} documents")
            return self._vector_store

        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            self._drop_collection(staging_name)
            raise

    def _drop_collection(self, name: str):
        """Delete a collection, ignoring the case where it does not exist"""
        try:
            self.chroma_client.delete_collection(name=name)
            logger.info(f"Deleted existing collection: {name}")
        except Exception:
            pass

    def _add_in_batches(self, store: Chroma, documents: List[Document]):
        """
        Embed documents in batches

        A single call covering every chunk keeps the whole PDF in flight at once,
        which is what exhausts memory on large files and loses all progress when
        one request fails.
        """
        total = len(documents)
        for start in range(0, total, self.batch_size):
            batch = documents[start:start + self.batch_size]
            store.add_documents(batch)
            logger.info(f"Embedded {min(start + len(batch), total)}/{total} chunks")

    def load_vector_store(self) -> Optional[Chroma]:
        """
        Load existing vector store from disk
        
        Returns:
            Chroma vector store instance or None if not exists
        """
        try:
            # Check if collection exists
            collections = self.chroma_client.list_collections()
            if not any(col.name == self.collection_name for col in collections):
                logger.warning(f"Collection {self.collection_name} not found")
                return None
            
            self._vector_store = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
            logger.info("Loaded existing vector store")
            return self._vector_store
        except Exception as e:
            logger.warning(f"Could not load vector store: {e}")
            return None
    
    def add_documents(self, documents: List[Document]):
        """
        Add new documents to existing vector store
        
        Args:
            documents: List of Document objects
        """
        if self._vector_store is None:
            # Try the persisted collection before falling back to a rebuild,
            # otherwise "adding" would wipe an index that is already on disk
            self.load_vector_store()

        if self._vector_store is None:
            self.create_vector_store(documents)
        else:
            self._add_in_batches(self._vector_store, documents)
            logger.info(f"Added {len(documents)} documents to vector store")
    
    def get_vector_store(self) -> Optional[Chroma]:
        """
        Get the current vector store instance
        
        Returns:
            Chroma vector store instance
        """
        if self._vector_store is None:
            self._vector_store = self.load_vector_store()
        return self._vector_store
    
    def clear_vector_store(self):
        """Clear the existing vector store"""
        try:
            # Try to delete collection via client
            try:
                self.chroma_client.delete_collection(name=self.collection_name)
                logger.info(f"Deleted collection: {self.collection_name}")
            except Exception:
                pass
            
            # Also clear the directory if it exists
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                os.makedirs(self.persist_directory)
                logger.info("Cleared vector store directory")
                
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
    
    def search(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant Document objects
        """
        if self._vector_store is None:
            logger.warning("No vector store available")
            return []
        
        results = self._vector_store.similarity_search(query, k=k)
        logger.info(f"Found {len(results)} relevant documents for query")
        return results
    
    def get_retriever(self, k: int = 4):
        """
        Get a retriever instance for the vector store
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            Retriever instance
        """
        if self._vector_store is None:
            raise ValueError("No vector store available")
        
        return self._vector_store.as_retriever(
            search_kwargs={"k": k}
        )