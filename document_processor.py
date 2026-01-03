"""
Document processing module for handling PDF files
"""
import os
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import config
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles PDF loading and text chunking operations"""
    
    def __init__(self, chunk_size: int = config.CHUNK_SIZE, 
                 chunk_overlap: int = config.CHUNK_OVERLAP):
        """
        Initialize the document processor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def save_uploaded_file(self, uploaded_file) -> str:
        """
        Save uploaded file to disk
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            Path to saved file
        """
        file_path = os.path.join(config.PDF_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info(f"Saved uploaded file to {file_path}")
        return file_path
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load PDF file and extract text
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of Document objects
        """
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from {file_path}")
            return documents
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            raise
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(chunks)} chunks")
        return chunks
    
    def process_pdf(self, uploaded_file) -> List[Document]:
        """
        Complete pipeline to process PDF file
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            List of chunked Document objects
        """
        # Save file
        file_path = self.save_uploaded_file(uploaded_file)
        
        # Load PDF
        documents = self.load_pdf(file_path)
        
        # Split into chunks
        chunks = self.split_documents(documents)
        
        return chunks
    
    def process_multiple_pdfs(self, uploaded_files: List) -> List[Document]:
        """
        Process multiple PDF files
        
        Args:
            uploaded_files: List of Streamlit UploadedFile objects
            
        Returns:
            Combined list of chunked Document objects
        """
        all_chunks = []
        for uploaded_file in uploaded_files:
            chunks = self.process_pdf(uploaded_file)
            all_chunks.extend(chunks)
        
        logger.info(f"Processed {len(uploaded_files)} PDFs into {len(all_chunks)} total chunks")
        return all_chunks