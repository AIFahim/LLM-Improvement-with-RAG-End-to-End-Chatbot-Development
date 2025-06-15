"""
Main chatbot module that orchestrates all components
"""
from typing import Optional, Dict, Any, List
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from llm_handler import LLMHandler
import config
import logging

logger = logging.getLogger(__name__)


class RAGChatbot:
    """Main chatbot class that integrates all components"""
    
    def __init__(self):
        """Initialize the RAG chatbot with all components"""
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.llm_handler = LLMHandler()
        self._is_initialized = False
        
        logger.info("Initialized RAG Chatbot")
    
    def process_pdfs(self, uploaded_files: List) -> bool:
        """
        Process uploaded PDF files and create vector store
        
        Args:
            uploaded_files: List of uploaded PDF files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Process PDFs into chunks
            if isinstance(uploaded_files, list):
                chunks = self.document_processor.process_multiple_pdfs(uploaded_files)
            else:
                chunks = self.document_processor.process_pdf(uploaded_files)
            
            # Create vector store
            self.vector_store_manager.create_vector_store(chunks)
            
            # Initialize QA chain
            retriever = self.vector_store_manager.get_retriever()
            self.llm_handler.create_qa_chain(retriever)
            
            self._is_initialized = True
            logger.info("Successfully processed PDFs and initialized chatbot")
            return True
            
        except Exception as e:
            logger.error(f"Error processing PDFs: {e}")
            return False
    
    def add_pdfs(self, uploaded_files: List) -> bool:
        """
        Add additional PDFs to existing vector store
        
        Args:
            uploaded_files: List of uploaded PDF files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Process new PDFs
            if isinstance(uploaded_files, list):
                chunks = self.document_processor.process_multiple_pdfs(uploaded_files)
            else:
                chunks = self.document_processor.process_pdf(uploaded_files)
            
            # Add to vector store
            self.vector_store_manager.add_documents(chunks)
            
            # Reinitialize QA chain with updated retriever
            retriever = self.vector_store_manager.get_retriever()
            self.llm_handler.create_qa_chain(retriever)
            
            logger.info("Successfully added new PDFs to chatbot")
            return True
            
        except Exception as e:
            logger.error(f"Error adding PDFs: {e}")
            return False
    
    def chat(self, question: str) -> Dict[str, Any]:
        """
        Chat with the bot about the uploaded documents
        
        Args:
            question: User question
            
        Returns:
            Dictionary with response and source documents
        """
        if not self._is_initialized:
            return {
                "result": "Please upload a PDF file first to start chatting.",
                "source_documents": []
            }
        
        try:
            response = self.llm_handler.query(question)
            return response
        except Exception as e:
            logger.error(f"Error during chat: {e}")
            return {
                "result": f"Sorry, I encountered an error: {str(e)}",
                "source_documents": []
            }
    
    def search_documents(self, query: str, k: int = 4) -> List:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of relevant documents
        """
        return self.vector_store_manager.search(query, k)
    
    def clear_chat_history(self):
        """Clear the conversation history"""
        self.llm_handler.clear_memory()
        logger.info("Cleared chat history")
    
    def get_chat_history(self) -> list:
        """
        Get conversation history
        
        Returns:
            List of conversation messages
        """
        return self.llm_handler.get_conversation_history()
    
    def reset(self):
        """Reset the entire chatbot state"""
        self.vector_store_manager.clear_vector_store()
        self.llm_handler.clear_memory()
        self._is_initialized = False
        logger.info("Reset chatbot state")
    
    @property
    def is_ready(self) -> bool:
        """Check if chatbot is ready for queries"""
        return self._is_initialized