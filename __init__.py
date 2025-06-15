"""
LLM RAG Chatbot - A modular chatbot for PDF document Q&A using Ollama and ChromaDB.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .chatbot import RAGChatbot
from .document_processor import DocumentProcessor
from .vector_store import VectorStoreManager
from .llm_handler import LLMHandler

__all__ = [
    "RAGChatbot",
    "DocumentProcessor",
    "VectorStoreManager",
    "LLMHandler"
]