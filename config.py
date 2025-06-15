"""
Configuration settings for the LLM RAG Chatbot
"""
import os
from pathlib import Path

# Base directory configuration
BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "pdfFiles"
VECTOR_DB_DIR = BASE_DIR / "vectorDB"

# Create directories if they don't exist
PDF_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)

# LLM Configuration
LLM_MODEL = "llama3.2:1b"
LLM_BASE_URL = "http://localhost:11434"
LLM_TEMPERATURE = 0.7

# Document Processing Configuration
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# Vector Database Configuration
CHROMA_PERSIST_DIR = str(VECTOR_DB_DIR)
COLLECTION_NAME = "pdf_documents"

# Streamlit Configuration
PAGE_TITLE = "RAG Chatbot Assistance"
PAGE_ICON = "🤖"
PAGE_LAYOUT = "wide"

# Session State Keys
SESSION_MESSAGES = "messages"
SESSION_VECTOR_STORE = "vector_store"
SESSION_CONVERSATION_CHAIN = "conversation_chain"

# UI Messages
WELCOME_MESSAGE = "Upload PDFs and get instant answers! 📄🤖"
UPLOAD_PROMPT = "Please upload a PDF file to start the conversation."
PROCESSING_MESSAGE = "Processing your PDF... This may take a moment."
SUCCESS_MESSAGE = "PDF processed successfully! You can now ask questions."
ERROR_MESSAGE = "An error occurred: {}"

# Disable telemetry (optional)
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"