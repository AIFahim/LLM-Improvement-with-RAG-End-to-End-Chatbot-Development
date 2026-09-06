"""
Configuration settings for the LLM RAG Chatbot
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Base directory configuration
BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "pdfFiles"
VECTOR_DB_DIR = BASE_DIR / "vectorDB"

# Create directories if they don't exist
PDF_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)

# LLM Provider Selection: "ollama" or "azure"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.7"))

# Ollama Configuration
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT = os.environ.get(
    "AZURE_LLM_DEPLOYMENT_NAME",
    os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4"),
)
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# Embedding model (chat models can't produce embeddings on newer Ollama)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")

# Backwards-compat aliases
LLM_MODEL = OLLAMA_MODEL
LLM_BASE_URL = OLLAMA_BASE_URL

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