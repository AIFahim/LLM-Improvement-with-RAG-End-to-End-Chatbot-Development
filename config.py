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

# LLM Provider: "ollama" or "azure"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# Ollama Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")  # e.g., https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("AZURE_LLM_DEPLOYMENT_NAME", "gpt-4")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# Common LLM settings
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

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

# =============================================================================
# AGENT CONFIGURATION (Class 06: LangChain Deep Dive)
# =============================================================================

# Agent Settings
AGENT_VERBOSE = os.getenv("AGENT_VERBOSE", "True").lower() == "true"
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))
AGENT_EARLY_STOPPING_METHOD = "generate"  # "force" or "generate"

# Memory Configuration
MEMORY_TYPE = os.getenv("MEMORY_TYPE", "buffer")  # "buffer", "summary", "vector", "combined"
MEMORY_MAX_TOKEN_LIMIT = int(os.getenv("MEMORY_MAX_TOKEN_LIMIT", "2000"))
MEMORY_RETURN_MESSAGES = True
MEMORY_K = int(os.getenv("MEMORY_K", "5"))  # Number of messages to keep in buffer

# Vector Memory Configuration
VECTOR_MEMORY_COLLECTION = "conversation_memory"
VECTOR_MEMORY_DIR = BASE_DIR / "memoryDB"
VECTOR_MEMORY_DIR.mkdir(exist_ok=True)

# Tool Configuration
TOOLS_ENABLED = {
    "calculator": True,
    "web_search": True,
    "python_repl": True,
    "datetime": True,
    "rag_search": True,
}

# Web Search Configuration (DuckDuckGo)
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))

# Python REPL Configuration
PYTHON_REPL_TIMEOUT = int(os.getenv("PYTHON_REPL_TIMEOUT", "30"))

# Agent UI Configuration
AGENT_PAGE_TITLE = "LangChain Agent with Memory & Tools"
AGENT_PAGE_ICON = "🤖"

# Session State Keys for Agent
SESSION_AGENT = "agent"
SESSION_MEMORY = "memory"
SESSION_TOOLS = "tools"
SESSION_AGENT_MESSAGES = "agent_messages"

# Agent UI Messages
AGENT_WELCOME_MESSAGE = "I'm an AI agent with multiple tools and memory. Ask me anything!"
AGENT_THINKING_MESSAGE = "Thinking and using tools..."

# Disable telemetry (optional)
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"