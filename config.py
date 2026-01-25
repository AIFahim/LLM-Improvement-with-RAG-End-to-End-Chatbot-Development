"""
Configuration settings for the LLM RAG Chatbot
Enhanced with secure API key handling and multiple LLM providers
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Base directory configuration
BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "pdfFiles"
VECTOR_DB_DIR = BASE_DIR / "vectorDB"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
PDF_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# =============================================================================
# LLM Provider Configuration
# =============================================================================

# LLM Provider: "ollama" or "azure"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# Ollama Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# Backward compatibility aliases
LLM_MODEL = OLLAMA_MODEL
LLM_BASE_URL = OLLAMA_BASE_URL

# =============================================================================
# Document Processing Configuration
# =============================================================================

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# =============================================================================
# Vector Database Configuration
# =============================================================================

CHROMA_PERSIST_DIR = str(VECTOR_DB_DIR)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_documents")

# =============================================================================
# Langfuse Observability Configuration
# =============================================================================

LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# =============================================================================
# Evaluation Configuration
# =============================================================================

EVALUATION_ENABLED = os.getenv("EVALUATION_ENABLED", "false").lower() == "true"
EVALUATION_MODEL = os.getenv("EVALUATION_MODEL", OLLAMA_MODEL)

# =============================================================================
# Performance Configuration
# =============================================================================

# Response caching
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "100"))

# Retry configuration
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY_SECONDS = float(os.getenv("RETRY_DELAY_SECONDS", "1.0"))

# =============================================================================
# API Configuration
# =============================================================================

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"

# =============================================================================
# Streamlit Configuration
# =============================================================================

PAGE_TITLE = "RAG Chatbot Assistance"
PAGE_ICON = "🤖"
PAGE_LAYOUT = "wide"

# Session State Keys
SESSION_MESSAGES = "messages"
SESSION_VECTOR_STORE = "vector_store"
SESSION_CONVERSATION_CHAIN = "conversation_chain"
SESSION_EVALUATION_ENABLED = "evaluation_enabled"
SESSION_DEBUG_MODE = "debug_mode"

# UI Messages
WELCOME_MESSAGE = "Upload PDFs and get instant answers!"
UPLOAD_PROMPT = "Please upload a PDF file to start the conversation."
PROCESSING_MESSAGE = "Processing your PDF... This may take a moment."
SUCCESS_MESSAGE = "PDF processed successfully! You can now ask questions."
ERROR_MESSAGE = "An error occurred: {}"

# =============================================================================
# Secure API Key Handling
# =============================================================================


class SecureConfig:
    """Secure configuration handler for API keys and sensitive data"""

    @staticmethod
    def load_api_key(key_name: str) -> Optional[str]:
        """
        Load an API key from environment variables

        Args:
            key_name: Name of the environment variable

        Returns:
            API key value or None if not found
        """
        value = os.getenv(key_name)
        if value:
            logger.debug(f"Loaded API key: {key_name}")
        return value

    @staticmethod
    def mask_key(key: str, visible_chars: int = 4) -> str:
        """
        Mask an API key for safe logging

        Args:
            key: The API key to mask
            visible_chars: Number of characters to show at start and end

        Returns:
            Masked key string
        """
        if not key or len(key) < visible_chars * 2:
            return "***"
        return f"{key[:visible_chars]}...{key[-visible_chars:]}"

    @staticmethod
    def validate_keys() -> dict:
        """
        Validate all configured API keys

        Returns:
            Dictionary with validation results
        """
        results = {
            "ollama": {
                "configured": True,  # Ollama doesn't need API key
                "valid": True,
                "message": "Ollama is configured (no API key needed)"
            },
            "azure_openai": {
                "configured": bool(AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT),
                "valid": False,
                "message": ""
            },
            "langfuse": {
                "configured": bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY),
                "valid": False,
                "message": ""
            }
        }

        # Validate Azure OpenAI
        if results["azure_openai"]["configured"]:
            results["azure_openai"]["valid"] = len(AZURE_OPENAI_API_KEY) > 10
            results["azure_openai"]["message"] = (
                f"Azure OpenAI configured with deployment: {AZURE_OPENAI_DEPLOYMENT}"
            )
        else:
            results["azure_openai"]["message"] = "Azure OpenAI not configured"

        # Validate Langfuse
        if results["langfuse"]["configured"]:
            results["langfuse"]["valid"] = (
                len(LANGFUSE_PUBLIC_KEY) > 5 and len(LANGFUSE_SECRET_KEY) > 5
            )
            results["langfuse"]["message"] = f"Langfuse configured with host: {LANGFUSE_HOST}"
        else:
            results["langfuse"]["message"] = "Langfuse not configured"

        return results

    @staticmethod
    def get_provider_config() -> dict:
        """
        Get current LLM provider configuration

        Returns:
            Dictionary with provider configuration
        """
        if LLM_PROVIDER == "azure":
            return {
                "provider": "azure",
                "model": AZURE_OPENAI_DEPLOYMENT,
                "endpoint": SecureConfig.mask_key(AZURE_OPENAI_ENDPOINT, 20)
                if AZURE_OPENAI_ENDPOINT else "Not configured"
            }
        else:
            return {
                "provider": "ollama",
                "model": OLLAMA_MODEL,
                "endpoint": OLLAMA_BASE_URL
            }


# =============================================================================
# Environment Validation
# =============================================================================

def validate_environment() -> bool:
    """
    Validate the environment configuration

    Returns:
        True if valid, False otherwise
    """
    errors = []

    # Check LLM provider
    if LLM_PROVIDER not in ["ollama", "azure"]:
        errors.append(f"Invalid LLM_PROVIDER: {LLM_PROVIDER}")

    # Check Azure configuration if using Azure
    if LLM_PROVIDER == "azure":
        if not AZURE_OPENAI_API_KEY:
            errors.append("AZURE_OPENAI_API_KEY not set")
        if not AZURE_OPENAI_ENDPOINT:
            errors.append("AZURE_OPENAI_ENDPOINT not set")

    # Check Langfuse if enabled
    if LANGFUSE_ENABLED:
        if not LANGFUSE_PUBLIC_KEY:
            errors.append("LANGFUSE_PUBLIC_KEY not set but LANGFUSE_ENABLED=true")
        if not LANGFUSE_SECRET_KEY:
            errors.append("LANGFUSE_SECRET_KEY not set but LANGFUSE_ENABLED=true")

    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
        return False

    logger.info(f"Environment validated: Provider={LLM_PROVIDER}")
    return True


# Disable telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
