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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
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
# Agent Configuration (from tool-calling-agents)
# =============================================================================

AGENT_VERBOSE = os.getenv("AGENT_VERBOSE", "True").lower() == "true"
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))
AGENT_EARLY_STOPPING_METHOD = "generate"

# Memory Configuration
MEMORY_TYPE = os.getenv("MEMORY_TYPE", "buffer")
MEMORY_MAX_TOKEN_LIMIT = int(os.getenv("MEMORY_MAX_TOKEN_LIMIT", "2000"))
MEMORY_RETURN_MESSAGES = True
MEMORY_K = int(os.getenv("MEMORY_K", "5"))

VECTOR_MEMORY_COLLECTION = "conversation_memory"
VECTOR_MEMORY_DIR = BASE_DIR / "memoryDB"
VECTOR_MEMORY_DIR.mkdir(exist_ok=True)

# Tools Configuration
TOOLS_ENABLED = {
    "calculator": True,
    "web_search": True,
    "python_repl": True,
    "datetime": True,
    "rag_search": True,
}
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))

# =============================================================================
# Vision Configuration (Class 11 - Multimodal)
# =============================================================================

VISION_ENABLED = os.getenv("VISION_ENABLED", "true").lower() == "true"
VISION_MODEL = os.getenv("VISION_MODEL", "moondream")
VISION_BASE_URL = os.getenv("VISION_BASE_URL", OLLAMA_BASE_URL)
VISION_MAX_IMAGE_SIZE_MB = int(os.getenv("VISION_MAX_IMAGE_SIZE_MB", "10"))
VISION_SUPPORTED_FORMATS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}
VISION_MAX_TOKENS = int(os.getenv("VISION_MAX_TOKENS", "1024"))

# =============================================================================
# Voice Configuration (Class 11 - Multimodal)
# =============================================================================

VOICE_ENABLED = os.getenv("VOICE_ENABLED", "true").lower() == "true"
STT_ENGINE = os.getenv("STT_ENGINE", "whisper")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
TTS_ENGINE = os.getenv("TTS_ENGINE", "gtts")  # gtts | pyttsx3 | orpheus
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "en")
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
AUDIO_MAX_DURATION_SECONDS = int(os.getenv("AUDIO_MAX_DURATION_SECONDS", "120"))
TEMP_MEDIA_DIR = BASE_DIR / "temp_media"
TEMP_MEDIA_DIR.mkdir(exist_ok=True)

# Orpheus TTS Configuration (Ollama-based, natural speech)
ORPHEUS_MODEL = os.getenv("ORPHEUS_MODEL", "legraphista/Orpheus")
ORPHEUS_BASE_URL = os.getenv("ORPHEUS_BASE_URL", OLLAMA_BASE_URL)
ORPHEUS_VOICE = os.getenv("ORPHEUS_VOICE", "tara")  # tara|leah|jess|leo|dan|mia|zac|zoe
ORPHEUS_SAMPLE_RATE = 24000  # SNAC decoder outputs 24kHz audio

# =============================================================================
# Multimodal Agent Configuration (Class 11)
# =============================================================================

MULTIMODAL_MODE = os.getenv("MULTIMODAL_MODE", "full")  # full | text | vision | voice
MULTIMODAL_AUTO_TTS = os.getenv("MULTIMODAL_AUTO_TTS", "false").lower() == "true"
MULTIMODAL_AGENT_VERBOSE = os.getenv("MULTIMODAL_AGENT_VERBOSE", "true").lower() == "true"
MULTIMODAL_TOOLS_ENABLED = {
    "image_analysis": True,
    "image_question": True,
    "voice_transcription": True,
    "text_to_speech": True,
    "image_to_rag": True,
}

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
