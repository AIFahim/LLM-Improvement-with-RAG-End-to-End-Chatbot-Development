"""
Utility functions for the RAG chatbot
"""
import time
import logging
from typing import Generator, List, Optional
import streamlit as st

logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    """
    Setup logging configuration
    
    Args:
        level: Logging level
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def display_message_with_typing(message: str, delay: float = 0.01) -> None:
    """
    Display message with typing animation effect
    
    Args:
        message: Message to display
        delay: Delay between characters
    """
    placeholder = st.empty()
    displayed_text = ""
    
    for char in message:
        displayed_text += char
        placeholder.markdown(displayed_text)
        time.sleep(delay)


def format_sources(source_documents: List) -> str:
    """
    Format source documents for display
    
    Args:
        source_documents: List of source documents
        
    Returns:
        Formatted string of sources
    """
    if not source_documents:
        return ""
    
    sources = []
    for i, doc in enumerate(source_documents, 1):
        page = doc.metadata.get('page', 'Unknown')
        source = doc.metadata.get('source', 'Unknown')
        sources.append(f"{i}. Page {page} from {source}")
    
    return "\n".join(sources)


def validate_pdf_file(file) -> bool:
    """
    Validate if uploaded file is a PDF
    
    Args:
        file: Uploaded file object
        
    Returns:
        True if valid PDF, False otherwise
    """
    if file is None:
        return False
    
    return file.name.lower().endswith('.pdf')


def get_file_size_mb(file) -> float:
    """
    Get file size in MB
    
    Args:
        file: File object
        
    Returns:
        File size in MB
    """
    return file.size / (1024 * 1024)


def create_download_link(text: str, filename: str) -> str:
    """
    Create a download link for text content
    
    Args:
        text: Text content to download
        filename: Name for the downloaded file
        
    Returns:
        HTML download link
    """
    import base64
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:text/plain;base64,{b64}" download="{filename}">Download {filename}</a>'


def format_chat_history(messages: List) -> str:
    """
    Format chat history for export
    
    Args:
        messages: List of chat messages
        
    Returns:
        Formatted chat history string
    """
    formatted = []
    for i, msg in enumerate(messages):
        role = msg.get('role', 'Unknown')
        content = msg.get('content', '')
        formatted.append(f"{role.upper()}: {content}\n")
    
    return "\n".join(formatted)


def estimate_token_count(text: str) -> int:
    """
    Estimate token count for text (rough approximation)
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    # Rough estimation: 1 token ≈ 4 characters
    return len(text) // 4


def truncate_text(text: str, max_length: int = 500) -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Input text
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    import re
    # Remove special characters and spaces
    sanitized = re.sub(r'[^\w\s.-]', '', filename)
    sanitized = re.sub(r'\s+', '_', sanitized)
    return sanitized


def create_session_id() -> str:
    """
    Create unique session ID
    
    Returns:
        Unique session ID
    """
    import uuid
    return str(uuid.uuid4())


def format_timestamp(timestamp: float) -> str:
    """
    Format timestamp to human-readable format
    
    Args:
        timestamp: Unix timestamp
        
    Returns:
        Formatted timestamp string
    """
    from datetime import datetime
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")