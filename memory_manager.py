"""
Memory Manager Module - LangChain Deep Dive (Class 06)

This module implements conversation memory using the new LangChain/LangGraph API:
1. InMemoryChatMessageHistory - Simple in-memory storage
2. RunnableWithMessageHistory - For chain integration
3. MemorySaver - LangGraph checkpoint-based memory

Updated for LangChain 1.x / LangGraph architecture.
"""

import logging
from typing import Optional, List, Dict, Any, Union
from enum import Enum

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

import config

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Enum for different memory types"""
    BUFFER = "buffer"                    # Simple in-memory buffer
    WINDOW = "window"                    # Last k messages only
    SUMMARY = "summary"                  # Summarized history
    VECTOR = "vector"                    # Vector store based
    LANGGRAPH = "langgraph"              # LangGraph MemorySaver


# Global session store for chat histories
_session_histories: Dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create a chat history for a session."""
    if session_id not in _session_histories:
        _session_histories[session_id] = InMemoryChatMessageHistory()
    return _session_histories[session_id]


class MemoryManager:
    """
    Manages conversation memory for LangChain agents.

    Supports multiple memory backends:
    - Buffer: Simple in-memory message storage
    - Window: Keeps only last K messages
    - Vector: Semantic search over history (using ChromaDB)
    - LangGraph: Checkpoint-based memory with MemorySaver
    """

    def __init__(
        self,
        memory_type: Union[MemoryType, str] = MemoryType.BUFFER,
        session_id: str = "default",
        k: int = None,
        llm: Any = None,
    ):
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type.lower())

        self.memory_type = memory_type
        self.session_id = session_id
        self.k = k or config.MEMORY_K
        self.llm = llm or self._get_default_llm()

        # Memory storage
        self._history: InMemoryChatMessageHistory = None
        self._vector_store: Optional[Chroma] = None
        self._langgraph_memory: Optional[MemorySaver] = None

        logger.info(f"MemoryManager initialized: type={memory_type.value}, session={session_id}")

    def _get_default_llm(self) -> Any:
        """Get default LLM based on configuration."""
        if config.LLM_PROVIDER == "azure":
            return AzureChatOpenAI(
                azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                api_key=config.AZURE_OPENAI_API_KEY,
                api_version=config.AZURE_OPENAI_API_VERSION,
                deployment_name=config.AZURE_OPENAI_DEPLOYMENT,
                temperature=config.LLM_TEMPERATURE,
            )
        return OllamaLLM(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.LLM_TEMPERATURE,
        )

    def _get_embeddings(self) -> OllamaEmbeddings:
        """Get embeddings for vector memory."""
        return OllamaEmbeddings(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )

    def get_history(self) -> InMemoryChatMessageHistory:
        """Get or create the chat history."""
        if self._history is None:
            self._history = get_session_history(self.session_id)
        return self._history

    def get_langgraph_memory(self) -> MemorySaver:
        """Get LangGraph MemorySaver for checkpoint-based memory."""
        if self._langgraph_memory is None:
            self._langgraph_memory = MemorySaver()
        return self._langgraph_memory

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to history."""
        history = self.get_history()
        history.add_message(message)

        # Apply window limit if needed
        if self.memory_type == MemoryType.WINDOW:
            self._apply_window_limit()

    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self.add_message(HumanMessage(content=content))

    def add_ai_message(self, content: str) -> None:
        """Add an AI message."""
        self.add_message(AIMessage(content=content))

    def _apply_window_limit(self) -> None:
        """Keep only the last k messages."""
        history = self.get_history()
        messages = history.messages
        if len(messages) > self.k * 2:  # k turns = k*2 messages
            # Keep only last k turns
            history.clear()
            for msg in messages[-(self.k * 2):]:
                history.add_message(msg)

    def get_messages(self) -> List[BaseMessage]:
        """Get all messages from history."""
        return self.get_history().messages

    def get_messages_as_string(self) -> str:
        """Get messages formatted as string."""
        messages = self.get_messages()
        lines = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                lines.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                lines.append(f"AI: {msg.content}")
            else:
                lines.append(f"{msg.type}: {msg.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all memory."""
        if self._history is not None:
            self._history.clear()

        if self.session_id in _session_histories:
            _session_histories[self.session_id].clear()

        if self._vector_store is not None:
            try:
                self._vector_store.delete_collection()
                self._vector_store = None
            except Exception as e:
                logger.warning(f"Could not clear vector store: {e}")

        logger.info(f"Memory cleared for session: {self.session_id}")

    def get_memory_info(self) -> Dict[str, Any]:
        """Get information about current memory state."""
        messages = self.get_messages()
        return {
            "memory_type": self.memory_type.value,
            "session_id": self.session_id,
            "message_count": len(messages),
            "window_size": self.k if self.memory_type == MemoryType.WINDOW else None,
        }

    def create_runnable_with_history(self, runnable: Any) -> RunnableWithMessageHistory:
        """
        Wrap a runnable with message history management.

        Args:
            runnable: A LangChain runnable (chain, llm, etc.)

        Returns:
            RunnableWithMessageHistory that automatically manages history
        """
        return RunnableWithMessageHistory(
            runnable,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )


class VectorMemory:
    """
    Vector-based semantic memory using ChromaDB.

    Stores conversation history in a vector database for semantic retrieval.
    Useful for finding relevant past conversations.
    """

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self._embeddings = OllamaEmbeddings(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )
        self._vector_store: Optional[Chroma] = None

    def _get_vector_store(self) -> Chroma:
        """Get or create the vector store."""
        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name=f"memory_{self.session_id}",
                embedding_function=self._embeddings,
                persist_directory=str(config.VECTOR_MEMORY_DIR),
            )
        return self._vector_store

    def add_conversation(self, human_msg: str, ai_msg: str) -> None:
        """Add a conversation turn to vector memory."""
        store = self._get_vector_store()
        text = f"Human: {human_msg}\nAI: {ai_msg}"
        store.add_texts([text])

    def search(self, query: str, k: int = 3) -> List[str]:
        """Search for relevant past conversations."""
        store = self._get_vector_store()
        docs = store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def clear(self) -> None:
        """Clear vector memory."""
        if self._vector_store is not None:
            self._vector_store.delete_collection()
            self._vector_store = None


# Convenience functions
def create_buffer_memory(session_id: str = "default") -> MemoryManager:
    """Create simple buffer memory."""
    return MemoryManager(memory_type=MemoryType.BUFFER, session_id=session_id)


def create_window_memory(k: int = 5, session_id: str = "default") -> MemoryManager:
    """Create windowed buffer memory."""
    return MemoryManager(memory_type=MemoryType.WINDOW, session_id=session_id, k=k)


def create_langgraph_memory() -> MemorySaver:
    """Create LangGraph checkpoint memory."""
    return MemorySaver()
