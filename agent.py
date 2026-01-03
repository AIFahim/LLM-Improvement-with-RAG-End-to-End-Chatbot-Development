"""
LangChain Agent Module - LangChain Deep Dive (Class 06)

Implements conversational agent using LangGraph's ReAct pattern.
Updated for LangChain 1.x / LangGraph architecture.

Components:
- create_react_agent from langgraph.prebuilt
- MemorySaver for conversation persistence
- Custom tools integration
"""

import logging
from typing import Optional, List, Dict, Any, Union

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from memory_manager import MemoryManager, MemoryType, get_session_history
from tools import ToolFactory, create_all_tools
import config

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a helpful AI assistant with access to various tools.

Available tools:
- calculator: For mathematical calculations
- datetime: For current date/time information
- web_search: For searching the internet
- python_repl: For executing Python code
- rag_search: For searching uploaded documents

Use tools when needed to provide accurate answers. Think step by step.
Be concise and helpful in your responses."""


class LangChainAgent:
    """
    Main agent class using LangGraph's ReAct pattern.

    Features:
    - ReAct reasoning with tool use
    - Multiple tool support
    - Conversation memory via MemorySaver
    """

    def __init__(
        self,
        memory_type: Union[MemoryType, str] = MemoryType.BUFFER,
        tools: Optional[List[BaseTool]] = None,
        vector_store: Any = None,
        verbose: bool = None,
    ):
        self.verbose = verbose if verbose is not None else config.AGENT_VERBOSE
        self.vector_store = vector_store

        # Initialize LLM (use ChatOllama for better tool support)
        self.llm = self._create_llm()

        # Initialize Memory
        self.memory_manager = MemoryManager(
            memory_type=memory_type,
            session_id="agent_session",
        )
        self.checkpointer = MemorySaver()

        # Initialize Tools
        self.tool_factory = ToolFactory(vector_store=vector_store)
        self.tools = tools or self.tool_factory.get_all_tools()

        # Create Agent
        self._agent = None
        self._thread_id = "agent_thread_1"

        logger.info(f"LangChainAgent initialized with {len(self.tools)} tools")

    def _create_llm(self) -> Any:
        """Create LLM instance (ChatOllama for better agent support)."""
        if config.LLM_PROVIDER == "azure":
            return AzureChatOpenAI(
                azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                api_key=config.AZURE_OPENAI_API_KEY,
                api_version=config.AZURE_OPENAI_API_VERSION,
                deployment_name=config.AZURE_OPENAI_DEPLOYMENT,
                temperature=config.LLM_TEMPERATURE,
            )
        return ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.LLM_TEMPERATURE,
        )

    def _create_agent(self):
        """Create the LangGraph ReAct agent."""
        return create_react_agent(
            model=self.llm,
            tools=self.tools,
            checkpointer=self.checkpointer,
        )

    @property
    def agent(self):
        """Lazy load agent."""
        if self._agent is None:
            self._agent = self._create_agent()
        return self._agent

    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the agent with a query.

        Args:
            query: User input/question

        Returns:
            Dict with 'output', 'messages', and metadata
        """
        try:
            logger.info(f"Agent processing: {query[:50]}...")

            # Add to memory
            self.memory_manager.add_user_message(query)

            # Create input with system message
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=query),
            ]

            # Run agent
            config_dict = {"configurable": {"thread_id": self._thread_id}}
            result = self.agent.invoke({"messages": messages}, config=config_dict)

            # Extract response
            output_messages = result.get("messages", [])
            final_response = ""
            tool_calls = []

            for msg in output_messages:
                if isinstance(msg, AIMessage):
                    if msg.content:
                        final_response = msg.content
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        tool_calls.extend(msg.tool_calls)

            # Add AI response to memory
            if final_response:
                self.memory_manager.add_ai_message(final_response)

            return {
                "output": final_response,
                "messages": output_messages,
                "tool_calls": tool_calls,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Agent error: {e}")
            return {
                "output": f"Error: {str(e)}",
                "messages": [],
                "tool_calls": [],
                "success": False,
                "error": str(e),
            }

    def chat(self, message: str) -> str:
        """Simple chat interface returning just the response."""
        result = self.run(message)
        return result["output"]

    def get_tool_names(self) -> List[str]:
        """Get list of available tool names."""
        return [tool.name for tool in self.tools]

    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory information."""
        return self.memory_manager.get_memory_info()

    def clear_memory(self) -> None:
        """Clear conversation memory."""
        self.memory_manager.clear()
        self.checkpointer = MemorySaver()
        self._agent = None
        logger.info("Agent memory cleared")

    def update_vector_store(self, vector_store: Any) -> None:
        """Update vector store for RAG tool."""
        self.vector_store = vector_store
        self.tool_factory.update_vector_store(vector_store)
        self.tools = self.tool_factory.get_all_tools()
        self._agent = None


class SimpleChain:
    """Simple chain for conversations without tools."""

    def __init__(self, memory_type: Union[MemoryType, str] = MemoryType.BUFFER):
        self.llm = self._create_llm()
        self.memory_manager = MemoryManager(
            memory_type=memory_type,
            session_id="simple_chain",
        )

    def _create_llm(self) -> Any:
        if config.LLM_PROVIDER == "azure":
            return AzureChatOpenAI(
                azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                api_key=config.AZURE_OPENAI_API_KEY,
                api_version=config.AZURE_OPENAI_API_VERSION,
                deployment_name=config.AZURE_OPENAI_DEPLOYMENT,
                temperature=config.LLM_TEMPERATURE,
            )
        return ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.LLM_TEMPERATURE,
        )

    def run(self, query: str) -> str:
        """Run the chain with a query."""
        self.memory_manager.add_user_message(query)

        # Get conversation history
        history = self.memory_manager.get_messages_as_string()

        # Create prompt with history
        messages = [
            SystemMessage(content="You are a helpful assistant."),
        ]

        if history:
            messages.append(SystemMessage(content=f"Previous conversation:\n{history}"))

        messages.append(HumanMessage(content=query))

        # Get response
        response = self.llm.invoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)

        self.memory_manager.add_ai_message(content)
        return content

    def clear_memory(self) -> None:
        """Clear conversation memory."""
        self.memory_manager.clear()


class ConversationalAgent:
    """
    High-level conversational agent combining memory and tools.
    Provides a simple interface for building chat applications.
    """

    def __init__(
        self,
        use_tools: bool = True,
        memory_type: Union[MemoryType, str] = MemoryType.BUFFER,
        vector_store: Any = None,
    ):
        self.use_tools = use_tools

        if use_tools:
            self._agent = LangChainAgent(
                memory_type=memory_type,
                vector_store=vector_store,
            )
        else:
            self._agent = SimpleChain(memory_type=memory_type)

    def chat(self, message: str) -> str:
        """Send a message and get a response."""
        if self.use_tools:
            return self._agent.chat(message)
        return self._agent.run(message)

    def get_full_response(self, message: str) -> Dict[str, Any]:
        """Get full response including tool calls (tools mode only)."""
        if self.use_tools:
            return self._agent.run(message)
        return {"output": self._agent.run(message), "messages": [], "tool_calls": []}

    def clear_memory(self) -> None:
        """Clear conversation memory."""
        self._agent.clear_memory()

    def get_tools(self) -> List[str]:
        """Get available tools (empty if tools disabled)."""
        if self.use_tools:
            return self._agent.get_tool_names()
        return []

    @property
    def memory_info(self) -> Dict[str, Any]:
        """Get memory information."""
        if self.use_tools:
            return self._agent.get_memory_info()
        return self._agent.memory_manager.get_memory_info()


def create_agent(
    memory_type: str = "buffer",
    use_tools: bool = True,
    vector_store: Any = None,
) -> ConversationalAgent:
    """
    Factory function to create a conversational agent.

    Args:
        memory_type: Type of memory (buffer, window, vector)
        use_tools: Whether to enable tools
        vector_store: Optional vector store for RAG

    Returns:
        ConversationalAgent instance
    """
    return ConversationalAgent(
        use_tools=use_tools,
        memory_type=memory_type,
        vector_store=vector_store,
    )
