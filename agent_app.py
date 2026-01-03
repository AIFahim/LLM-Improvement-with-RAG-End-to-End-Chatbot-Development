"""
Streamlit UI for LangChain Agent - Class 06: LangChain Deep Dive

This application provides a web interface for the conversational agent
with memory and tools support.
"""

import streamlit as st
import logging
from typing import Optional

from agent import LangChainAgent, ConversationalAgent, create_agent
from memory_manager import MemoryType
from tools import ToolFactory
from vector_store import VectorStoreManager
from document_processor import DocumentProcessor
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "memory_type" not in st.session_state:
        st.session_state.memory_type = "buffer"
    if "tools_enabled" not in st.session_state:
        st.session_state.tools_enabled = True
    if "show_thinking" not in st.session_state:
        st.session_state.show_thinking = True


def create_agent_instance(memory_type: str, use_tools: bool, vector_store=None):
    """Create or recreate the agent instance."""
    st.session_state.agent = create_agent(
        memory_type=memory_type,
        use_tools=use_tools,
        vector_store=vector_store,
    )
    return st.session_state.agent


def display_chat_messages():
    """Display chat message history."""
    for message in st.session_state.agent_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("thinking"):
                with st.expander("Agent Thinking Process"):
                    st.markdown(message["thinking"])


def format_thinking_steps(steps: list) -> str:
    """Format intermediate steps for display."""
    if not steps:
        return ""

    thinking = []
    for i, (action, observation) in enumerate(steps, 1):
        tool_name = action.tool
        tool_input = action.tool_input
        thinking.append(f"**Step {i}: Used `{tool_name}`**")
        thinking.append(f"- Input: `{tool_input}`")
        thinking.append(f"- Result: {observation[:200]}..." if len(str(observation)) > 200 else f"- Result: {observation}")
        thinking.append("")

    return "\n".join(thinking)


def sidebar_controls():
    """Render sidebar with agent configuration."""
    with st.sidebar:
        st.header("Agent Configuration")

        # Memory Type Selection
        memory_options = {
            "Buffer (Short-term)": "buffer",
            "Buffer Window (Last K)": "buffer_window",
            "Summary (Long-term)": "summary",
            "Summary Buffer (Hybrid)": "summary_buffer",
            "Vector (Semantic)": "vector",
            "Combined (Buffer + Vector)": "combined",
        }

        selected_memory = st.selectbox(
            "Memory Type",
            options=list(memory_options.keys()),
            help="Choose how the agent remembers conversation history"
        )
        new_memory_type = memory_options[selected_memory]

        # Tools Toggle
        use_tools = st.checkbox(
            "Enable Tools",
            value=st.session_state.tools_enabled,
            help="Enable/disable agent tools (calculator, web search, etc.)"
        )

        # Show Thinking Toggle
        show_thinking = st.checkbox(
            "Show Agent Thinking",
            value=st.session_state.show_thinking,
            help="Display the agent's reasoning process"
        )
        st.session_state.show_thinking = show_thinking

        # Apply Configuration Button
        if st.button("Apply Configuration"):
            st.session_state.memory_type = new_memory_type
            st.session_state.tools_enabled = use_tools
            create_agent_instance(
                new_memory_type,
                use_tools,
                st.session_state.vector_store
            )
            st.success("Configuration applied!")
            st.rerun()

        st.divider()

        # Available Tools Display
        if st.session_state.agent and st.session_state.tools_enabled:
            st.subheader("Available Tools")
            tools = st.session_state.agent.get_tools()
            for tool in tools:
                st.markdown(f"- {tool}")

        st.divider()

        # Memory Info
        if st.session_state.agent:
            st.subheader("Memory Info")
            info = st.session_state.agent.memory_info
            st.json(info)

        st.divider()

        # PDF Upload for RAG
        st.subheader("Document Upload (RAG)")
        uploaded_files = st.file_uploader(
            "Upload PDFs for RAG Search",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload documents to enable RAG search tool"
        )

        if uploaded_files and st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                try:
                    processor = DocumentProcessor()
                    vector_manager = VectorStoreManager()

                    all_docs = []
                    for file in uploaded_files:
                        docs = processor.process_pdf(file)
                        all_docs.extend(docs)

                    if all_docs:
                        vector_manager.create_vector_store(all_docs)
                        st.session_state.vector_store = vector_manager.get_vector_store()

                        # Recreate agent with vector store
                        create_agent_instance(
                            st.session_state.memory_type,
                            st.session_state.tools_enabled,
                            st.session_state.vector_store
                        )
                        st.success(f"Processed {len(uploaded_files)} documents!")
                except Exception as e:
                    st.error(f"Error processing documents: {e}")

        st.divider()

        # Clear History Button
        if st.button("Clear Chat History"):
            st.session_state.agent_messages = []
            if st.session_state.agent:
                st.session_state.agent.clear_memory()
            st.success("Chat history cleared!")
            st.rerun()


def main():
    """Main application function."""
    st.set_page_config(
        page_title=config.AGENT_PAGE_TITLE,
        page_icon=config.AGENT_PAGE_ICON,
        layout="wide",
    )

    st.title("LangChain Agent with Memory & Tools")
    st.caption("Class 06: LangChain Deep Dive - Memory, Tools & Agents")

    # Initialize session state
    initialize_session_state()

    # Create default agent if not exists
    if st.session_state.agent is None:
        create_agent_instance(
            st.session_state.memory_type,
            st.session_state.tools_enabled,
            st.session_state.vector_store
        )

    # Render sidebar
    sidebar_controls()

    # Main chat interface
    st.markdown("---")

    # Display chat history
    display_chat_messages()

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.agent_messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.agent.get_full_response(prompt)
                    response = result["output"]
                    steps = result.get("intermediate_steps", [])

                    st.markdown(response)

                    # Show thinking process
                    thinking = ""
                    if steps and st.session_state.show_thinking:
                        thinking = format_thinking_steps(steps)
                        with st.expander("Agent Thinking Process"):
                            st.markdown(thinking)

                    # Save to history
                    st.session_state.agent_messages.append({
                        "role": "assistant",
                        "content": response,
                        "thinking": thinking if st.session_state.show_thinking else None
                    })

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.agent_messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

    # Footer with info
    st.markdown("---")
    cols = st.columns(3)
    with cols[0]:
        st.caption(f"Memory: {st.session_state.memory_type}")
    with cols[1]:
        st.caption(f"Tools: {'Enabled' if st.session_state.tools_enabled else 'Disabled'}")
    with cols[2]:
        st.caption(f"Provider: {config.LLM_PROVIDER}")


if __name__ == "__main__":
    main()
