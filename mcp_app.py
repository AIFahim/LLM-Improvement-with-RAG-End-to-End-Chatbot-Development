"""
MCP Chatbot — Streamlit UI.

Thin presentation layer. All orchestration logic lives in mcp_main.py /
mcp_agents.py / mcp_tasks.py — this file only handles widget rendering
and Streamlit session state.

Run:
    streamlit run mcp_app.py
"""

import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st
from mcp import StdioServerParameters

from mcp_main import MCPChatbot, build_llm

REPO_ROOT = Path(__file__).resolve().parent
LOCAL_SERVER_SCRIPT = REPO_ROOT / "mcp_server.py"
DEEPWIKI_URL = "https://mcp.deepwiki.com/mcp"

SERVER_OPTIONS = {
    "Local Python (mcp_server.py)": "local",
    "Remote DeepWiki (HTTP)": "remote",
}


st.set_page_config(
    page_title="MCP Chatbot",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Streamlit session state
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    st.session_state.setdefault("chatbot", MCPChatbot())
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("server_label", None)
    st.session_state.setdefault("summary", None)


def server_params_for(kind: str) -> Any:
    """Build the MCP connection params for one of the supported shapes."""
    if kind == "local":
        return StdioServerParameters(
            command=sys.executable,
            args=[str(LOCAL_SERVER_SCRIPT)],
            env=None,
        )
    if kind == "remote":
        return {"url": DEEPWIKI_URL, "transport": "streamable-http"}
    raise ValueError(f"Unknown server kind: {kind}")


def connect(label: str) -> None:
    chatbot: MCPChatbot = st.session_state["chatbot"]
    chatbot.connect(server_params_for(SERVER_OPTIONS[label]))
    st.session_state["server_label"] = label
    st.session_state["messages"] = []
    st.session_state["summary"] = None


def disconnect() -> None:
    chatbot: MCPChatbot = st.session_state["chatbot"]
    chatbot.disconnect()
    st.session_state["server_label"] = None
    st.session_state["messages"] = []
    st.session_state["summary"] = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _server_params_summary(kind: str) -> str:
    if kind == "local":
        return (
            f"command={sys.executable!r}\n"
            f"args=[{str(LOCAL_SERVER_SCRIPT)!r}]\n"
            f"transport='stdio'"
        )
    if kind == "remote":
        return f"url={DEEPWIKI_URL!r}\ntransport='streamable-http'"
    return ""


def sidebar() -> tuple[str, str]:
    chatbot: MCPChatbot = st.session_state["chatbot"]

    st.sidebar.title("MCP Chatbot")
    st.sidebar.caption(
        "A CrewAI agent that gets its tools from an MCP server. "
        "Pick a server, connect, and chat."
    )

    st.sidebar.subheader("MCP server")
    label = st.sidebar.radio(
        "Choose a server",
        options=list(SERVER_OPTIONS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    chosen_kind = SERVER_OPTIONS[label]
    st.sidebar.code(_server_params_summary(chosen_kind), language="python")

    cols = st.sidebar.columns([1, 1])
    if cols[0].button("Connect", use_container_width=True):
        with st.spinner(f"Connecting to {label}..."):
            try:
                connect(label)
                st.sidebar.success(f"Connected to {label}.")
            except Exception as e:
                st.sidebar.error(f"Connect failed: {e}")
    if cols[1].button("Disconnect", use_container_width=True):
        disconnect()

    if chatbot.is_connected:
        st.sidebar.success(
            f"Active: {st.session_state['server_label']} "
            f"({len(chatbot.tools)} tools)"
        )
        with st.sidebar.expander("Available tools", expanded=False):
            for t in chatbot.tools:
                st.markdown(f"**`{t.name}`**")
                desc = (t.description or "").strip()
                st.caption(desc[:200] + ("..." if len(desc) > 200 else ""))
    else:
        st.sidebar.info("Not connected. Click Connect above.")

    st.sidebar.subheader("LLM")
    model = st.sidebar.text_input(
        "Ollama model",
        value=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
        help="Bigger models route tools more reliably.",
    )
    base_url = st.sidebar.text_input(
        "Ollama base URL",
        value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    st.sidebar.divider()
    st.sidebar.subheader("Conversation")
    cc = st.sidebar.columns([1, 1])
    if cc[0].button("Summarize", use_container_width=True):
        try:
            st.session_state["summary"] = chatbot.recap(
                st.session_state["messages"], build_llm(model, base_url)
            )
        except Exception as e:
            st.session_state["summary"] = f"Summarize failed: {e}"
    if cc[1].button("Clear", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["summary"] = None
        chatbot.reset_chat()
        st.rerun()

    return model, base_url


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

def render_messages() -> None:
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("trace"):
                with st.expander(
                    "MCP protocol trace (raw tool calls + outputs)"
                ):
                    st.code(msg["trace"], language="text")


def main() -> None:
    init_session_state()
    chatbot: MCPChatbot = st.session_state["chatbot"]

    st.title("MCP Chatbot")
    st.caption(
        "Class 07 — chat with a CrewAI agent whose tools come from an "
        "MCP server. Switch the server in the sidebar to see the same "
        "agent code reach for different tools."
    )

    model, base_url = sidebar()
    llm = build_llm(model, base_url)

    if st.session_state.get("summary"):
        with st.expander("Conversation summary", expanded=True):
            st.markdown(st.session_state["summary"])
            if st.button("Dismiss summary"):
                st.session_state["summary"] = None
                st.rerun()

    render_messages()

    user_message = st.chat_input("Ask the agent something...")
    if user_message:
        if not chatbot.is_connected:
            st.error("Connect to an MCP server first (sidebar).")
            return

        st.session_state["messages"].append(
            {"role": "user", "content": user_message}
        )
        with st.chat_message("user"):
            st.markdown(user_message)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    reply, trace = chatbot.chat_turn(user_message, llm)
                except Exception as e:
                    reply = f"Error: {e}"
                    trace = ""
            st.markdown(reply)
            if trace:
                with st.expander(
                    "MCP protocol trace (raw tool calls + outputs)"
                ):
                    st.code(trace, language="text")

        st.session_state["messages"].append(
            {"role": "assistant", "content": reply, "trace": trace}
        )


if __name__ == "__main__":
    main()
