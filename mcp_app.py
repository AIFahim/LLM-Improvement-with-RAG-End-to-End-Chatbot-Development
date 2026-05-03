"""
Class 07: MCP Chatbot - Streamlit UI

A conversational chatbot powered by CrewAI, where the agent's tools come
from an MCP server. Pick which MCP server to connect to (a local Python
stdio server or a remote DeepWiki HTTP server) and chat with the agent.

This is the "everyday" shape students should recognize: a chat box, with
the agent transparently calling MCP tools when needed.

Run:
    streamlit run mcp_app.py
"""

import io
import os
import re
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any

import streamlit as st
from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters

REPO_ROOT = Path(__file__).resolve().parent
LOCAL_SERVER_SCRIPT = REPO_ROOT / "mcp_server.py"
DEEPWIKI_URL = "https://mcp.deepwiki.com/mcp"

SERVER_OPTIONS = {
    "Local Python (mcp_server.py)": "local",
    "Remote DeepWiki (HTTP)": "remote",
}


st.set_page_config(
    page_title="Class 07: MCP Chatbot",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def init_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("adapter", None)
    st.session_state.setdefault("tools", [])
    st.session_state.setdefault("server_kind", None)
    st.session_state.setdefault("server_label", None)


def server_params_for(kind: str) -> Any:
    if kind == "local":
        return StdioServerParameters(
            command=sys.executable,
            args=[str(LOCAL_SERVER_SCRIPT)],
            env=None,
        )
    if kind == "remote":
        return {"url": DEEPWIKI_URL, "transport": "streamable-http"}
    raise ValueError(f"Unknown server kind: {kind}")


def disconnect() -> None:
    adapter = st.session_state.get("adapter")
    if adapter is not None:
        try:
            adapter.stop()
        except Exception:
            pass
    st.session_state["adapter"] = None
    st.session_state["tools"] = []
    st.session_state["server_kind"] = None
    st.session_state["server_label"] = None


def connect(kind: str, label: str) -> None:
    disconnect()
    adapter = MCPServerAdapter(server_params_for(kind), connect_timeout=120)
    st.session_state["adapter"] = adapter
    st.session_state["tools"] = list(adapter.tools)
    st.session_state["server_kind"] = kind
    st.session_state["server_label"] = label


def build_llm(model: str, base_url: str) -> LLM:
    return LLM(model=f"ollama/{model}", base_url=base_url)


def render_history_for_prompt(max_turns: int = 8) -> str:
    """Render recent conversation turns as plain text for the task prompt.

    CrewAI's Task doesn't carry conversation memory natively, so we
    inject the recent transcript into the task description on each turn.
    Limit to the last N turns to keep prompts bounded.
    """
    history = st.session_state["messages"][-(max_turns * 2):]
    if not history:
        return ""
    lines = []
    for m in history:
        speaker = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {m['content']}")
    return "\n".join(lines)


def chat_turn(user_message: str, llm: LLM, base_url: str) -> tuple[str, str]:
    """Run one CrewAI turn given the new user message. Returns (reply, trace).

    Memory: we use *manual transcript injection* via
    `render_history_for_prompt`, NOT CrewAI's `memory=True`. Why:

    - CrewAI 1.14's `memory=True` requires Chroma vector storage, and on
      this stack Chroma fails to initialize without OPENAI_API_KEY even
      when EMBEDDINGS_OLLAMA_* env vars are set. Verified empirically:
      Turn 1 errors with "Memory requires an embedder for vector search
      but initialization failed: The CHROMA_OPENAI_API_KEY..." This is
      why commit f0a8ce7 explicitly disabled it on this branch.
    - Manual transcript injection works without any embedder, gives
      literal recall of recent turns (what a chatbot needs), and keeps
      the prompt fully visible to students.

    If you upgrade CrewAI / Chroma in the future, retry `memory=True` —
    it may start working without OPENAI_API_KEY.
    """
    tools = st.session_state["tools"]
    history = render_history_for_prompt()

    backstory = (
        "You are a helpful conversational assistant. You answer the user's "
        "questions and help with their tasks. When tools are available you "
        "use them — never invent facts that a tool could verify. You speak "
        "in the same friendly tone the user uses."
    )

    task_description = (
        f"{history}\n\n" if history else ""
    ) + f"User: {user_message}\nAssistant:"

    agent = Agent(
        role="Helpful Assistant",
        goal="Have a useful conversation with the user, using tools when needed.",
        backstory=backstory,
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
    task = Task(
        description=task_description.strip(),
        expected_output="A direct, conversational reply to the user.",
        agent=agent,
    )
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
        memory=False,
    )

    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        result = crew.kickoff()

    return str(result), strip_ansi(buf.getvalue())


def sidebar() -> tuple[str, str]:
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

    st.sidebar.code(
        _server_params_summary(chosen_kind),
        language="python",
    )

    cols = st.sidebar.columns([1, 1])
    if cols[0].button("Connect", use_container_width=True):
        with st.spinner(f"Connecting to {label}..."):
            try:
                connect(chosen_kind, label)
                st.sidebar.success(f"Connected to {label}.")
            except Exception as e:
                st.sidebar.error(f"Connect failed: {e}")
    if cols[1].button("Disconnect", use_container_width=True):
        disconnect()

    if st.session_state.get("adapter") is not None:
        st.sidebar.success(
            f"Active: {st.session_state['server_label']} "
            f"({len(st.session_state['tools'])} tools)"
        )
        with st.sidebar.expander("Available tools", expanded=False):
            for t in st.session_state["tools"]:
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
    if st.sidebar.button("Clear conversation"):
        st.session_state["messages"] = []
        st.rerun()

    return model, base_url


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


def render_messages() -> None:
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            trace = msg.get("trace")
            if trace:
                with st.expander("MCP protocol trace (raw tool calls + outputs)"):
                    st.code(trace, language="text")


def main() -> None:
    init_session_state()
    st.title("MCP Chatbot")
    st.caption(
        "Class 07 — chat with a CrewAI agent whose tools come from an MCP "
        "server. Switch the server in the sidebar to see the same agent "
        "code reach for different tools."
    )

    model, base_url = sidebar()
    llm = build_llm(model, base_url)

    render_messages()

    user_message = st.chat_input("Ask the agent something...")
    if user_message:
        if st.session_state.get("adapter") is None:
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
                    reply, trace = chat_turn(user_message, llm, base_url)
                except Exception as e:
                    reply = f"Error: {e}"
                    trace = ""
            st.markdown(reply)
            if trace:
                with st.expander("MCP protocol trace (raw tool calls + outputs)"):
                    st.code(trace, language="text")

        st.session_state["messages"].append(
            {"role": "assistant", "content": reply, "trace": trace}
        )


if __name__ == "__main__":
    main()
