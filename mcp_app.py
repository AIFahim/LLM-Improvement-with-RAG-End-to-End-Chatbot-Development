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
    st.session_state.setdefault("summary", None)
    # Rolling running summary used as context on each agent turn.
    # Grows turn-by-turn instead of stuffing literal history into the prompt.
    st.session_state.setdefault("running_summary", "")


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
    # Fresh conversation per server: the new server has different tools,
    # so carrying over the prior chat would just confuse both the user
    # (different tool set in trace) and the agent (history references
    # tools no longer present).
    st.session_state["messages"] = []
    st.session_state["summary"] = None
    st.session_state["running_summary"] = ""


def build_llm(model: str, base_url: str) -> LLM:
    return LLM(model=f"ollama/{model}", base_url=base_url)


def update_running_summary(
    prev_summary: str, user_msg: str, assistant_reply: str, llm: LLM
) -> str:
    """Roll the conversation summary forward by one turn.

    Folds the new (user, assistant) exchange into the previous summary so
    the next agent turn sees a single compact context string instead of
    an ever-growing literal transcript. Classic LangChain
    ConversationSummaryMemory pattern, here as a tiny no-tools CrewAI
    task so it stays consistent with the rest of the app.
    """
    description = (
        f"Previous summary of the conversation:\n"
        f"{prev_summary or '(no prior turns yet)'}\n\n"
        f"New turn just completed:\n"
        f"User: {user_msg}\n"
        f"Assistant: {assistant_reply}\n\n"
        f"Produce an UPDATED summary that covers everything the user and "
        f"assistant have discussed so far, including the new turn. Keep it "
        f"under 5 sentences. If a tool was called, mention it briefly. "
        f"This summary will be passed to the assistant as context for the "
        f"NEXT user message, so preserve any facts/preferences/identifiers "
        f"the user has shared."
    )
    agent = Agent(
        role="Conversation Summarizer",
        goal="Maintain a short rolling summary of an ongoing conversation.",
        backstory=(
            "You produce neutral, faithful, compact summaries that preserve "
            "the user's stated facts and preferences across turns."
        ),
        tools=[],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    task = Task(
        description=description,
        expected_output="Updated rolling summary, under 5 sentences.",
        agent=agent,
    )
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
        memory=False,
    )
    return str(crew.kickoff()).strip()


def summarize_conversation(llm: LLM) -> str:
    """Produce a short summary of the current chat. No MCP tools needed —
    summarization is a pure LLM task, so the agent has no tools."""
    messages = st.session_state["messages"]
    if not messages:
        return "No conversation to summarize yet."

    transcript = "\n".join(
        f"{m['role'].title()}: {m['content']}" for m in messages
    )

    agent = Agent(
        role="Summarizer",
        goal="Summarize a chatbot conversation accurately and concisely.",
        backstory=(
            "You write faithful, neutral summaries. You don't add facts "
            "that weren't in the conversation, and you don't omit major "
            "topics that were."
        ),
        tools=[],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    task = Task(
        description=(
            "Summarize the conversation below as 3-5 bullet points covering "
            "the main topics discussed and any conclusions reached. If a "
            "tool was called, mention which tool and what it returned at a "
            "high level.\n\n"
            f"---\n{transcript}\n---"
        ),
        expected_output="A short bulleted summary.",
        agent=agent,
    )
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
        memory=False,
    )
    return str(crew.kickoff())


def chat_turn(user_message: str, llm: LLM, base_url: str) -> tuple[str, str]:
    """Run one CrewAI turn given the new user message. Returns (reply, trace).

    Memory: we use a *rolling summary* (st.session_state['running_summary'])
    that gets folded forward after each turn, NOT CrewAI's `memory=True`.
    Why not memory=True? CrewAI 1.14's memory layer requires Chroma vector
    storage, which on this stack fails to init without OPENAI_API_KEY even
    when EMBEDDINGS_OLLAMA_* env vars are set (commit f0a8ce7 hit the same
    issue). The rolling summary works without any embedder, keeps prompts
    bounded regardless of conversation length, and is fully visible to
    students. Cost: one extra LLM call per turn to roll the summary
    forward.
    """
    tools = st.session_state["tools"]
    running_summary = st.session_state.get("running_summary", "")

    backstory = (
        "You are a friendly conversational assistant. You have access to "
        "MCP tools (which may include math, date/time, web search, file/"
        "report operations, or repo Q&A depending on the connected server) "
        "— but you only reach for a tool when the user actually asks for "
        "something a tool can answer. Most messages (greetings, opinions, "
        "casual chat, general questions you already know) need NO tool "
        "calls — just reply naturally. If the user says 'hi', say 'hi' "
        "back. If they ask 'what's 17 * 23', use the calculator. The "
        "principle: tools serve the user's actual request; never call "
        "tools to demonstrate capability or fill silence."
    )

    task_description = (
        f"Summary of the conversation so far:\n{running_summary}\n\n"
        if running_summary else ""
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

    reply = str(result)
    trace = strip_ansi(buf.getvalue())

    # Roll the summary forward so the NEXT turn sees this exchange
    # compressed into the running summary. If summarization fails we
    # don't want it to break the chat, so fall back to a literal
    # concatenation.
    try:
        st.session_state["running_summary"] = update_running_summary(
            running_summary, user_message, reply, llm
        )
    except Exception:
        prev = running_summary
        st.session_state["running_summary"] = (
            (prev + "\n" if prev else "")
            + f"User: {user_message}\nAssistant: {reply}"
        )

    return reply, trace


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
    st.sidebar.subheader("Conversation")
    cols = st.sidebar.columns([1, 1])
    if cols[0].button("Summarize", use_container_width=True):
        if not st.session_state["messages"]:
            st.session_state["summary"] = "No conversation to summarize yet."
        else:
            with st.spinner("Summarizing..."):
                try:
                    st.session_state["summary"] = summarize_conversation(
                        build_llm(model, base_url)
                    )
                except Exception as e:
                    st.session_state["summary"] = f"Summarize failed: {e}"
    if cols[1].button("Clear", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["summary"] = None
        st.session_state["running_summary"] = ""
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

    if st.session_state.get("summary"):
        with st.expander("Conversation summary", expanded=True):
            st.markdown(st.session_state["summary"])
            if st.button("Dismiss summary"):
                st.session_state["summary"] = None
                st.rerun()

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
