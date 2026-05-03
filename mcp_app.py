"""
Class 07: MCP Demo - Streamlit UI

Interactive demo of CrewAI agents calling tools across three MCP server
deployment shapes:

  Tab 1: Local stdio  - our own Python FastMCP server (mcp_server.py)
  Tab 2: External stdio - Anthropic's @modelcontextprotocol/server-filesystem (npx)
  Tab 3: Remote HTTP - DeepWiki's hosted SaaS MCP server (no auth)

Same agent code in each tab; only the server params change. That's the
whole MCP value proposition in one screen.

Run:
    streamlit run mcp_app.py
"""

import io
import os
import re
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import streamlit as st
from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters

REPO_ROOT = Path(__file__).resolve().parent
LOCAL_SERVER_SCRIPT = REPO_ROOT / "mcp_server.py"
SANDBOX = REPO_ROOT / "mcp_sandbox"
DEEPWIKI_URL = "https://mcp.deepwiki.com/mcp"


st.set_page_config(
    page_title="Class 07: MCP Demos",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state() -> None:
    for key in ("adapter_local", "adapter_external", "adapter_remote"):
        st.session_state.setdefault(key, None)
    for key in ("tools_local", "tools_external", "tools_remote"):
        st.session_state.setdefault(key, [])
    for key in ("result_local", "result_external", "result_remote"):
        st.session_state.setdefault(key, None)
    for key in ("trace_local", "trace_external", "trace_remote"):
        st.session_state.setdefault(key, None)


def build_llm(model: str, base_url: str) -> LLM:
    return LLM(model=f"ollama/{model}", base_url=base_url)


def disconnect(slot: str) -> None:
    adapter = st.session_state.get(f"adapter_{slot}")
    if adapter is not None:
        try:
            adapter.stop()
        except Exception:
            pass
    st.session_state[f"adapter_{slot}"] = None
    st.session_state[f"tools_{slot}"] = []


def connect(slot: str, server_params) -> None:
    disconnect(slot)
    adapter = MCPServerAdapter(server_params, connect_timeout=120)
    st.session_state[f"adapter_{slot}"] = adapter
    st.session_state[f"tools_{slot}"] = list(adapter.tools)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def run_crew(
    slot: str,
    role: str,
    goal: str,
    backstory: str,
    task_description: str,
    expected_output: str,
    llm: LLM,
) -> tuple[str, str]:
    """Run the crew and return (final_answer, captured_trace).

    The trace is the verbose stdout from crew.kickoff() — it contains the
    tool calls and their raw outputs, which is the high-value teaching
    artifact. Showing it lets students see what the MCP server actually
    returned, separate from how the LLM summarized it.
    """
    tools = st.session_state[f"tools_{slot}"]
    agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
    task = Task(
        description=task_description,
        expected_output=expected_output,
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
    trace = _strip_ansi(buf.getvalue())
    return str(result), trace


def render_tools(slot: str) -> None:
    tools = st.session_state[f"tools_{slot}"]
    if not tools:
        st.info("Not connected. Click **Connect** above.")
        return
    st.success(f"Connected. {len(tools)} tools discovered.")
    with st.expander(f"Inspect tools ({len(tools)})", expanded=False):
        for t in tools:
            st.markdown(f"**`{t.name}`**")
            desc = (t.description or "").strip()
            st.caption(desc[:300] + ("..." if len(desc) > 300 else ""))


def tab_local(llm: LLM) -> None:
    st.subheader("① Local stdio — your own Python MCP server")
    st.caption(
        "The agent calls tools defined in `mcp_server.py` (this repo). The "
        "server is launched as a stdio subprocess. You wrote it."
    )
    st.code(
        f"command={sys.executable!r}\nargs=[{str(LOCAL_SERVER_SCRIPT)!r}]\n"
        f"transport='stdio'",
        language="python",
    )

    c1, c2 = st.columns([1, 1])
    if c1.button("Connect", key="btn_connect_local"):
        with st.spinner("Launching local MCP server..."):
            connect(
                "local",
                StdioServerParameters(
                    command=sys.executable,
                    args=[str(LOCAL_SERVER_SCRIPT)],
                    env=None,
                ),
            )
    if c2.button("Disconnect", key="btn_disconnect_local"):
        disconnect("local")

    render_tools("local")

    if st.session_state["tools_local"]:
        if st.button("Run demo task: ping → list_reports → save_report",
                     key="btn_run_local"):
            with st.spinner("Agent is working..."):
                try:
                    result, trace = run_crew(
                        slot="local",
                        role="Report Archivist",
                        goal="Use MCP tools to inspect and save reports.",
                        backstory=(
                            "You manage a library of reports. You only ever "
                            "interact with the report store through MCP tools."
                        ),
                        task_description=(
                            "Step 1: call `ping` to confirm the server.\n"
                            "Step 2: call `list_reports`.\n"
                            "Step 3: call `save_report` with title 'mcp-ui' "
                            "and content 'Saved by Streamlit MCP demo.'\n"
                            "Return: ping response, count of reports, and "
                            "the new file path."
                        ),
                        expected_output="A short confirmation report.",
                        llm=llm,
                    )
                    st.session_state["result_local"] = result
                    st.session_state["trace_local"] = trace
                except Exception as e:
                    st.error(f"Run failed: {e}")

    if st.session_state["result_local"]:
        st.markdown("### Final answer (from the LLM)")
        st.markdown(st.session_state["result_local"])
        if st.session_state.get("trace_local"):
            with st.expander("Show MCP protocol trace (raw tool calls + outputs)"):
                st.code(st.session_state["trace_local"], language="text")


def tab_external(llm: LLM) -> None:
    st.subheader("② External stdio — Anthropic's npm filesystem server")
    st.caption(
        "The agent calls Anthropic-published Node.js code "
        "(`@modelcontextprotocol/server-filesystem`), launched on demand "
        "via `npx -y`. Tool author and language are different — the "
        "agent doesn't know or care."
    )
    st.code(
        f"command='npx'\n"
        f"args=['-y', '@modelcontextprotocol/server-filesystem', "
        f"{str(SANDBOX)!r}]\n"
        f"transport='stdio'",
        language="python",
    )
    st.warning(
        "First connect downloads the npm package (~30s). Requires `npx` on "
        "your PATH."
    )

    SANDBOX.mkdir(exist_ok=True)

    c1, c2 = st.columns([1, 1])
    if c1.button("Connect", key="btn_connect_external"):
        with st.spinner("Fetching and starting external MCP server..."):
            try:
                connect(
                    "external",
                    StdioServerParameters(
                        command="npx",
                        args=[
                            "-y",
                            "@modelcontextprotocol/server-filesystem",
                            str(SANDBOX),
                        ],
                        env=None,
                    ),
                )
            except Exception as e:
                st.error(f"Connect failed: {e}")
    if c2.button("Disconnect", key="btn_disconnect_external"):
        disconnect("external")

    render_tools("external")

    if st.session_state["tools_external"]:
        if st.button(
            "Run demo task: list_directory → write_file → read_file",
            key="btn_run_external",
        ):
            with st.spinner("Agent is working..."):
                try:
                    sandbox_path = str(SANDBOX)
                    result, trace = run_crew(
                        slot="external",
                        role="Sandbox Librarian",
                        goal=(
                            "Use the filesystem MCP tools to inventory the "
                            "sandbox directory and add a small note."
                        ),
                        backstory=(
                            "You only ever interact with the directory "
                            "through MCP tools — never invent contents."
                        ),
                        task_description=(
                            f"Step 1: call `list_directory` with path "
                            f"'{sandbox_path}'.\n"
                            f"Step 2: call `write_file` to create "
                            f"'{sandbox_path}/note.md' with content "
                            f"'# Note\\n\\nWritten by the Streamlit MCP demo.'\n"
                            f"Step 3: call `read_file` on "
                            f"'{sandbox_path}/note.md' to confirm."
                        ),
                        expected_output=(
                            "Original directory contents, write confirmation, "
                            "and the read-back file content."
                        ),
                        llm=llm,
                    )
                    st.session_state["result_external"] = result
                    st.session_state["trace_external"] = trace
                except Exception as e:
                    st.error(f"Run failed: {e}")

    if st.session_state["result_external"]:
        st.markdown("### Final answer (from the LLM)")
        st.markdown(st.session_state["result_external"])
        if st.session_state.get("trace_external"):
            with st.expander("Show MCP protocol trace (raw tool calls + outputs)"):
                st.code(st.session_state["trace_external"], language="text")


def tab_remote(llm: LLM) -> None:
    st.subheader("③ Remote HTTP — DeepWiki SaaS MCP server")
    st.caption(
        "The agent calls a server that isn't even on your machine — just a "
        "URL. Transport is Streamable HTTP (the modern MCP transport, "
        "replaces SSE)."
    )
    st.code(
        f"url={DEEPWIKI_URL!r}\ntransport='streamable-http'\n# no auth",
        language="python",
    )

    c1, c2 = st.columns([1, 1])
    if c1.button("Connect", key="btn_connect_remote"):
        with st.spinner("Connecting to DeepWiki..."):
            try:
                connect(
                    "remote",
                    {"url": DEEPWIKI_URL, "transport": "streamable-http"},
                )
            except Exception as e:
                st.error(f"Connect failed: {e}")
    if c2.button("Disconnect", key="btn_disconnect_remote"):
        disconnect("remote")

    render_tools("remote")

    if st.session_state["tools_remote"]:
        repo = st.text_input(
            "GitHub repo (owner/name)",
            value="crewAIInc/crewAI",
            key="remote_repo",
        )
        question = st.text_area(
            "Question for DeepWiki",
            value=(
                "What is the main purpose of this project, and what are its "
                "core abstractions?"
            ),
            height=80,
            key="remote_question",
        )

        if st.button("Run agent", key="btn_run_remote"):
            with st.spinner("Agent is querying DeepWiki..."):
                try:
                    result, trace = run_crew(
                        slot="remote",
                        role="Open-Source Researcher",
                        goal=(
                            "Use DeepWiki's MCP tools to answer questions "
                            "about real GitHub repositories."
                        ),
                        backstory=(
                            "You research open-source projects by querying "
                            "DeepWiki over MCP. You never invent answers."
                        ),
                        task_description=(
                            f"Use the `ask_question` tool with "
                            f"repoName='{repo}' and question='{question}'. "
                            f"Then summarize the answer for a student."
                        ),
                        expected_output=(
                            "A short summary sourced from DeepWiki."
                        ),
                        llm=llm,
                    )
                    st.session_state["result_remote"] = result
                    st.session_state["trace_remote"] = trace
                except Exception as e:
                    st.error(f"Run failed: {e}")

    if st.session_state["result_remote"]:
        st.markdown("### Final answer (from the LLM)")
        st.markdown(st.session_state["result_remote"])
        if st.session_state.get("trace_remote"):
            with st.expander(
                "Show MCP protocol trace (the raw DeepWiki answer is here)",
                expanded=False,
            ):
                st.code(st.session_state["trace_remote"], language="text")


def sidebar() -> tuple[str, str]:
    st.sidebar.title("Configuration")
    st.sidebar.markdown(
        "Three MCP servers, one protocol. Same agent code on every tab — "
        "only the server params change."
    )

    st.sidebar.subheader("LLM")
    model = st.sidebar.text_input(
        "Ollama model",
        value=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
        help="Bump to qwen2.5:7b-instruct or llama3.1:8b for more reliable "
             "tool routing if 3B flakes.",
    )
    base_url = st.sidebar.text_input(
        "Ollama base URL",
        value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    st.sidebar.subheader("Active connections")
    for label, slot in (
        ("① Local stdio", "local"),
        ("② External npx", "external"),
        ("③ Remote HTTP", "remote"),
    ):
        on = st.session_state.get(f"adapter_{slot}") is not None
        st.sidebar.markdown(
            f"- {label}: {'connected' if on else 'disconnected'}"
        )

    if st.sidebar.button("Disconnect all"):
        for s in ("local", "external", "remote"):
            disconnect(s)

    return model, base_url


def main() -> None:
    init_session_state()
    st.title("Multi-Agent Communication via MCP")
    st.caption(
        "Class 07 — same agent calling tools on three different kinds of "
        "MCP server. Watch how only the connection params change."
    )

    model, base_url = sidebar()
    llm = build_llm(model, base_url)

    t1, t2, t3 = st.tabs([
        "① Local stdio (Python)",
        "② External stdio (npx)",
        "③ Remote HTTP (DeepWiki)",
    ])
    with t1:
        tab_local(llm)
    with t2:
        tab_external(llm)
    with t3:
        tab_remote(llm)


if __name__ == "__main__":
    main()
