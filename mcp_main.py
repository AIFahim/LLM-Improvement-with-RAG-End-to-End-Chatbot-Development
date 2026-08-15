"""
MCP Chatbot — orchestration engine.

This is where the Agent / Task / Crew machinery is wired together. The
Streamlit UI in mcp_app.py only ever talks to this module, never to
CrewAI directly.

Memory: we use a *rolling summary* that gets folded forward after each
turn, NOT CrewAI's `memory=True`. Why not memory=True? CrewAI 1.14's
memory layer requires Chroma vector storage, which on this stack fails
to initialize without OPENAI_API_KEY even when EMBEDDINGS_OLLAMA_* env
vars are set. The rolling summary works without any embedder, keeps
prompts bounded regardless of conversation length, and is fully
visible to students.
"""

import io
import re
from contextlib import redirect_stdout, redirect_stderr
from typing import Any

from crewai import Crew, Process
from crewai.llm import LLM
from crewai_tools import MCPServerAdapter

from mcp_agents import AgentFactory
from mcp_tasks import (
    create_chat_task,
    create_recap_task,
    create_running_summary_task,
)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Strip ANSI color codes from CrewAI's verbose output for display."""
    return _ANSI_RE.sub("", text)


def build_llm(model: str, base_url: str) -> LLM:
    """Construct an Ollama-backed CrewAI LLM.

    Note on context size: CrewAI does NOT accept num_ctx here — it
    forwards unknown kwargs into the OpenAI-compatible completion call,
    which rejects them. Set the window on the Ollama server instead:

        OLLAMA_CONTEXT_LENGTH=8192 ollama serve

    This matters when a tool returns a lot of text (a DeepWiki page runs
    to thousands of tokens). Past the window the result is truncated
    before the model sees it, and the agent replies "I need more
    information on how to proceed" rather than raising an error.
    """
    return LLM(model=f"ollama/{model}", base_url=base_url)


class MCPChatbot:
    """One chatbot session against one MCP server.

    Lifecycle:
        chatbot = MCPChatbot()
        chatbot.connect(server_params)        # spawn / dial the server
        reply, trace = chatbot.chat_turn(msg, llm)
        ...
        chatbot.disconnect()                  # shut it down

    Each call to chat_turn() is one full agent run:
      1. Build the chat Agent + Task with the running summary as context
      2. Crew.kickoff() — LLM picks tools, MCPServerAdapter routes calls
      3. Roll the conversation summary forward (a separate no-tools Crew)
    """

    def __init__(self, *, connect_timeout: int = 120, verbose: bool = True):
        self._connect_timeout = connect_timeout
        self._verbose = verbose
        self._adapter: MCPServerAdapter | None = None
        self.tools: list = []
        self.running_summary: str = ""

    # -- connection lifecycle ----------------------------------------------

    def connect(self, server_params: Any) -> list:
        """Connect to an MCP server and discover its tools."""
        self.disconnect()
        self._adapter = MCPServerAdapter(
            server_params, connect_timeout=self._connect_timeout
        )
        self.tools = list(self._adapter.tools)
        self.running_summary = ""
        return self.tools

    def disconnect(self) -> None:
        if self._adapter is not None:
            try:
                self._adapter.stop()
            except Exception:
                pass
        self._adapter = None
        self.tools = []

    @property
    def is_connected(self) -> bool:
        return self._adapter is not None

    # -- per-turn chat -----------------------------------------------------

    def chat_turn(self, user_message: str, llm: LLM) -> tuple[str, str]:
        """Run one chat turn. Returns (reply, captured_trace).

        The trace is the verbose stdout from crew.kickoff() — the raw
        MCP tool calls + outputs. Showing it lets students see what the
        server actually returned vs how the LLM summarized it.
        """
        factory = AgentFactory(llm, verbose=self._verbose)
        agent = factory.create_chatbot(self.tools)
        task = create_chat_task(agent, user_message, self.running_summary)
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=self._verbose,
            memory=False,
        )

        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            result = crew.kickoff()

        reply = str(result)
        trace = strip_ansi(buf.getvalue())
        self._roll_summary_forward(user_message, reply, llm)
        return reply, trace

    def _roll_summary_forward(
        self, user_msg: str, assistant_reply: str, llm: LLM
    ) -> None:
        """Fold the new exchange into self.running_summary.

        Falls back to literal concatenation if the summarization LLM call
        fails — we don't want a flaky summary call to break the chat.
        """
        try:
            factory = AgentFactory(llm, verbose=False)
            agent = factory.create_summarizer()
            task = create_running_summary_task(
                agent, self.running_summary, user_msg, assistant_reply
            )
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
                memory=False,
            )
            self.running_summary = str(crew.kickoff()).strip()
        except Exception:
            prev = self.running_summary
            self.running_summary = (
                (prev + "\n" if prev else "")
                + f"User: {user_msg}\nAssistant: {assistant_reply}"
            )

    # -- on-demand recap ---------------------------------------------------

    def recap(self, messages: list[dict], llm: LLM) -> str:
        """One-shot recap of all messages so far (sidebar button)."""
        if not messages:
            return "No conversation to summarize yet."
        transcript = "\n".join(
            f"{m['role'].title()}: {m['content']}" for m in messages
        )
        factory = AgentFactory(llm, verbose=False)
        agent = factory.create_recap_writer()
        task = create_recap_task(agent, transcript)
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
            memory=False,
        )
        return str(crew.kickoff())

    def reset_chat(self) -> None:
        """Clear the running summary (call when the user clears the chat)."""
        self.running_summary = ""
