"""
POC: a CrewAI agent calls tools that live on a separate MCP server process.

Run:
    python run_mcp_demo.py

What this proves:
- mcp_server.py is launched as a stdio subprocess
- CrewAI auto-discovers the server's tools via the `mcps=[...]` DSL
- An Agent invokes those tools during crew.kickoff()
- Lifecycle (start/stop of the MCP subprocess) is managed by CrewAI

Pedagogically: contrast this with run_crew.py, where every tool is an
in-process Python object handed to the agent directly. Here the tools
live in a separate process and are accessed via the MCP wire protocol.

This file uses the modern `mcps=[...]` DSL (CrewAI >= 1.14). For the
older MCPServerAdapter pattern see run_mcp_demo_adapter.py.
"""

import os
import sys
from pathlib import Path

from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai.mcp import MCPServerStdio

REPO_ROOT = Path(__file__).resolve().parent
SERVER_SCRIPT = REPO_ROOT / "mcp_server.py"


def build_llm() -> LLM:
    """Local Ollama LLM. Bump via OLLAMA_MODEL env var if tool routing flakes."""
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return LLM(model=f"ollama/{model}", base_url=base_url)


def main() -> None:
    print("=" * 60)
    print("Class 07 POC — CrewAI agent calling MCP server tools (DSL)")
    print("=" * 60)

    llm = build_llm()

    archivist = Agent(
        role="Report Archivist",
        goal="Use MCP tools to inspect existing reports and save new ones.",
        backstory=(
            "You manage a library of reports. You only ever interact with "
            "the report store through the MCP tools provided to you — "
            "never invent filenames or contents from prior knowledge."
        ),
        mcps=[
            MCPServerStdio(
                command=sys.executable,
                args=[str(SERVER_SCRIPT)],
                cache_tools_list=True,
            ),
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    task = Task(
        description=(
            "Step 1: call the `ping` tool and confirm the server responded.\n"
            "Step 2: call the `list_reports` tool to see what's already saved.\n"
            "Step 3: call the `save_report` tool with title 'mcp-poc-dsl' and "
            "content describing in one sentence what MCP gave us. "
            "Return the saved file path."
        ),
        expected_output=(
            "A short confirmation including: the ping response, the count "
            "of existing reports, and the relative path of the new report."
        ),
        agent=archivist,
    )

    crew = Crew(
        agents=[archivist],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
        memory=False,
    )

    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
