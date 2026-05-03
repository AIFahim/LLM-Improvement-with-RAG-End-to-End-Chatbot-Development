"""
POC: a CrewAI agent calls tools that live on a separate MCP server process.

Run:
    python run_mcp_demo.py

What this proves:
- mcp_server.py is launched as a stdio subprocess
- CrewAI's MCPServerAdapter discovers its tools
- An Agent invokes those tools during crew.kickoff()
- Tools keep their short, original names (ping, list_reports, save_report) —
  important for small local models that struggle with long tool names.

Pedagogically: contrast this with run_crew.py, where every tool is an
in-process Python object handed to the agent directly. Here the tools
live in a separate process and are accessed via the MCP wire protocol.

This file uses the `MCPServerAdapter` route via mcp_client.MCPCrewClient.
For the modern DSL approach (`mcps=[MCPServerStdio(...)]` directly on the
Agent), see run_mcp_demo_dsl.py. Trade-off: the DSL is cleaner code but
auto-prefixes tool names with the server's command path, which confuses
small Ollama models like qwen2.5:3b.
"""

import os
from pathlib import Path

from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM

from mcp_client import MCPCrewClient

REPO_ROOT = Path(__file__).resolve().parent
SERVER_SCRIPT = REPO_ROOT / "mcp_server.py"


def build_llm() -> LLM:
    """Local Ollama LLM. Bump via OLLAMA_MODEL env var if tool routing flakes."""
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return LLM(model=f"ollama/{model}", base_url=base_url)


def main() -> None:
    print("=" * 60)
    print("Class 07 POC — CrewAI agent calling MCP server tools")
    print("=" * 60)

    client = MCPCrewClient(server_script=str(SERVER_SCRIPT))
    mcp_tools = client.start()

    try:
        print(f"\n[OK] MCP server connected. Tools discovered: "
              f"{[t.name for t in mcp_tools]}\n")

        llm = build_llm()

        archivist = Agent(
            role="Report Archivist",
            goal="Use MCP tools to inspect existing reports and save new ones.",
            backstory=(
                "You manage a library of reports. You only ever interact with "
                "the report store through the MCP tools provided to you — "
                "never invent filenames or contents from prior knowledge."
            ),
            tools=mcp_tools,
            llm=llm,
            verbose=True,
            allow_delegation=False,
        )

        task = Task(
            description=(
                "Step 1: call the `ping` tool and confirm the server responded.\n"
                "Step 2: call the `list_reports` tool to see what's already saved.\n"
                "Step 3: call the `save_report` tool with title 'mcp-poc' and "
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
    finally:
        client.stop()
        print("\n[OK] MCP server stopped.")


if __name__ == "__main__":
    main()
