"""
POC: a CrewAI agent calls tools from an EXTERNAL MCP server it does not own.

Run:
    python run_mcp_demo_external.py

What this proves:
- The MCP server is `@modelcontextprotocol/server-filesystem`, published by
  Anthropic on npm — none of the tool code lives in this repo.
- The server is launched on demand via `npx -y` (no global install needed).
- CrewAI's MCPServerAdapter discovers the server's tools (read_file,
  write_file, list_directory, etc.) and hands them to a CrewAI Agent.
- The agent uses them like any other tool — it has no idea they live in a
  separate Node.js process authored by a different team.

Pedagogical pitch for students:
- run_mcp_demo.py     -> calls OUR mcp_server.py (Python, in-repo)
- run_mcp_demo_external.py -> calls Anthropic's filesystem server (npm)
  Same protocol. Same agent code shape. Different tool author. That's MCP's
  whole value proposition: the agent doesn't care where the tools come from.

Sandbox:
The filesystem server is restricted to ./mcp_sandbox/ so the agent cannot
touch the rest of the repo. This is how MCP filesystem servers do
authorization — the allowed root is a CLI argument.
"""

import os
from pathlib import Path

from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters

REPO_ROOT = Path(__file__).resolve().parent
SANDBOX = REPO_ROOT / "mcp_sandbox"


def build_llm() -> LLM:
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return LLM(model=f"ollama/{model}", base_url=base_url)


def main() -> None:
    print("=" * 60)
    print("Class 07 POC — CrewAI agent calling EXTERNAL MCP server")
    print("(Anthropic's @modelcontextprotocol/server-filesystem via npx)")
    print("=" * 60)

    SANDBOX.mkdir(exist_ok=True)

    # External server: Anthropic's filesystem MCP server, run via npx.
    # `-y` auto-confirms install on first run; subsequent runs use the cache.
    server_params = StdioServerParameters(
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            str(SANDBOX),
        ],
        env=None,
    )

    adapter = MCPServerAdapter(server_params, connect_timeout=120)
    try:
        tools = adapter.tools
        print(f"\n[OK] External MCP server connected. "
              f"Tools discovered: {[t.name for t in tools]}\n")

        llm = build_llm()

        librarian = Agent(
            role="Sandbox Librarian",
            goal=(
                "Use the filesystem MCP tools to inventory the sandbox "
                "directory and add a small note about today's date."
            ),
            backstory=(
                "You manage the contents of a small sandboxed directory. "
                "You only ever interact with the directory through MCP "
                "tools — you never invent filenames or contents."
            ),
            tools=tools,
            llm=llm,
            verbose=True,
            allow_delegation=False,
        )

        sandbox_path = str(SANDBOX)
        task = Task(
            description=(
                f"Step 1: call `list_directory` with path '{sandbox_path}' to "
                f"see what's currently in the sandbox.\n"
                f"Step 2: call `write_file` to create '{sandbox_path}/note.md' "
                f"with content '# Note\\n\\nWritten by a CrewAI agent through "
                f"an external MCP server.'\n"
                f"Step 3: call `read_file` on '{sandbox_path}/note.md' to "
                f"confirm the write succeeded.\n"
                f"Return: the list of files before, the write confirmation, "
                f"and the file content you read back."
            ),
            expected_output=(
                "A short report: (a) original directory contents, "
                "(b) confirmation that note.md was written, "
                "(c) the content of note.md as read back."
            ),
            agent=librarian,
        )

        crew = Crew(
            agents=[librarian],
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
        adapter.stop()
        print("\n[OK] External MCP server stopped.")


if __name__ == "__main__":
    main()
