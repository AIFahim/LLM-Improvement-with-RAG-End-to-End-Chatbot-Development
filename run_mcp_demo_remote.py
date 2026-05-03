"""
POC: a CrewAI agent calls a REMOTE, public MCP server over Streamable HTTP.

Run:
    python run_mcp_demo_remote.py

What this proves:
- The MCP server is hosted by a third party (DeepWiki by Devin/Cognition).
  No code, no subprocess — just a URL.
- Transport is Streamable HTTP (the modern MCP transport — replaces SSE).
- CrewAI's MCPServerAdapter discovers the server's tools and hands them to
  a CrewAI Agent.
- The agent uses them to answer a question about a real GitHub repo.

Pedagogical pitch:
- run_mcp_demo.py        -> calls OUR Python MCP server (stdio, in-repo)
- run_mcp_demo_remote.py -> calls a SaaS-hosted MCP server (HTTP)

Same protocol, two different deployment shapes. The agent code looks
identical in both. That's the whole point of MCP — the tool author,
language, and deployment are abstracted away.

DeepWiki MCP details:
- Endpoint: https://mcp.deepwiki.com/mcp
- Free, no API key required
- Tools: read_wiki_structure, read_wiki_contents, ask_question
- Indexes GitHub repos and exposes a Q&A interface over their wikis
"""

import os

from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai_tools import MCPServerAdapter

DEEPWIKI_URL = "https://mcp.deepwiki.com/mcp"


def build_llm() -> LLM:
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return LLM(model=f"ollama/{model}", base_url=base_url)


def main() -> None:
    print("=" * 60)
    print("Class 07 POC — CrewAI agent calling a REMOTE MCP server")
    print(f"(DeepWiki @ {DEEPWIKI_URL}, Streamable HTTP, no auth)")
    print("=" * 60)

    server_params = {
        "url": DEEPWIKI_URL,
        "transport": "streamable-http",
    }

    adapter = MCPServerAdapter(server_params, connect_timeout=60)
    try:
        tools = adapter.tools
        print(f"\n[OK] Remote MCP server connected. "
              f"Tools discovered: {[t.name for t in tools]}\n")

        llm = build_llm()

        researcher = Agent(
            role="Open-Source Researcher",
            goal=(
                "Use DeepWiki's MCP tools to answer questions about real "
                "GitHub repositories. Always cite which tool you called."
            ),
            backstory=(
                "You research open-source projects by querying DeepWiki, "
                "which indexes GitHub repos and exposes a Q&A interface "
                "over the MCP protocol. You never invent answers from "
                "prior knowledge — every claim is grounded in a tool call."
            ),
            tools=tools,
            llm=llm,
            verbose=True,
            allow_delegation=False,
        )

        task = Task(
            description=(
                "Use the `ask_question` tool with repoName='crewAIInc/crewAI' "
                "and question='What is the main purpose of this project, "
                "and what are its core abstractions?' "
                "Then summarize the answer in 2-3 sentences for a student."
            ),
            expected_output=(
                "A 2-3 sentence summary of CrewAI's purpose and core "
                "abstractions, sourced from the DeepWiki MCP server."
            ),
            agent=researcher,
        )

        crew = Crew(
            agents=[researcher],
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
        print("\n[OK] Remote MCP adapter stopped.")


if __name__ == "__main__":
    main()
