"""
MCP client wrapper for CrewAI.

Wraps `MCPServerAdapter` with manual start()/stop() to dodge a known issue
where the `with MCPServerAdapter(...)` context manager closes the asyncio
loop mid-`crew.kickoff()` and breaks subsequent tool calls.

Usage:
    client = MCPCrewClient(server_script="mcp_server.py")
    tools = client.start()             # list[BaseTool] usable by CrewAI agents
    try:
        crew.kickoff()
    finally:
        client.stop()
"""

import sys
from pathlib import Path

from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters


class MCPCrewClient:
    def __init__(self, server_script: str, connect_timeout: int = 60):
        script_path = Path(server_script).resolve()
        if not script_path.exists():
            raise FileNotFoundError(f"MCP server script not found: {script_path}")

        self._params = StdioServerParameters(
            command=sys.executable,
            args=[str(script_path)],
            env=None,
        )
        self._connect_timeout = connect_timeout
        self._adapter: MCPServerAdapter | None = None

    def start(self):
        self._adapter = MCPServerAdapter(self._params, connect_timeout=self._connect_timeout)
        return self._adapter.tools

    def stop(self) -> None:
        if self._adapter is not None:
            self._adapter.stop()
            self._adapter = None
