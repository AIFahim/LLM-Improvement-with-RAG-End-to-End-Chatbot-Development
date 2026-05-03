"""
MCP Server for Class 07 demo — Multi-Agent Communication via Model Context Protocol.

Run standalone (for `mcp dev` or manual poking):
    python mcp_server.py

This server is intended to be launched as a stdio subprocess by run_mcp_demo.py.
The CrewAI Researcher agent calls these tools through the MCP protocol — they
are NOT in-process Python functions from the agent's perspective.
"""

import os
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

REPO_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = REPO_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

mcp = FastMCP("class07-report-tools")


@mcp.tool()
def ping() -> str:
    """Health check — returns a constant string so the caller can confirm
    that the MCP server is reachable and responding to tool calls."""
    return "pong from MCP server"


@mcp.tool()
def list_reports() -> list[str]:
    """List the filenames of all previously generated reports in reports/.
    Returns an empty list if no reports exist yet."""
    return sorted(p.name for p in REPORTS_DIR.iterdir() if p.is_file())


@mcp.tool()
def save_report(title: str, content: str) -> str:
    """Save a report to the reports/ directory.

    Args:
        title: short title used for the filename (will be slugified)
        content: full markdown content of the report

    Returns the relative path of the saved file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)[:40]
    path = REPORTS_DIR / f"{timestamp}_mcp_{slug}.md"
    path.write_text(f"# {title}\n\n{content}\n", encoding="utf-8")
    return str(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    mcp.run(transport="stdio")
