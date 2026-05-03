"""
MCP Server for Class 07 demo — Multi-Agent Communication via Model Context Protocol.

Run standalone (for `mcp dev` or manual poking):
    python mcp_server.py

This server is intended to be launched as a stdio subprocess by run_mcp_demo.py.
The CrewAI Researcher agent calls these tools through the MCP protocol — they
are NOT in-process Python functions from the agent's perspective.

Tools exposed:
- ping, list_reports, save_report: report-management primitives
- calculator: safe math evaluator (ported from tool-calling-agents branch)
- datetime: current date/time operations (ported from tool-calling-agents branch)
- web_search: DuckDuckGo wrapper (ported from tool-calling-agents branch)

Deliberately NOT exposed: python_repl. Exposing arbitrary Python execution over
MCP would let any connected agent run code on this machine — instructive
non-example for students.
"""

import math
import re
from datetime import datetime as _dt
from pathlib import Path

from langchain_community.tools import DuckDuckGoSearchRun
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
    timestamp = _dt.now().strftime("%Y%m%d_%H%M%S")
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)[:40]
    path = REPORTS_DIR / f"{timestamp}_mcp_{slug}.md"
    path.write_text(f"# {title}\n\n{content}\n", encoding="utf-8")
    return str(path.relative_to(REPO_ROOT))


_CALC_SAFE: dict = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow,
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "log": math.log, "log10": math.log10, "log2": math.log2,
    "exp": math.exp, "floor": math.floor, "ceil": math.ceil,
    "factorial": math.factorial, "gcd": math.gcd,
    "degrees": math.degrees, "radians": math.radians,
    "pi": math.pi, "e": math.e, "tau": math.tau,
}
_CALC_BLOCKED = re.compile(
    r"__|import|exec|eval|open|os\.|sys\.|subprocess|lambda",
    re.IGNORECASE,
)


@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Supports: arithmetic, sqrt, sin/cos/tan/asin/acos/atan, log, log10, log2,
    exp, floor, ceil, factorial, gcd, abs, round, min, max, sum, pow,
    constants pi/e/tau.

    Examples: '2 + 2', 'sqrt(16)', 'sin(pi/2)', 'log(100, 10)'.
    """
    expr = expression.strip()
    if _CALC_BLOCKED.search(expr):
        return "Error: expression rejected by safety filter"
    try:
        return str(eval(expr, {"__builtins__": {}}, _CALC_SAFE))
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        return f"Error: could not evaluate expression — {e}"


@mcp.tool()
def datetime_now(operation: str = "now") -> str:
    """Get current date/time in various forms.

    operation:
      - 'now'       -> 'YYYY-MM-DD HH:MM:SS'
      - 'date'      -> 'YYYY-MM-DD'
      - 'time'      -> 'HH:MM:SS'
      - 'weekday'   -> e.g. 'Sunday'
      - 'timestamp' -> Unix timestamp (int as string)
      - 'format:<strftime>'  -> custom strftime format
    """
    now = _dt.now()
    op = operation.strip().lower()
    if op == "now":
        return now.strftime("%Y-%m-%d %H:%M:%S")
    if op == "date":
        return now.strftime("%Y-%m-%d")
    if op == "time":
        return now.strftime("%H:%M:%S")
    if op == "weekday":
        return now.strftime("%A")
    if op == "timestamp":
        return str(int(now.timestamp()))
    if op.startswith("format:"):
        return now.strftime(operation[7:])
    return (
        f"Unknown operation: {operation}. "
        f"Use now, date, time, weekday, timestamp, or format:<strftime>."
    )


_DDG = DuckDuckGoSearchRun()


@mcp.tool()
def web_search(query: str) -> str:
    """Search the public web via DuckDuckGo. No API key required.

    Returns a plain-text snippet of results. Use this when the user asks
    about recent or external information you don't already know.
    """
    try:
        return _DDG.run(query)
    except Exception as e:
        return f"Error: web search failed — {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
