# Multi-Agent Systems with CrewAI + MCP

This branch implements **Class 07** in two halves:

1. **Multi-agent orchestration** — CrewAI's Planner/Researcher/Writer/Critic/Summarizer pattern for collaborative report writing.
2. **Multi-agent communication via MCP** — agents calling tools that live in *separate processes* (local Python, npm, or remote SaaS) over the Model Context Protocol.

## Features

### Orchestration half
- **Multi-Role Agents**: Planner, Researcher, Writer, Critic, Summarizer
- **Planner-Executor-Critic Model**: Structured workflow for quality output
- **CrewAI Integration**: Agent orchestration and task management
- **Multiple Workflows**: Full report, quick report, research-only
- **Streamlit UI**: Interactive web interface for report generation
- **CLI Support**: Generate reports from command line

### MCP half
- **FastMCP server** with 6 tools: ping, list_reports, save_report, calculator, datetime_now, web_search
- **Three demo deployment shapes** so students see the same agent code calling tools across very different worlds:
  - Local Python stdio (our own server)
  - External Node.js stdio (Anthropic's published filesystem server, via `npx`)
  - Remote Streamable HTTP (DeepWiki's hosted server, no auth)
- **Chatbot UI** with rolling conversation summary and live MCP protocol trace per turn

## Project Structure

```
.
# CrewAI orchestration
├── crew_agents.py            # Multi-role agent definitions
├── crew_tasks.py             # Task definitions for workflows
├── crew_main.py              # CrewAI orchestration
├── crew_app.py               # Streamlit UI for report writing
├── run_crew.py               # CLI launcher

# MCP server + clients
├── mcp_server.py             # FastMCP server (6 tools, stdio)
├── mcp_client.py             # MCPServerAdapter lifecycle wrapper
├── run_mcp_demo.py           # Local stdio demo (terminal)
├── run_mcp_demo_external.py  # External npm filesystem-server demo
├── run_mcp_demo_remote.py    # Remote DeepWiki HTTP demo
├── run_mcp_demo_dsl.py       # Same as run_mcp_demo.py via mcps=[] DSL
├── mcp_app.py                # Streamlit MCP chatbot UI
├── mcp_sandbox/              # Sandbox dir for the external filesystem demo

# Carryover from earlier classes
├── agent.py                  # LangGraph ReAct agent (Class 06)
├── tools.py                  # In-process LangChain tools (Class 06)
├── memory_manager.py         # Memory types
├── app.py                    # RAG chatbot UI (Class 05)
├── chatbot.py                # RAG orchestrator (Class 05)

# Generated / data
├── reports/                  # Generated reports output
├── pdfFiles/                 # PDFs for the RAG demo
└── requirements.txt          # Dependencies
```

## New Modules (Class 07)

### 1. crew_agents.py

Defines multi-role agents for collaborative work:

| Agent | Role | Capabilities |
|-------|------|--------------|
| Planner | Report Planner | Creates outlines, identifies research areas |
| Researcher | Research Analyst | Gathers facts, statistics, evidence |
| Writer | Content Writer | Composes clear, engaging content |
| Critic | Quality Reviewer | Reviews and provides feedback |
| Summarizer | Executive Summarizer | Creates concise summaries |

```python
from crew_agents import AgentFactory

factory = AgentFactory(verbose=True)
agents = factory.create_all_agents()

# Or create specific agents
planner = factory.create_planner()
researcher = factory.create_researcher()
```

### 2. crew_tasks.py

Defines tasks for the report writing workflow:

```python
from crew_tasks import TaskFactory, create_planning_task

# Use factory with agents
factory = TaskFactory(agents)
tasks = factory.create_report_workflow(
    topic="AI in Healthcare",
    style="professional",
    word_count=1500,
)
```

### 3. crew_main.py

Orchestrates multi-agent collaboration:

```python
from crew_main import create_report_crew

# Create and run crew
crew = create_report_crew(crew_type="report_writing")
result = crew.create_report(
    topic="The Future of Renewable Energy",
    requirements="Focus on solar and wind technologies",
    style="professional",
    workflow="full_report",
)

print(result["result"])  # The generated report
```

## Installation

```bash
# Clone and checkout branch
git clone https://github.com/AIFahim/LLM-Improvement-with-RAG-End-to-End-Chatbot-Development.git
cd LLM-Improvement-with-RAG-End-to-End-Chatbot-Development
git checkout class-07-multi-agent-crewai

# Create environment
conda create -n crewai-agents python=3.11
conda activate crewai-agents

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Start Ollama

```bash
# Using Docker
docker start ollama

# Or native
ollama serve
```

### 2. Run a Streamlit UI

Two UIs ship with this branch — pick whichever matches the lesson.

```bash
# A) Multi-agent report writer (orchestration half of Class 07)
streamlit run crew_app.py

# B) MCP chatbot (communication half of Class 07)
streamlit run mcp_app.py
```

### 3. Generate a report via CLI

```bash
# Full report
python run_crew.py --topic "AI in Healthcare" --workflow full_report

# Quick report
python run_crew.py --topic "Machine Learning Basics" --workflow quick_report --style academic

# List options
python run_crew.py --list
```

### 4. Run an MCP demo from the terminal

```bash
python run_mcp_demo.py            # local Python MCP server (stdio)
python run_mcp_demo_remote.py     # DeepWiki SaaS MCP server (Streamable HTTP)
python run_mcp_demo_external.py   # Anthropic's npm filesystem server (npx, requires Node)
python run_mcp_demo_dsl.py        # local server but via the modern crewai.mcp DSL
```

## Workflows

### Full Report Workflow
```
Planner -> Researcher -> Writer -> Critic -> Summarizer
```

### Quick Report Workflow
```
Researcher -> Writer
```

### Research Plan Workflow
```
Planner -> Researcher
```

### Write & Review Workflow
```
Writer -> Critic
```

## Crew Types

| Crew Type | Agents | Description |
|-----------|--------|-------------|
| `report_writing` | All 5 agents | Full team for comprehensive reports |
| `quick_report` | Researcher, Writer | Minimal team for quick output |
| `research_only` | Planner, Researcher | Focus on research and planning |
| `review_team` | Writer, Critic | Writing with quality review |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Streamlit UI (crew_app.py)                │
│  ┌─────────────────┐  ┌────────────────────────────────┐   │
│  │ Configuration   │  │      Report Generation         │   │
│  │ - Crew Type     │  │ - Topic Input                  │   │
│  │ - Workflow      │  │ - Progress Display             │   │
│  │ - Style         │  │ - Report Output                │   │
│  └────────┬────────┘  └───────────────┬────────────────┘   │
└───────────┼───────────────────────────┼────────────────────┘
            │                           │
            └───────────────┬───────────┘
                            │
            ┌───────────────▼───────────────┐
            │   ReportWritingCrew           │
            │   (crew_main.py)              │
            └───────────────┬───────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼───────┐  ┌────────▼────────┐  ┌──────▼──────┐
│ AgentFactory  │  │  TaskFactory    │  │ CrewAI Crew │
│ (crew_agents) │  │  (crew_tasks)   │  │  Process    │
└───────┬───────┘  └────────┬────────┘  └──────┬──────┘
        │                   │                   │
┌───────▼───────────────────▼───────────────────▼───────┐
│                    Agents & Tasks                      │
│  ┌─────────┐ ┌──────────┐ ┌────────┐ ┌────────┐      │
│  │ Planner │→│Researcher│→│ Writer │→│ Critic │      │
│  └─────────┘ └──────────┘ └────────┘ └────────┘      │
│                                              ↓        │
│                                      ┌───────────┐   │
│                                      │Summarizer │   │
│                                      └───────────┘   │
└───────────────────────────────────────────────────────┘
```

## Planner-Executor-Critic Model

```
┌──────────────────────────────────────────────────────────┐
│                      PLANNING PHASE                       │
│  ┌─────────┐                                             │
│  │ Planner │ → Creates outline, identifies research      │
│  └────┬────┘   areas, sets structure                     │
│       │                                                   │
└───────┼──────────────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────────────────────┐
│                     EXECUTION PHASE                       │
│  ┌──────────┐                                            │
│  │Researcher│ → Gathers facts, statistics, evidence      │
│  └────┬─────┘                                            │
│       ↓                                                   │
│  ┌────────┐                                              │
│  │ Writer │ → Composes content following outline         │
│  └────┬───┘                                              │
│       │                                                   │
└───────┼──────────────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────────────────────┐
│                      CRITIQUE PHASE                       │
│  ┌────────┐                                              │
│  │ Critic │ → Reviews quality, accuracy, completeness    │
│  └────┬───┘                                              │
│       ↓                                                   │
│  ┌───────────┐                                           │
│  │Summarizer │ → Creates executive summary               │
│  └───────────┘                                           │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

## API Reference

### ReportWritingCrew

```python
crew = ReportWritingCrew(
    verbose=True,           # Log agent actions
    process="sequential",   # or "hierarchical"
    memory=True,            # Enable agent memory
    output_dir="reports",   # Output directory
)

crew.setup_crew(crew_type="report_writing")

result = crew.create_report(
    topic="Report Topic",
    requirements="Optional requirements",
    style="professional",
    word_count=1500,
    workflow="full_report",
)
```

### AgentFactory

```python
factory = AgentFactory(verbose=True)

# Create all agents
agents = factory.create_all_agents()

# Create individual agents
planner = factory.create_planner()
researcher = factory.create_researcher()
writer = factory.create_writer()
critic = factory.create_critic()
summarizer = factory.create_summarizer()
```

### TaskFactory

```python
factory = TaskFactory(agents)

# Full workflow
tasks = factory.create_report_workflow(topic, requirements, style, word_count)

# Quick workflow
tasks = factory.create_quick_report_workflow(topic, style)

# Research workflow
tasks = factory.create_research_workflow(topic, research_areas)
```

## Configuration

Edit `config.py` for settings:

```python
# LLM Provider
LLM_PROVIDER = "ollama"  # or "azure"
OLLAMA_MODEL = "qwen2.5:1.5b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Azure (if using)
AZURE_OPENAI_API_KEY = "your-key"
AZURE_OPENAI_ENDPOINT = "your-endpoint"
AZURE_DEPLOYMENT_NAME = "your-deployment"
```

## Example Output

```
python run_crew.py --topic "Impact of AI on Healthcare" --workflow full_report

============================================================
Multi-Agent Report Writer - CLI Mode
============================================================

Topic: Impact of AI on Healthcare
Workflow: full_report
Crew: report_writing
Style: professional

Initializing crew...
Crew ready with agents: planner, researcher, writer, critic, summarizer

Generating report... This may take a few minutes.

============================================================
Report Generated Successfully!
============================================================

Execution Time: 45.23 seconds
Tasks Completed: 5
Agents Used: planner, researcher, writer, critic, summarizer

--- REPORT ---

# Impact of AI on Healthcare

## Executive Summary
...

## Introduction
...

## Key Findings
...

## Conclusion
...

--- END REPORT ---

Report saved to: reports/
```

## Multi-Agent Communication via MCP

The `MultiAgentOrchestrator.send_message` method in `crew_main.py` is an
**in-process message queue** — useful for crew↔crew handoffs but not actual
MCP. The real MCP integration lives in the `mcp_*.py` files described below.

### What MCP buys you
The Model Context Protocol (Anthropic, Nov 2024) is a JSON-RPC standard for
*agent ↔ tool server*. With CrewAI's `MCPServerAdapter`, an agent can use
tools that live in a separate process — possibly written in a different
language, possibly running on a different machine — without changing the
agent code at all. The protocol abstracts away the tool's deployment shape.

### Three demo scripts, one teaching arc

```bash
# 1. Local stdio: agent calls our own Python server
python run_mcp_demo.py

# 2. External stdio: agent calls Anthropic's published filesystem server
#    (npx fetches the npm package on first run)
python run_mcp_demo_external.py

# 3. Remote HTTP: agent calls DeepWiki's hosted MCP server (no auth)
python run_mcp_demo_remote.py
```

All three use the same Agent / Task / Crew shape. **Only the
`server_params` differ.** Side-by-side these three files are the punchline
of the whole MCP story.

### The chatbot UI

```bash
streamlit run mcp_app.py
# opens at http://localhost:8501
```

What students see:
- **Sidebar**: pick a server (Local Python / Remote DeepWiki), Connect,
  inspect discovered tools, choose Ollama model, Summarize / Clear chat.
- **Main area**: standard chat input. Each assistant reply expands a
  *MCP protocol trace* showing the raw tool calls + outputs for that turn.
- **Switching servers** clears the chat (different tools = fresh context).
- **Memory**: a rolling summary of the conversation is folded forward
  after each turn and passed back as context (LangChain
  `ConversationSummaryMemory` shape, kept transparent in code).

### The MCP server

`mcp_server.py` exposes 6 tools via FastMCP:

| Tool | Purpose |
|---|---|
| `ping` | Health check (returns `"pong from MCP server"`) |
| `list_reports` | List filenames in `reports/` |
| `save_report(title, content)` | Write a markdown report to `reports/` |
| `calculator(expression)` | Safe math evaluator (`sqrt(16)`, `sin(pi/2)`, etc.) |
| `datetime_now(operation)` | now / date / time / weekday / timestamp / format:`<strftime>` |
| `web_search(query)` | DuckDuckGo wrapper, no API key |

Run it standalone for poking with `mcp dev` or any MCP client:
```bash
python mcp_server.py    # blocks, talks JSON-RPC over stdio
```

### Pitfalls worth teaching

- **`crewai-tools` < 1.14.4** has a bug where `MCPServerAdapter` prompts to
  install `mcp` even when it's already installed. Pin `crewai-tools[mcp] >= 1.14.4`.
- **DSL tool-name prefixing**: `Agent(..., mcps=[MCPServerStdio(...)])` is
  the modern pattern, but it auto-prefixes tool names with the server's
  command path (e.g. `home_aifahim_miniconda3_bin_python_..._29c0b316`).
  That breaks small models like `qwen2.5:3b`. We use `MCPServerAdapter`
  for clean tool names. See `run_mcp_demo_dsl.py` for the comparison.
- **CrewAI `memory=True` is broken with Ollama** on this stack — Chroma
  still demands `OPENAI_API_KEY` even with `EMBEDDINGS_OLLAMA_*` env vars
  set. The chatbot uses a manual rolling summary instead. (Same reason
  commit `f0a8ce7` disabled it.)
- **Don't expose `python_repl` over MCP**. Arbitrary code execution
  through a public protocol = remote code execution for any connected
  agent. Worth showing students as a *non*-example.
- **Small-model tool-eagerness**: `qwen2.5:3b` will fire 3+ tool calls on
  `"Hi"` unless the agent's backstory explicitly tells it *when not to
  use tools*. See `mcp_app.py:chat_turn`'s backstory string.

## Troubleshooting

### CrewAI Import Error
```bash
pip install crewai crewai-tools --upgrade
```

### MCP: "You are missing the 'mcp' package" prompt on startup
You're on `crewai-tools < 1.14.4`. Upgrade:
```bash
pip install -U "crewai-tools[mcp]>=1.14.4"
```

### MCP: Memory init errors mentioning `CHROMA_OPENAI_API_KEY`
CrewAI 1.14's `memory=True` doesn't work with Ollama embedders on this
stack. Set `memory=False` and use the manual rolling-summary approach
(see `mcp_app.py:chat_turn`).

### MCP: Agent fires tools on "Hi" / over-uses tools
Small models default to "I have tools, therefore I should demo them."
Strengthen the agent's backstory to spell out when *not* to call tools.
Or bump to `qwen2.5:7b-instruct` for more discerning routing.

### Ollama Connection Error
```bash
docker ps | grep ollama
docker start ollama
```

### Reports / vectorDB Issues
```bash
rm -rf reports/
# vectorDB corruption (chromadb version mismatch):
mv vectorDB vectorDB.bak-$(date +%Y%m%d)
```

## Related Classes

- **Class 06**: LangChain Deep Dive (Memory, Tools, Agents)
- **Class 05**: RAG Chatbot (Document Q&A)

## License

MIT License

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request
