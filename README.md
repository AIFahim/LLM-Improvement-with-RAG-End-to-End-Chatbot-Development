# MCP Chatbot — CrewAI + Model Context Protocol

A conversational chatbot where the **CrewAI agent gets its tools from
an MCP server**. The same agent code runs against very different
servers (a local Python subprocess, a remote SaaS endpoint) — only the
connection params change. That is the Model Context Protocol value
proposition in one demo.

This branch is the **Class 07** material on multi-agent communication
via MCP.

## Features

- **FastMCP server** with 6 tools: `ping`, `list_reports`,
  `save_report`, `calculator`, `datetime_now`, `web_search`.
- **CrewAI integration** via `MCPServerAdapter` — tools discovered at
  connect time, called over JSON-RPC during `crew.kickoff()`.
- **Two server shapes** in the chatbot UI:
  - Local Python stdio (our own `mcp_server.py` subprocess)
  - Remote Streamable HTTP (DeepWiki's hosted server, no auth)
- **Streamlit chatbot** with rolling conversation summary and a live
  per-turn MCP protocol trace expander.
- **Three terminal demos** (`run_mcp_demo*.py`) showing the same agent
  code calling local, external (npm), remote, or DSL-style servers.

## Project structure

```
.
# Modular MCP chatbot
├── mcp_agents.py             # Agent definitions (chatbot, summarizer, recap)
├── mcp_tasks.py              # Task templates (chat, rolling summary, recap)
├── mcp_main.py                # MCPChatbot orchestration class
├── mcp_app.py                 # Streamlit UI (thin presentation layer)

# MCP server + client
├── mcp_server.py              # FastMCP server (6 tools, stdio)
├── mcp_client.py              # MCPServerAdapter lifecycle wrapper

# Terminal demos (one shape per file)
├── run_mcp_demo.py            # Local stdio server (mcp_server.py)
├── run_mcp_demo_remote.py     # Remote DeepWiki SaaS over Streamable HTTP
├── run_mcp_demo_external.py   # Anthropic's @modelcontextprotocol/server-filesystem via npx
├── run_mcp_demo_dsl.py        # Local server using the modern crewai.mcp DSL

# Generated / data
├── reports/                   # Read/written by the report tools
├── mcp_sandbox/               # Sandbox dir for the external filesystem demo
├── mcp_diagrams/              # PNG diagrams used in the slide deck

# Misc
├── requirements.txt
├── setup_ollama.sh            # One-time Ollama bootstrap
└── .env.example
```

## Installation

```bash
git clone https://github.com/AIFahim/LLM-Improvement-with-RAG-End-to-End-Chatbot-Development.git
cd LLM-Improvement-with-RAG-End-to-End-Chatbot-Development
git checkout class-07-multi-agent-crewai

conda create -n mcp-chatbot python=3.11 -y
conda activate mcp-chatbot
pip install -r requirements.txt
```

Pull a small Ollama model so you don't burn resources:

```bash
ollama pull qwen2.5:3b
ollama serve   # if not already running
```

## Run the chatbot

```bash
streamlit run mcp_app.py
# opens at http://localhost:8501
```

In the sidebar:

1. Pick a server (Local Python or Remote DeepWiki).
2. Click **Connect**.
3. Type a message in the chat box.
4. Expand **MCP protocol trace** under any assistant reply to see the
   raw tool calls + outputs that crossed the wire.
5. Use **Summarize** for a recap, **Clear** to wipe the chat, or
   switch the server (which auto-clears).

## Run a one-shot terminal demo

```bash
python run_mcp_demo.py            # local Python MCP server
python run_mcp_demo_remote.py     # DeepWiki HTTP MCP server (no auth)
python run_mcp_demo_external.py   # Anthropic's npm filesystem server (requires Node)
python run_mcp_demo_dsl.py        # same as run_mcp_demo.py via mcps=[] DSL
```

Diff any two of these to see what changes between deployment shapes —
spoiler: only the `server_params`.

## The MCP server (`mcp_server.py`)

FastMCP server exposing 6 tools via the `@mcp.tool()` decorator:

| Tool | Purpose |
|---|---|
| `ping` | Health check (returns `"pong from MCP server"`) |
| `list_reports` | List filenames in `reports/` |
| `save_report(title, content)` | Write a markdown report to `reports/` |
| `calculator(expression)` | Safe math evaluator (`sqrt(16)`, `sin(pi/2)`) |
| `datetime_now(operation)` | now / date / time / weekday / timestamp / format:`<strftime>` |
| `web_search(query)` | DuckDuckGo wrapper, no API key |

Run standalone for poking with `mcp dev` or any MCP client:

```bash
python mcp_server.py    # blocks, talks JSON-RPC over stdio
```

Deliberately NOT exposed: `python_repl`. Exposing arbitrary code
execution as a tool would let any connected agent run code on your
host. Worth showing students as a non-example.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                  Streamlit UI (mcp_app.py)                     │
│  ┌──────────────┐                  ┌──────────────────────┐   │
│  │ Sidebar      │                  │ Chat surface         │   │
│  │ - server     │                  │ - st.chat_message    │   │
│  │ - model      │                  │ - protocol trace     │   │
│  │ - summarize  │                  │ - recap expander     │   │
│  └──────┬───────┘                  └──────────┬───────────┘   │
└─────────┼──────────────────────────────────────┼──────────────┘
          │                                      │
          │   ┌──────────────────────────────────┴───┐
          └──►│       MCPChatbot (mcp_main.py)        │
              │  - connect / disconnect               │
              │  - chat_turn                          │
              │  - rolling summary                    │
              │  - on-demand recap                    │
              └──┬─────────────────┬─────────────────┘
                 │                 │
        ┌────────▼─────┐   ┌───────▼─────────┐
        │ AgentFactory │   │  Task templates │
        │ (mcp_agents) │   │  (mcp_tasks)    │
        └──────────────┘   └─────────────────┘
                 │
                 ▼
        ┌─────────────────────────────────────────┐
        │   MCPServerAdapter (crewai_tools)       │
        │   discovery + JSON-RPC tool calls       │
        └─────────────┬───────────────────────────┘
                      │
        ┌─────────────┴───────────────────────────┐
        │                                         │
   ┌────▼─────────────┐          ┌─────────────────▼────────┐
   │ Local Python     │          │ Remote DeepWiki          │
   │ mcp_server.py    │          │ mcp.deepwiki.com/mcp     │
   │ (stdio)          │          │ (Streamable HTTP)        │
   └──────────────────┘          └──────────────────────────┘
```

## How a chat turn works

```
User types → mcp_app.py
  └── chatbot.chat_turn(msg, llm)               ## mcp_main.py
        ├── factory.create_chatbot(tools)       ## mcp_agents.py
        ├── create_chat_task(agent, msg, sum)   ## mcp_tasks.py
        ├── crew.kickoff()                      ## CrewAI
        │     ├── LLM #1 (Ollama)  → tool call decision
        │     ├── MCPServerAdapter → JSON-RPC over stdio/HTTP
        │     │     ├── (local) mcp_server.py @mcp.tool() runs
        │     │     └── (remote) DeepWiki processes the call
        │     └── LLM #2 (Ollama)  → final reply with tool result
        ├── _roll_summary_forward()             ## mcp_main.py
        │     └── crew.kickoff() (summarizer agent, no tools)
        │           └── LLM #3 (Ollama) → updated rolling summary
        └── return (reply, captured_trace)
```

Three Ollama calls per turn: 2 for the agent, 1 for the rolling summary
update.

## Pitfalls worth teaching

- **`crewai-tools` < 1.14.4** has a bug where `MCPServerAdapter` prompts
  to install `mcp` even when it is installed. Pin
  `crewai-tools[mcp] >= 1.14.4`.
- **DSL tool-name prefixing**: `Agent(..., mcps=[MCPServerStdio(...)])`
  is the modern pattern, but it auto-prefixes tool names with the
  server's command path. That breaks small models like `qwen2.5:3b`.
  We use `MCPServerAdapter` for clean tool names. See
  `run_mcp_demo_dsl.py` for the comparison.
- **CrewAI `memory=True` is broken with Ollama** on this stack — Chroma
  demands `OPENAI_API_KEY` even with `EMBEDDINGS_OLLAMA_*` env vars
  set. The chatbot uses a manual rolling summary instead.
- **Don't expose `python_repl` over MCP**. Arbitrary code execution
  through a public protocol = remote code execution for any connected
  agent.
- **Small-model tool-eagerness**: `qwen2.5:3b` will fire 3+ tool calls
  on `"Hi"` unless the agent's backstory explicitly tells it *when not
  to use tools*. See `mcp_agents.py:CHATBOT_BACKSTORY`.

## Configuration

Two env vars (defaults already work):

```bash
export OLLAMA_MODEL=qwen2.5:3b              # bigger = better tool routing
export OLLAMA_BASE_URL=http://localhost:11434
```

Both are surfaced as text inputs in the Streamlit sidebar, so you can
swap models live without restarting.

## Troubleshooting

### `MCPServerAdapter` prompts to install `mcp` on connect
You're on `crewai-tools < 1.14.4`. Upgrade:
```bash
pip install -U "crewai-tools[mcp]>=1.14.4"
```

### Memory init errors mentioning `CHROMA_OPENAI_API_KEY`
CrewAI 1.14's `memory=True` doesn't work with Ollama embedders here.
The chatbot doesn't use `memory=True`; if you toggle it on yourself,
expect this error.

### Agent fires tools on "Hi" / over-uses tools
Strengthen the backstory in `mcp_agents.py:CHATBOT_BACKSTORY` to spell
out when *not* to call tools, or bump to `qwen2.5:7b-instruct` for
more discerning routing.

### Ollama connection error
```bash
curl -s http://localhost:11434/api/tags && echo "Ollama up" || ollama serve &
```

### Leaked MCP server subprocesses
The chatbot disconnects cleanly on Disconnect, but if Streamlit was
killed mid-conversation the subprocess can survive:
```bash
pkill -f "python.*mcp_server.py"
```

## License

MIT.
