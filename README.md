# LangChain Agent with Memory & Tools

## Class 06: LangChain Deep Dive - Memory, Tools & Agents

This branch extends the RAG chatbot with LangChain's advanced features including conversational memory, custom tools, and ReAct agents.

## Features

- **Multiple Memory Types**: Buffer, Window, Vector, and LangGraph memory
- **Custom Tools**: Calculator, Web Search, Python REPL, DateTime, RAG Search
- **ReAct Agent**: Reasoning + Acting pattern using LangGraph
- **Dual Mode**: RAG-only chatbot OR Agent with tools
- **Streamlit UI**: Interactive web interface for both modes

## Project Structure

```
.
├── agent.py               # LangGraph ReAct agent implementation
├── agent_app.py           # Streamlit UI for agent mode
├── memory_manager.py      # Memory types (buffer, window, vector)
├── tools.py               # Custom tools (calculator, search, REPL, etc.)
├── run_agent.py           # CLI launcher for agent app
├── example_agent_usage.py # Demo examples
├── app.py                 # Original RAG chatbot UI
├── chatbot.py             # RAG chatbot orchestrator
├── config.py              # Configuration settings
├── document_processor.py  # PDF processing
├── llm_handler.py         # LLM integration (Ollama/Azure)
├── vector_store.py        # ChromaDB vector store
├── utils.py               # Utility functions
├── requirements.txt       # Dependencies
├── pdfFiles/              # PDF storage
├── vectorDB/              # Vector database
└── memoryDB/              # Vector memory storage
```

## New Modules

### 1. memory_manager.py

Implements multiple memory types for conversation history:

| Memory Type | Description | Use Case |
|-------------|-------------|----------|
| `BUFFER` | Stores all messages | Short conversations |
| `WINDOW` | Keeps last K messages | Long conversations |
| `VECTOR` | Semantic search over history | Topic recall |
| `LANGGRAPH` | Checkpoint-based memory | Persistent agents |

```python
from memory_manager import MemoryManager, MemoryType

# Create buffer memory
memory = MemoryManager(memory_type=MemoryType.BUFFER)
memory.add_user_message("Hello")
memory.add_ai_message("Hi there!")

# Create window memory (last 5 turns)
memory = MemoryManager(memory_type=MemoryType.WINDOW, k=5)
```

### 2. tools.py

Custom tools for the agent:

| Tool | Description | Example |
|------|-------------|---------|
| `CalculatorTool` | Math expressions | `sqrt(16) + 2^3` |
| `WebSearchTool` | DuckDuckGo search | Search the internet |
| `SafePythonREPLTool` | Execute Python code | Data processing |
| `DateTimeTool` | Current date/time | `now`, `weekday` |
| `RAGSearchTool` | Search uploaded PDFs | Document Q&A |

```python
from tools import CalculatorTool, DateTimeTool, ToolFactory

# Use individual tools
calc = CalculatorTool()
result = calc._run("25 * 4")  # Returns "100"

# Get all tools
factory = ToolFactory(vector_store=my_vector_store)
tools = factory.get_all_tools()
```

### 3. agent.py

LangGraph ReAct agent with tools and memory:

```python
from agent import create_agent

# Create agent with tools
agent = create_agent(
    memory_type="buffer",
    use_tools=True,
    vector_store=None  # Optional: for RAG search
)

# Chat with agent
response = agent.chat("What is 25 * 4?")
print(response)  # "The result is 100"

# Get full response with tool calls
result = agent.get_full_response("What day is it?")
print(result["output"])       # "Today is Saturday"
print(result["tool_calls"])   # List of tool calls made
```

### 4. agent_app.py

Streamlit UI for the agent with:
- Memory type selection
- Tool enable/disable
- Agent thinking process display
- PDF upload for RAG search
- Chat history management

## Installation

```bash
# Clone and checkout branch
git clone https://github.com/AIFahim/LLM-Improvement-with-RAG-End-to-End-Chatbot-Development.git
cd LLM-Improvement-with-RAG-End-to-End-Chatbot-Development
git checkout tool-calling-agents

# Create conda environment (optional)
conda create -n langchain-agent python=3.11
conda activate langchain-agent

# Install dependencies
pip install -r requirements.txt
```

## Requirements

Key dependencies:
- `langchain>=1.0.0`
- `langchain-core`
- `langchain-community`
- `langchain-ollama`
- `langchain-openai`
- `langchain-chroma`
- `langchain-experimental`
- `langgraph`
- `duckduckgo-search`
- `chromadb`
- `streamlit`

## Usage

### 1. Start Ollama

```bash
# Using Docker
docker start ollama

# Or native Ollama
ollama serve
```

### 2. Run Agent App

```bash
# Using run script
python run_agent.py

# Or directly with Streamlit
streamlit run agent_app.py --server.port 8502
```

### 3. Run with Options

```bash
# Use different memory type
python run_agent.py --memory summary

# Use Azure OpenAI
python run_agent.py --provider azure --api-key YOUR_KEY --endpoint YOUR_ENDPOINT

# Check configuration only
python run_agent.py --check
```

### 4. Access the App

Open browser: **http://localhost:8502**

## Configuration

Edit `config.py` for settings:

```python
# LLM Provider
LLM_PROVIDER = "ollama"  # or "azure"
OLLAMA_MODEL = "qwen2.5:1.5b"

# Agent Settings
AGENT_VERBOSE = True
AGENT_MAX_ITERATIONS = 10

# Memory Settings
MEMORY_TYPE = "buffer"
MEMORY_K = 5

# Tools
TOOLS_ENABLED = {
    "calculator": True,
    "web_search": True,
    "python_repl": True,
    "datetime": True,
    "rag_search": True,
}
```

## Examples

### Calculator

```
User: What is the square root of 144 plus 10?
Agent: [Uses calculator tool]
       sqrt(144) = 12
       12 + 10 = 22
       The answer is 22.
```

### DateTime

```
User: What day is it today?
Agent: [Uses datetime tool]
       Today is Saturday, January 3, 2026.
```

### Web Search

```
User: What is the latest Python version?
Agent: [Uses web_search tool]
       Based on my search, Python 3.12 is the latest stable version...
```

### RAG Search

```
User: What does the document say about reinforcement learning?
Agent: [Uses rag_search tool]
       According to the uploaded documents, reinforcement learning is...
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Streamlit UI (agent_app.py)               │
│  ┌─────────────────┐  ┌────────────────────────────────┐   │
│  │ Sidebar Config  │  │      Chat Interface            │   │
│  │ - Memory Type   │  │ - Message History              │   │
│  │ - Tools Toggle  │  │ - Agent Thinking Display       │   │
│  │ - PDF Upload    │  │ - Tool Call Visualization      │   │
│  └────────┬────────┘  └───────────────┬────────────────┘   │
└───────────┼───────────────────────────┼────────────────────┘
            │                           │
            └───────────────┬───────────┘
                            │
            ┌───────────────▼───────────────┐
            │   ConversationalAgent         │
            │   (agent.py)                  │
            └───────────────┬───────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼───────┐  ┌────────▼────────┐  ┌──────▼──────┐
│ LangChainAgent│  │ MemoryManager   │  │ ToolFactory │
│ (ReAct Agent) │  │ (memory_manager)│  │ (tools.py)  │
└───────┬───────┘  └─────────────────┘  └──────┬──────┘
        │                                      │
        │                              ┌───────┴───────┐
┌───────▼───────┐                      │   Tools       │
│ LangGraph     │                      ├───────────────┤
│ create_react  │                      │ Calculator    │
│ _agent        │                      │ WebSearch     │
└───────┬───────┘                      │ PythonREPL    │
        │                              │ DateTime      │
┌───────▼───────┐                      │ RAGSearch     │
│ LLM Provider  │                      └───────────────┘
│ (Ollama/Azure)│
└───────────────┘
```

## ReAct Pattern Flow

```
1. User Question
        ↓
2. Agent Thought: "I need to calculate this..."
        ↓
3. Action: calculator
4. Action Input: "25 * 4"
        ↓
5. Observation: "100"
        ↓
6. Thought: "I now know the answer"
        ↓
7. Final Answer: "25 * 4 equals 100"
```

## API Reference

### create_agent()

```python
def create_agent(
    memory_type: str = "buffer",    # buffer, window, vector
    use_tools: bool = True,         # Enable/disable tools
    vector_store: Any = None,       # For RAG search
) -> ConversationalAgent
```

### ConversationalAgent

```python
agent.chat(message: str) -> str
agent.get_full_response(message: str) -> Dict
agent.clear_memory() -> None
agent.get_tools() -> List[str]
agent.memory_info -> Dict
```

### MemoryManager

```python
manager.add_user_message(content: str)
manager.add_ai_message(content: str)
manager.get_messages() -> List[BaseMessage]
manager.clear() -> None
manager.get_memory_info() -> Dict
```

## Comparison: RAG Chatbot vs Agent

| Feature | RAG Chatbot | Agent with Tools |
|---------|-------------|------------------|
| PDF Q&A | Yes | Yes (via RAG tool) |
| Calculations | No | Yes |
| Web Search | No | Yes |
| Code Execution | No | Yes |
| Current Time | No | Yes |
| Memory Types | Buffer only | Multiple |
| Reasoning | Simple | ReAct pattern |

## Troubleshooting

### Ollama Connection Error
```bash
# Check if Ollama is running
docker ps | grep ollama
# Start if needed
docker start ollama
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Memory Issues
```bash
# Clear memory directory
rm -rf memoryDB/
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request
