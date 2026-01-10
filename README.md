# Multi-Agent Systems with CrewAI

This branch implements Class 07: Multi-Agent Systems using CrewAI for collaborative report writing.

## Features

- **Multi-Role Agents**: Planner, Researcher, Writer, Critic, Summarizer
- **Planner-Executor-Critic Model**: Structured workflow for quality output
- **CrewAI Integration**: Agent orchestration and task management
- **Multiple Workflows**: Full report, quick report, research-only
- **Streamlit UI**: Interactive web interface for report generation
- **CLI Support**: Generate reports from command line

## Project Structure

```
.
├── crew_agents.py        # Multi-role agent definitions
├── crew_tasks.py         # Task definitions for workflows
├── crew_main.py          # CrewAI orchestration
├── crew_app.py           # Streamlit UI
├── run_crew.py           # CLI launcher
├── reports/              # Generated reports output
├── agent.py              # LangGraph ReAct agent (Class 06)
├── tools.py              # Custom tools
├── memory_manager.py     # Memory types
├── app.py                # RAG chatbot UI
├── chatbot.py            # RAG orchestrator
└── requirements.txt      # Dependencies
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

### 2. Run Streamlit UI

```bash
python run_crew.py
# Access at: http://localhost:8503
```

### 3. Generate via CLI

```bash
# Full report
python run_crew.py --topic "AI in Healthcare" --workflow full_report

# Quick report
python run_crew.py --topic "Machine Learning Basics" --workflow quick_report --style academic

# List options
python run_crew.py --list
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

## Multi-Agent Communication (MCP-Style)

The `MultiAgentOrchestrator` class supports communication between crews:

```python
from crew_main import MultiAgentOrchestrator

orchestrator = MultiAgentOrchestrator()

# Create multiple crews
crew1 = orchestrator.create_crew("research_team", crew_type="research_only")
crew2 = orchestrator.create_crew("writing_team", crew_type="review_team")

# Send messages between crews
orchestrator.send_message(
    from_crew="research_team",
    to_crew="writing_team",
    message_type="research_complete",
    content={"findings": "..."}
)
```

## Troubleshooting

### CrewAI Import Error
```bash
pip install crewai crewai-tools --upgrade
```

### Ollama Connection Error
```bash
docker ps | grep ollama
docker start ollama
```

### Memory Issues
```bash
rm -rf reports/
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
