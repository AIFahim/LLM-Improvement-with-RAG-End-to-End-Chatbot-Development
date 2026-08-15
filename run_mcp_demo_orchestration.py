"""
CrewAI orchestration features, driven against the MCP tool server.

Run:
    python run_mcp_demo_orchestration.py                 # sequential pipeline
    python run_mcp_demo_orchestration.py --demo hierarchical
    python run_mcp_demo_orchestration.py --demo inputs
    python run_mcp_demo_orchestration.py --demo memory
    python run_mcp_demo_orchestration.py --demo all

What this adds over the other runners:

    run_mcp_demo.py           one agent, one task, one crew
    run_mcp_demo_dsl.py       same, via the mcps=[...] DSL
    run_mcp_demo_remote.py    same, against a remote server
    this file                 several agents inside ONE crew, plus the
                              orchestration machinery CrewAI provides to
                              coordinate them

Specifically, each demo below isolates one mechanism:

  sequential    Three agents in a single Crew. Tasks are wired with
                context=[...] so each task receives the previous task's
                output — the researcher -> analyst -> archivist pipeline.
                Also shows task_callback, max_iter and max_retry_limit.

  hierarchical  Process.hierarchical with a manager_llm. Tasks carry NO
                agent= assignment; the manager picks who runs what and
                the workers need allow_delegation=True to accept it.

  inputs        crew.kickoff(inputs={...}) filling {placeholders} in a
                task description, so one crew definition serves many runs.

  memory        CrewAI's real memory layer, working fully locally.

A note on that last one. mcp_main.py deliberately avoids memory=True and
says why: CrewAI's memory defaults to OpenAI for BOTH halves of the
subsystem, so it dies without OPENAI_API_KEY. Passing an embedder alone
is not enough — memory also runs an LLM to extract and recall entities,
and that one is a separate default. Configuring both is what makes it
work offline:

    Memory(llm="ollama/...", embedder={"provider": "ollama", ...})

Requires an embedding model alongside the chat model:

    ollama pull nomic-embed-text

The rolling-summary approach in mcp_main.py is still the right choice
for the chatbot — it needs no embedder and keeps prompts bounded. This
demo exists to show the alternative actually running.

Small local models are the practical limit here, not the wiring. A 3B
model will sometimes return an empty completion mid tool-call, which
surfaces as "Invalid response from LLM call". Each demo is therefore
run independently and reports its own failure without taking down the
rest of the file. Bump OLLAMA_MODEL if a stage flakes repeatedly.
"""

import argparse
import os
from pathlib import Path

from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai.memory.unified_memory import Memory

from mcp_client import MCPCrewClient

REPO_ROOT = Path(__file__).resolve().parent
SERVER_SCRIPT = REPO_ROOT / "mcp_server.py"

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def build_llm() -> LLM:
    """Local Ollama LLM. Bump via OLLAMA_MODEL env var if tool routing flakes."""
    return LLM(model=f"ollama/{OLLAMA_MODEL}", base_url=OLLAMA_BASE_URL)


def build_memory() -> Memory:
    """CrewAI memory with both halves pointed at Ollama.

    llm      -> the extraction/recall analysis calls
    embedder -> the vector search over stored memories

    Leaving either at its default sends that half to OpenAI.
    """
    return Memory(
        llm=f"ollama/{OLLAMA_MODEL}",
        embedder={
            "provider": "ollama",
            "config": {
                "url": f"{OLLAMA_BASE_URL}/api/embeddings",
                "model_name": OLLAMA_EMBED_MODEL,
            },
        },
    )


# -- agents ----------------------------------------------------------------


def build_crew_agents(tools: list, llm: LLM, *, delegating: bool = False):
    """Three specialised agents that share one crew.

    delegating=True is required for the hierarchical demo: a worker that
    cannot be delegated to will never be given work by the manager.
    """
    researcher = Agent(
        role="Research Assistant",
        goal="Gather concrete facts using the MCP tools, never from memory.",
        backstory=(
            "You look things up rather than recalling them. When a "
            "question involves arithmetic or the current date, you call "
            "the matching tool and report exactly what it returned."
        ),
        tools=tools,
        llm=llm,
        max_iter=3,
        max_retry_limit=2,
        allow_delegation=delegating,
        verbose=True,
    )

    analyst = Agent(
        role="Analyst",
        goal="Turn raw tool output into a short, plain-language finding.",
        backstory=(
            "You explain what numbers mean. You never re-run the "
            "research; you work from what the researcher handed you."
        ),
        tools=[],
        llm=llm,
        max_iter=2,
        max_retry_limit=1,
        allow_delegation=delegating,
        verbose=True,
    )

    archivist = Agent(
        role="Report Archivist",
        goal="Persist the finished write-up through the MCP report tools.",
        backstory=(
            "You manage a library of reports. You only ever write to the "
            "report store through the MCP tools provided to you."
        ),
        tools=tools,
        llm=llm,
        max_iter=3,
        max_retry_limit=2,
        allow_delegation=delegating,
        verbose=True,
    )

    return researcher, analyst, archivist


# -- demo 1: sequential pipeline with task context --------------------------


def demo_sequential(tools: list, llm: LLM) -> None:
    """Three agents, one crew, tasks chained with context=[...].

    This is the piece the other runners never show: task 2 receives
    task 1's output automatically, because it names task 1 as context.
    """
    researcher, analyst, archivist = build_crew_agents(tools, llm)

    research_task = Task(
        description=(
            "Call the `calculator` tool to evaluate 17 * 23, then call "
            "the `datetime_now` tool with operation 'date'. Report both "
            "results exactly as the tools returned them."
        ),
        expected_output="The product, and today's date, as returned by the tools.",
        agent=researcher,
    )

    analysis_task = Task(
        description=(
            "Using only the researcher's findings, write one sentence "
            "stating the product and the date. Do not call any tools."
        ),
        expected_output="A single plain sentence.",
        agent=analyst,
        context=[research_task],  # <- task chaining
    )

    archive_task = Task(
        description=(
            "Call the `save_report` tool with title 'orchestration-demo' "
            "and the analyst's sentence as the content. Return the path "
            "the tool gave back."
        ),
        expected_output="The relative path of the saved report.",
        agent=archivist,
        context=[research_task, analysis_task],  # <- sees both predecessors
    )

    def on_task_done(task_output) -> None:
        """task_callback fires after every task completes."""
        raw = str(getattr(task_output, "raw", task_output))
        print(f"\n[callback] task finished -> {raw[:100].strip()}...\n")

    crew = Crew(
        agents=[researcher, analyst, archivist],  # <- three, not one
        tasks=[research_task, analysis_task, archive_task],
        process=Process.sequential,
        task_callback=on_task_done,
        verbose=True,
        memory=False,
    )
    print(crew.kickoff())


# -- demo 2: hierarchical delegation ---------------------------------------


def demo_hierarchical(tools: list, llm: LLM) -> None:
    """A manager LLM assigns the work instead of the author assigning it.

    Two differences from the sequential demo:
      - tasks carry no agent=, so the manager chooses
      - workers set allow_delegation=True, so they can be assigned to
    """
    researcher, analyst, archivist = build_crew_agents(
        tools, llm, delegating=True
    )

    lookup_task = Task(
        description=(
            "Find out what 144 divided by 12 is, using the `calculator` "
            "tool rather than working it out unaided."
        ),
        expected_output="The quotient, as returned by the tool.",
    )

    summary_task = Task(
        description=(
            "State the result from the previous step in one short sentence."
        ),
        expected_output="A single plain sentence.",
        context=[lookup_task],
    )

    crew = Crew(
        agents=[researcher, analyst, archivist],
        tasks=[lookup_task, summary_task],
        process=Process.hierarchical,
        manager_llm=llm,  # <- the delegating brain
        verbose=True,
        memory=False,
    )
    print(crew.kickoff())


# -- demo 3: templated inputs ----------------------------------------------


def demo_inputs(tools: list, llm: LLM) -> None:
    """One crew definition, many runs, via kickoff(inputs={...})."""
    researcher, _, _ = build_crew_agents(tools, llm)

    task = Task(
        description=(
            "Call the `calculator` tool to evaluate {expression}. "
            "Report only what the tool returned."
        ),
        expected_output="The value of the expression.",
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
        memory=False,
    )
    # {expression} is filled at kickoff time, not at definition time
    print(crew.kickoff(inputs={"expression": "sqrt(1764)"}))


# -- demo 4: CrewAI memory, fully local ------------------------------------


def demo_memory(tools: list, llm: LLM) -> None:
    """Two crews sharing one Memory: the second recalls the first.

    Nothing here reaches OpenAI, which is the point — see the module
    docstring for why both llm= and embedder= have to be set.
    """
    memory = build_memory()
    researcher, analyst, _ = build_crew_agents(tools, llm)

    store_task = Task(
        description=(
            "The user's name is Fahim and he is working on the Class 07 "
            "MCP demo. Acknowledge this in one short sentence."
        ),
        expected_output="A one-sentence acknowledgement.",
        agent=researcher,
    )
    Crew(
        agents=[researcher],
        tasks=[store_task],
        process=Process.sequential,
        memory=memory,  # <- shared instance
        verbose=True,
    ).kickoff()

    recall_task = Task(
        description=(
            "What is the user's name, and what are they working on? "
            "Answer from memory. Do not call any tools."
        ),
        expected_output="The user's name and current project.",
        agent=analyst,
    )
    print(
        Crew(
            agents=[analyst],
            tasks=[recall_task],
            process=Process.sequential,
            memory=memory,  # <- same store, different crew
            verbose=True,
        ).kickoff()
    )

    # Show the raw store too, so it is clear the recall was not luck
    print("\n--- what memory actually holds ---")
    for match in memory.recall("user name and project", limit=3):
        print(f"  [{match.score:.2f}] {match.record.content}")


DEMOS = {
    "sequential": demo_sequential,
    "hierarchical": demo_hierarchical,
    "inputs": demo_inputs,
    "memory": demo_memory,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo",
        choices=[*DEMOS, "all"],
        default="sequential",
        help="which orchestration mechanism to run (default: sequential)",
    )
    args = parser.parse_args()
    selected = list(DEMOS) if args.demo == "all" else [args.demo]

    client = MCPCrewClient(server_script=str(SERVER_SCRIPT))
    tools = client.start()
    try:
        print(f"\n[OK] MCP tools discovered: {[t.name for t in tools]}\n")
        llm = build_llm()

        for name in selected:
            print("=" * 60)
            print(f"DEMO: {name}")
            print("=" * 60)
            try:
                DEMOS[name](tools, llm)
            except Exception as e:
                # One flaky stage should not hide the results of the others
                print(f"\n[FAILED] {name}: {type(e).__name__}: {e}\n")
    finally:
        client.stop()
        print("\n[OK] MCP server stopped.")


if __name__ == "__main__":
    main()
