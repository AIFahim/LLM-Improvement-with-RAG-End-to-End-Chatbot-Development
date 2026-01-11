"""
Class 07: Multi-Agent Systems - CrewAI Orchestration

This module orchestrates multi-agent collaboration using CrewAI.
Implements the Planner -> Executor -> Critic model for report writing.

Key concepts:
- Crew: A team of agents working together
- Process: Sequential or Hierarchical execution
- Tasks: Work items assigned to agents
"""

from crewai import Crew, Process
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
import json
import os

from crew_agents import AgentFactory, get_agent_info, AGENT_CONFIGS
from crew_tasks import TaskFactory, get_workflow_info, WORKFLOW_TEMPLATES


class ReportWritingCrew:
    """
    A multi-agent crew for collaborative report writing.

    This crew implements the Planner -> Executor -> Critic model:
    1. Planner: Creates report structure and research strategy
    2. Researchers: Gather information
    3. Writers: Compose content
    4. Critics: Review and provide feedback
    """

    def __init__(
        self,
        verbose: bool = True,
        process: str = "sequential",
        memory: bool = False,  # Disabled by default - requires OpenAI API key for embeddings
        output_dir: str = "reports",
    ):
        """
        Initialize the ReportWritingCrew.

        Args:
            verbose: Whether to log agent actions
            process: Execution process ("sequential" or "hierarchical")
            memory: Whether agents should have memory (requires OpenAI API key)
            output_dir: Directory to save generated reports
        """
        self.verbose = verbose
        self.process = Process.sequential if process == "sequential" else Process.hierarchical
        self.memory = memory
        self.output_dir = output_dir

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize factories
        self.agent_factory = AgentFactory(verbose=verbose)
        self.agents = None
        self.task_factory = None
        self.crew = None

        # Execution history
        self.execution_history: List[Dict] = []

    def setup_crew(
        self,
        crew_type: str = "report_writing",
        custom_agents: Optional[List[str]] = None,
    ) -> None:
        """
        Set up the crew with agents.

        Args:
            crew_type: Predefined crew type from AGENT_CONFIGS
            custom_agents: Custom list of agent names to use
        """
        if custom_agents:
            # Create only specified agents
            all_agents = self.agent_factory.create_all_agents()
            self.agents = {k: v for k, v in all_agents.items() if k in custom_agents}
        elif crew_type in AGENT_CONFIGS:
            # Use predefined configuration
            agent_names = AGENT_CONFIGS[crew_type]["agents"]
            all_agents = self.agent_factory.create_all_agents()
            self.agents = {k: v for k, v in all_agents.items() if k in agent_names}
        else:
            # Default: create all agents
            self.agents = self.agent_factory.create_all_agents()

        # Initialize task factory with agents
        self.task_factory = TaskFactory(self.agents)

    def create_report(
        self,
        topic: str,
        requirements: Optional[str] = None,
        style: str = "professional",
        word_count: Optional[int] = None,
        workflow: str = "full_report",
        callback: Optional[Callable[[str, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Create a report using the multi-agent crew.

        Args:
            topic: The report topic
            requirements: Additional requirements or constraints
            style: Writing style (professional, academic, casual)
            word_count: Target word count
            workflow: Workflow template to use
            callback: Optional callback function(agent_name, output) for progress

        Returns:
            Dictionary with report content and metadata
        """
        if not self.agents:
            self.setup_crew()

        # Create tasks based on workflow
        if workflow == "full_report":
            tasks = self.task_factory.create_report_workflow(
                topic=topic,
                requirements=requirements,
                style=style,
                word_count=word_count,
                include_review=True,
                include_summary=True,
            )
        elif workflow == "quick_report":
            tasks = self.task_factory.create_quick_report_workflow(
                topic=topic,
                style=style,
            )
        elif workflow == "research_plan":
            tasks = self.task_factory.create_research_workflow(
                topic=topic,
                research_areas=requirements,
            )
        else:
            # Default to full workflow
            tasks = self.task_factory.create_report_workflow(
                topic=topic,
                requirements=requirements,
                style=style,
                word_count=word_count,
            )

        # Create the crew (memory disabled to avoid OpenAI API key requirement)
        self.crew = Crew(
            agents=list(self.agents.values()),
            tasks=tasks,
            process=self.process,
            verbose=self.verbose,
            memory=False,
            embedder={
                "provider": "ollama",
                "config": {
                    "model": "qwen2.5:1.5b",
                    "base_url": "http://localhost:11434",
                }
            },
        )

        # Execute the crew
        start_time = datetime.now()

        try:
            result = self.crew.kickoff()

            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            # Prepare output
            output = {
                "success": True,
                "topic": topic,
                "workflow": workflow,
                "result": str(result),
                "execution_time": execution_time,
                "timestamp": start_time.isoformat(),
                "agents_used": list(self.agents.keys()),
                "tasks_completed": len(tasks),
            }

            # Save to history
            self.execution_history.append(output)

            # Save report to file
            self._save_report(output)

            return output

        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            output = {
                "success": False,
                "topic": topic,
                "workflow": workflow,
                "error": str(e),
                "execution_time": execution_time,
                "timestamp": start_time.isoformat(),
            }

            self.execution_history.append(output)
            return output

    def _save_report(self, output: Dict) -> str:
        """
        Save the report to a file.

        Args:
            output: The report output dictionary

        Returns:
            Path to the saved file
        """
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_slug = output["topic"][:30].replace(" ", "_").replace("/", "_")
        filename = f"{timestamp}_{topic_slug}.md"
        filepath = os.path.join(self.output_dir, filename)

        # Write report content
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# Report: {output['topic']}\n\n")
            f.write(f"**Generated:** {output['timestamp']}\n")
            f.write(f"**Workflow:** {output['workflow']}\n")
            f.write(f"**Agents:** {', '.join(output.get('agents_used', []))}\n")
            f.write(f"**Execution Time:** {output['execution_time']:.2f} seconds\n\n")
            f.write("---\n\n")
            f.write(output.get("result", "No content generated"))

        # Also save JSON metadata
        json_filepath = filepath.replace(".md", ".json")
        with open(json_filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)

        return filepath

    def get_crew_info(self) -> Dict:
        """
        Get information about the current crew.

        Returns:
            Dictionary with crew information
        """
        return {
            "agents": list(self.agents.keys()) if self.agents else [],
            "agent_details": get_agent_info(),
            "process": "sequential" if self.process == Process.sequential else "hierarchical",
            "memory_enabled": self.memory,
            "available_workflows": get_workflow_info(),
            "available_crews": AGENT_CONFIGS,
        }

    def get_execution_history(self) -> List[Dict]:
        """
        Get the execution history.

        Returns:
            List of previous execution results
        """
        return self.execution_history


class MultiAgentOrchestrator:
    """
    High-level orchestrator for multi-agent systems.

    Supports:
    - Multiple crew configurations
    - Different execution processes
    - Communication between crews (MCP-style)
    """

    def __init__(self):
        """Initialize the orchestrator."""
        self.crews: Dict[str, ReportWritingCrew] = {}
        self.message_queue: List[Dict] = []

    def create_crew(
        self,
        crew_id: str,
        crew_type: str = "report_writing",
        **kwargs,
    ) -> ReportWritingCrew:
        """
        Create a new crew.

        Args:
            crew_id: Unique identifier for the crew
            crew_type: Type of crew to create
            **kwargs: Additional arguments for crew initialization

        Returns:
            The created ReportWritingCrew instance
        """
        crew = ReportWritingCrew(**kwargs)
        crew.setup_crew(crew_type=crew_type)
        self.crews[crew_id] = crew
        return crew

    def get_crew(self, crew_id: str) -> Optional[ReportWritingCrew]:
        """
        Get a crew by ID.

        Args:
            crew_id: The crew identifier

        Returns:
            The ReportWritingCrew or None if not found
        """
        return self.crews.get(crew_id)

    def send_message(
        self,
        from_crew: str,
        to_crew: str,
        message_type: str,
        content: Any,
    ) -> None:
        """
        Send a message between crews (MCP-style communication).

        Args:
            from_crew: Source crew ID
            to_crew: Destination crew ID
            message_type: Type of message
            content: Message content
        """
        message = {
            "from": from_crew,
            "to": to_crew,
            "type": message_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        self.message_queue.append(message)

    def get_messages(self, crew_id: str) -> List[Dict]:
        """
        Get messages for a specific crew.

        Args:
            crew_id: The crew identifier

        Returns:
            List of messages addressed to the crew
        """
        return [m for m in self.message_queue if m["to"] == crew_id]

    def run_parallel_reports(
        self,
        topics: List[str],
        **kwargs,
    ) -> List[Dict]:
        """
        Run multiple reports in parallel (each with its own crew).

        Args:
            topics: List of topics to create reports for
            **kwargs: Arguments passed to create_report

        Returns:
            List of report results
        """
        results = []

        for i, topic in enumerate(topics):
            crew_id = f"crew_{i}"
            crew = self.create_crew(crew_id, crew_type="quick_report")
            result = crew.create_report(topic=topic, workflow="quick_report", **kwargs)
            results.append(result)

        return results

    def get_all_crews_status(self) -> Dict:
        """
        Get status of all crews.

        Returns:
            Dictionary with crew statuses
        """
        return {
            crew_id: {
                "agents": list(crew.agents.keys()) if crew.agents else [],
                "reports_generated": len(crew.execution_history),
            }
            for crew_id, crew in self.crews.items()
        }


def create_report_crew(
    crew_type: str = "report_writing",
    verbose: bool = True,
    process: str = "sequential",
) -> ReportWritingCrew:
    """
    Convenience function to create a report writing crew.

    Args:
        crew_type: Type of crew to create
        verbose: Whether to log actions
        process: Execution process

    Returns:
        Configured ReportWritingCrew instance
    """
    crew = ReportWritingCrew(verbose=verbose, process=process)
    crew.setup_crew(crew_type=crew_type)
    return crew


# Example usage and demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Class 07: Multi-Agent Systems - CrewAI Demo")
    print("=" * 60)

    # Create a crew
    crew = create_report_crew(
        crew_type="report_writing",
        verbose=True,
        process="sequential",
    )

    # Show crew info
    info = crew.get_crew_info()
    print("\nCrew Information:")
    print(f"  Agents: {info['agents']}")
    print(f"  Process: {info['process']}")
    print(f"  Memory: {info['memory_enabled']}")

    print("\nAvailable Workflows:")
    for name, details in info["available_workflows"].items():
        print(f"  - {name}: {details['description']}")

    print("\nAvailable Crew Types:")
    for name, details in info["available_crews"].items():
        print(f"  - {name}: {details['description']}")

    # Demo report creation
    print("\n" + "=" * 60)
    print("Creating a demo report...")
    print("=" * 60)

    result = crew.create_report(
        topic="The Impact of Artificial Intelligence on Modern Healthcare",
        requirements="Focus on diagnostic applications and patient care improvements",
        style="professional",
        word_count=1500,
        workflow="full_report",
    )

    if result["success"]:
        print(f"\nReport generated successfully!")
        print(f"Execution time: {result['execution_time']:.2f} seconds")
        print(f"Agents used: {', '.join(result['agents_used'])}")
        print(f"\nReport saved to: {crew.output_dir}/")
    else:
        print(f"\nReport generation failed: {result.get('error', 'Unknown error')}")
