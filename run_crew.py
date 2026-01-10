#!/usr/bin/env python
"""
Class 07: Multi-Agent Systems - CLI Launcher

Run the multi-agent report writing system via command line.
Supports both Streamlit UI and direct CLI report generation.

Usage:
    # Run Streamlit UI
    python run_crew.py

    # Generate report via CLI
    python run_crew.py --topic "AI in Healthcare" --workflow full_report

    # List available options
    python run_crew.py --list
"""

import argparse
import subprocess
import sys
import os
from termcolor import colored


def check_dependencies():
    """Check if required dependencies are installed."""
    required = [
        ("crewai", "crewai"),
        ("streamlit", "streamlit"),
        ("langchain", "langchain"),
    ]

    missing = []
    for module, package in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print(colored("Missing dependencies:", "red"))
        for pkg in missing:
            print(colored(f"  - {pkg}", "yellow"))
        print(colored("\nInstall with: pip install -r requirements.txt", "cyan"))
        return False

    return True


def list_options():
    """List available crews, workflows, and styles."""
    from crew_agents import AGENT_CONFIGS, get_agent_info
    from crew_tasks import get_workflow_info

    print(colored("\n" + "=" * 60, "blue"))
    print(colored("Multi-Agent Report Writer - Available Options", "blue", attrs=["bold"]))
    print(colored("=" * 60, "blue"))

    # Crew Types
    print(colored("\nCrew Types:", "green", attrs=["bold"]))
    for name, config in AGENT_CONFIGS.items():
        print(f"  {colored(name, 'cyan')}: {config['description']}")
        print(f"    Agents: {', '.join(config['agents'])}")

    # Agents
    print(colored("\nAvailable Agents:", "green", attrs=["bold"]))
    for name, info in get_agent_info().items():
        delegate = " (can delegate)" if info["can_delegate"] else ""
        print(f"  {colored(name, 'cyan')}: {info['role']}{delegate}")
        print(f"    {info['description']}")

    # Workflows
    print(colored("\nWorkflow Templates:", "green", attrs=["bold"]))
    for name, workflow in get_workflow_info().items():
        print(f"  {colored(name, 'cyan')}: {workflow['description']}")
        print(f"    Tasks: {' -> '.join(workflow['tasks'])}")

    # Writing Styles
    print(colored("\nWriting Styles:", "green", attrs=["bold"]))
    styles = ["professional", "academic", "casual", "technical"]
    for style in styles:
        print(f"  - {style}")

    print()


def run_streamlit():
    """Run the Streamlit UI."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "crew_app.py")

    print(colored("\nStarting Multi-Agent Report Writer UI...", "green"))
    print(colored("Access at: http://localhost:8503", "cyan"))
    print(colored("Press Ctrl+C to stop\n", "yellow"))

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                app_path,
                "--server.port",
                "8503",
                "--server.headless",
                "true",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        print(colored("\nShutting down...", "yellow"))
    except subprocess.CalledProcessError as e:
        print(colored(f"\nError running Streamlit: {e}", "red"))
        sys.exit(1)


def run_cli_report(args):
    """Generate a report via CLI."""
    from crew_main import create_report_crew

    print(colored("\n" + "=" * 60, "blue"))
    print(colored("Multi-Agent Report Writer - CLI Mode", "blue", attrs=["bold"]))
    print(colored("=" * 60, "blue"))

    print(colored(f"\nTopic: {args.topic}", "green"))
    print(colored(f"Workflow: {args.workflow}", "cyan"))
    print(colored(f"Crew: {args.crew}", "cyan"))
    print(colored(f"Style: {args.style}", "cyan"))

    if args.requirements:
        print(colored(f"Requirements: {args.requirements}", "cyan"))

    print(colored("\nInitializing crew...", "yellow"))

    # Create crew
    crew = create_report_crew(
        crew_type=args.crew,
        verbose=args.verbose,
        process=args.process,
    )

    print(colored(f"Crew ready with agents: {', '.join(crew.agents.keys())}", "green"))
    print(colored("\nGenerating report... This may take a few minutes.\n", "yellow"))

    # Generate report
    result = crew.create_report(
        topic=args.topic,
        requirements=args.requirements,
        style=args.style,
        word_count=args.word_count,
        workflow=args.workflow,
    )

    # Display results
    if result.get("success"):
        print(colored("\n" + "=" * 60, "green"))
        print(colored("Report Generated Successfully!", "green", attrs=["bold"]))
        print(colored("=" * 60, "green"))

        print(f"\nExecution Time: {result['execution_time']:.2f} seconds")
        print(f"Tasks Completed: {result.get('tasks_completed', 'N/A')}")
        print(f"Agents Used: {', '.join(result.get('agents_used', []))}")

        print(colored("\n--- REPORT ---\n", "cyan"))
        print(result.get("result", "No content"))
        print(colored("\n--- END REPORT ---\n", "cyan"))

        print(colored(f"Report saved to: {crew.output_dir}/", "green"))
    else:
        print(colored("\n" + "=" * 60, "red"))
        print(colored("Report Generation Failed", "red", attrs=["bold"]))
        print(colored("=" * 60, "red"))
        print(f"\nError: {result.get('error', 'Unknown error')}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Report Writer - Class 07",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run the Streamlit UI
  python run_crew.py

  # Generate a report via CLI
  python run_crew.py --topic "The Future of AI" --workflow full_report

  # Quick report with specific style
  python run_crew.py --topic "Machine Learning Basics" --workflow quick_report --style academic

  # List all available options
  python run_crew.py --list
        """,
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available crews, workflows, and styles",
    )

    parser.add_argument(
        "--topic",
        type=str,
        help="Report topic (enables CLI mode)",
    )

    parser.add_argument(
        "--requirements",
        type=str,
        default=None,
        help="Additional requirements for the report",
    )

    parser.add_argument(
        "--workflow",
        type=str,
        default="full_report",
        choices=["full_report", "quick_report", "research_plan", "write_review"],
        help="Workflow template to use (default: full_report)",
    )

    parser.add_argument(
        "--crew",
        type=str,
        default="report_writing",
        choices=["report_writing", "quick_report", "research_only", "review_team"],
        help="Crew type to use (default: report_writing)",
    )

    parser.add_argument(
        "--style",
        type=str,
        default="professional",
        choices=["professional", "academic", "casual", "technical"],
        help="Writing style (default: professional)",
    )

    parser.add_argument(
        "--word-count",
        type=int,
        default=1500,
        help="Target word count (default: 1500)",
    )

    parser.add_argument(
        "--process",
        type=str,
        default="sequential",
        choices=["sequential", "hierarchical"],
        help="Execution process (default: sequential)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose output (default: True)",
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check dependencies only",
    )

    args = parser.parse_args()

    # Check dependencies
    if args.check:
        if check_dependencies():
            print(colored("All dependencies installed!", "green"))
        sys.exit(0)

    if not check_dependencies():
        sys.exit(1)

    # List options
    if args.list:
        list_options()
        sys.exit(0)

    # CLI mode or UI mode
    if args.topic:
        run_cli_report(args)
    else:
        run_streamlit()


if __name__ == "__main__":
    main()
