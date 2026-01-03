#!/usr/bin/env python3
"""
Run Script for LangChain Agent Application
Class 06: LangChain Deep Dive - Memory, Tools & Agents

Usage:
    python run_agent.py                           # Run with Ollama (default)
    python run_agent.py --provider azure          # Run with Azure OpenAI
    python run_agent.py --memory summary          # Use summary memory
    python run_agent.py --no-tools                # Disable tools
    python run_agent.py --check                   # Check configuration only
"""

import argparse
import os
import sys
import subprocess
import requests
from termcolor import colored


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the LangChain Agent application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_agent.py                         # Default: Ollama with buffer memory
  python run_agent.py --memory summary        # Use summary memory
  python run_agent.py --memory vector         # Use vector memory
  python run_agent.py --provider azure        # Use Azure OpenAI
  python run_agent.py --verbose               # Enable verbose agent output
  python run_agent.py --check                 # Check configuration only
        """
    )

    # Provider selection
    parser.add_argument(
        "--provider",
        choices=["ollama", "azure"],
        default="ollama",
        help="LLM provider to use (default: ollama)"
    )

    # Memory configuration
    parser.add_argument(
        "--memory",
        choices=["buffer", "buffer_window", "summary", "summary_buffer", "vector", "combined"],
        default="buffer",
        help="Memory type to use (default: buffer)"
    )

    # Ollama options
    parser.add_argument(
        "--model",
        default="qwen2.5:1.5b",
        help="Ollama model to use (default: qwen2.5:1.5b)"
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)"
    )

    # Azure options
    parser.add_argument("--api-key", help="Azure OpenAI API key")
    parser.add_argument("--endpoint", help="Azure OpenAI endpoint URL")
    parser.add_argument("--deployment", default="gpt-4", help="Azure deployment name")

    # Agent options
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Disable agent tools"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose agent output"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum agent iterations (default: 10)"
    )

    # General options
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8502,
        help="Streamlit port (default: 8502)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check configuration without starting the app"
    )

    return parser.parse_args()


def check_ollama(base_url: str) -> bool:
    """Check if Ollama is running."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def validate_azure_config(args) -> bool:
    """Validate Azure OpenAI configuration."""
    api_key = args.api_key or os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = args.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")

    if not api_key:
        print(colored("Error: Azure API key not provided", "red"))
        print("Set AZURE_OPENAI_API_KEY env var or use --api-key")
        return False

    if not endpoint:
        print(colored("Error: Azure endpoint not provided", "red"))
        print("Set AZURE_OPENAI_ENDPOINT env var or use --endpoint")
        return False

    return True


def set_environment(args):
    """Set environment variables from arguments."""
    os.environ["LLM_PROVIDER"] = args.provider
    os.environ["LLM_TEMPERATURE"] = str(args.temperature)
    os.environ["MEMORY_TYPE"] = args.memory
    os.environ["AGENT_VERBOSE"] = str(args.verbose).lower()
    os.environ["AGENT_MAX_ITERATIONS"] = str(args.max_iterations)

    if args.provider == "ollama":
        os.environ["OLLAMA_MODEL"] = args.model
        os.environ["OLLAMA_BASE_URL"] = args.ollama_url
    elif args.provider == "azure":
        if args.api_key:
            os.environ["AZURE_OPENAI_API_KEY"] = args.api_key
        if args.endpoint:
            os.environ["AZURE_OPENAI_ENDPOINT"] = args.endpoint
        os.environ["AZURE_OPENAI_DEPLOYMENT"] = args.deployment


def print_config(args):
    """Print current configuration."""
    print(colored("\n=== Agent Configuration ===", "cyan", attrs=["bold"]))
    print(f"Provider: {colored(args.provider, 'yellow')}")
    print(f"Memory Type: {colored(args.memory, 'yellow')}")
    print(f"Tools Enabled: {colored(str(not args.no_tools), 'yellow')}")
    print(f"Temperature: {colored(str(args.temperature), 'yellow')}")
    print(f"Max Iterations: {colored(str(args.max_iterations), 'yellow')}")
    print(f"Verbose: {colored(str(args.verbose), 'yellow')}")

    if args.provider == "ollama":
        print(f"Model: {colored(args.model, 'yellow')}")
        print(f"Ollama URL: {colored(args.ollama_url, 'yellow')}")
    elif args.provider == "azure":
        print(f"Deployment: {colored(args.deployment, 'yellow')}")
        endpoint = args.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "Not set")
        print(f"Endpoint: {colored(endpoint, 'yellow')}")

    print()


def main():
    """Main entry point."""
    args = parse_args()

    print(colored("\n" + "="*50, "magenta"))
    print(colored("  LangChain Agent Launcher", "magenta", attrs=["bold"]))
    print(colored("  Class 06: Memory, Tools & Agents", "magenta"))
    print(colored("="*50 + "\n", "magenta"))

    # Print configuration
    print_config(args)

    # Validate configuration
    if args.provider == "ollama":
        print("Checking Ollama connection...", end=" ")
        if check_ollama(args.ollama_url):
            print(colored("OK", "green"))
        else:
            print(colored("FAILED", "red"))
            print(f"\nMake sure Ollama is running at {args.ollama_url}")
            print("Start Ollama with: ollama serve")
            if not args.check:
                sys.exit(1)

    elif args.provider == "azure":
        print("Validating Azure configuration...", end=" ")
        if validate_azure_config(args):
            print(colored("OK", "green"))
        else:
            if not args.check:
                sys.exit(1)

    # Set environment variables
    set_environment(args)

    # Check only mode
    if args.check:
        print(colored("\nConfiguration check complete!", "green"))
        sys.exit(0)

    # Launch Streamlit app
    print(colored("\nStarting Agent application...", "green"))
    print(f"Access at: http://localhost:{args.port}\n")

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "agent_app.py",
            "--server.port", str(args.port),
            "--server.headless", "true",
        ], check=True)
    except KeyboardInterrupt:
        print(colored("\n\nApplication stopped.", "yellow"))
    except subprocess.CalledProcessError as e:
        print(colored(f"\nError running application: {e}", "red"))
        sys.exit(1)


if __name__ == "__main__":
    main()
