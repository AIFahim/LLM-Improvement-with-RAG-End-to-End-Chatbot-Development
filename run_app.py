#!/usr/bin/env python3
"""
RAG Chatbot Launcher with argparse support
Run with: python run_app.py --provider ollama
          python run_app.py --provider azure --deployment gpt-4
"""
import argparse
import os
import sys
import subprocess

# Load .env file if exists
from dotenv import load_dotenv
load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(
        description="RAG Chatbot - Launch with Ollama or Azure OpenAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with Ollama (default)
  python run_app.py
  python run_app.py --provider ollama --model qwen2.5:1.5b

  # Run with Azure OpenAI
  python run_app.py --provider azure --api-key YOUR_KEY --endpoint https://xxx.openai.azure.com/ --deployment gpt-4

  # Run with Azure using environment variables
  export AZURE_OPENAI_API_KEY=your-key
  export AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com/
  python run_app.py --provider azure
        """
    )

    # Provider selection
    parser.add_argument(
        "--provider", "-p",
        choices=["ollama", "azure"],
        default="ollama",
        help="LLM provider to use (default: ollama)"
    )

    # Ollama options
    ollama_group = parser.add_argument_group("Ollama Options")
    ollama_group.add_argument(
        "--model", "-m",
        default="qwen2.5:1.5b",
        help="Ollama model name (default: qwen2.5:1.5b)"
    )
    ollama_group.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)"
    )

    # Azure options
    azure_group = parser.add_argument_group("Azure OpenAI Options")
    azure_group.add_argument(
        "--api-key",
        default=os.getenv("AZURE_OPENAI_API_KEY", ""),
        help="Azure OpenAI API key (or set AZURE_OPENAI_API_KEY env var)"
    )
    azure_group.add_argument(
        "--endpoint",
        default=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        help="Azure OpenAI endpoint URL (or set AZURE_OPENAI_ENDPOINT env var)"
    )
    azure_group.add_argument(
        "--deployment", "-d",
        default=os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("AZURE_LLM_DEPLOYMENT_NAME", "gpt-4"),
        help="Azure OpenAI deployment name (default: gpt-4)"
    )
    azure_group.add_argument(
        "--api-version",
        default="2024-02-15-preview",
        help="Azure OpenAI API version (default: 2024-02-15-preview)"
    )

    # Common options
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Streamlit port (default: 8501)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check configuration, don't start the app"
    )

    return parser.parse_args()


def validate_azure_config(args):
    """Validate Azure configuration"""
    if not args.api_key:
        print("ERROR: Azure API key not provided.")
        print("Use --api-key or set AZURE_OPENAI_API_KEY environment variable")
        return False
    if not args.endpoint:
        print("ERROR: Azure endpoint not provided.")
        print("Use --endpoint or set AZURE_OPENAI_ENDPOINT environment variable")
        return False
    return True


def check_ollama():
    """Check if Ollama is accessible"""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def set_environment(args):
    """Set environment variables based on arguments"""
    os.environ["LLM_PROVIDER"] = args.provider
    os.environ["LLM_TEMPERATURE"] = str(args.temperature)

    if args.provider == "ollama":
        os.environ["OLLAMA_MODEL"] = args.model
        os.environ["OLLAMA_BASE_URL"] = args.ollama_url
    else:
        os.environ["AZURE_OPENAI_API_KEY"] = args.api_key
        os.environ["AZURE_OPENAI_ENDPOINT"] = args.endpoint
        os.environ["AZURE_OPENAI_DEPLOYMENT"] = args.deployment
        os.environ["AZURE_OPENAI_API_VERSION"] = args.api_version


def print_config(args):
    """Print current configuration"""
    print("\n" + "=" * 50)
    print("RAG Chatbot Configuration")
    print("=" * 50)
    print(f"Provider:    {args.provider.upper()}")
    print(f"Temperature: {args.temperature}")

    if args.provider == "ollama":
        print(f"Model:       {args.model}")
        print(f"Ollama URL:  {args.ollama_url}")
        ollama_ok = check_ollama()
        print(f"Ollama Status: {'✓ Running' if ollama_ok else '✗ Not accessible'}")
    else:
        print(f"Deployment:  {args.deployment}")
        print(f"Endpoint:    {args.endpoint}")
        print(f"API Version: {args.api_version}")
        print(f"API Key:     {'✓ Set' if args.api_key else '✗ Not set'}")

    print("=" * 50 + "\n")


def main():
    args = parse_args()

    # Validate Azure config if selected
    if args.provider == "azure":
        if not validate_azure_config(args):
            sys.exit(1)

    # Set environment variables
    set_environment(args)

    # Print configuration
    print_config(args)

    # If check mode, exit here
    if args.check:
        print("Configuration check complete.")
        if args.provider == "ollama" and not check_ollama():
            print("\nWARNING: Ollama is not running!")
            print("Start it with: docker start ollama")
            sys.exit(1)
        sys.exit(0)

    # Check Ollama if using it
    if args.provider == "ollama" and not check_ollama():
        print("ERROR: Ollama is not running!")
        print("Start it with: docker start ollama")
        sys.exit(1)

    # Launch Streamlit
    print(f"Starting Streamlit on port {args.port}...")
    print(f"Open: http://localhost:{args.port}")
    print("-" * 50)

    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", str(args.port),
        "--server.headless", "true"
    ])


if __name__ == "__main__":
    main()
