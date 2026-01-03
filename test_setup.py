#!/usr/bin/env python3
"""
Test script to verify all components are working
"""

import sys
import requests
from termcolor import colored

def test_imports():
    """Test if all modules can be imported"""
    print("Testing module imports...")
    try:
        import config
        import document_processor
        import vector_store
        import llm_handler
        import chatbot
        import utils
        print(colored("✓ All modules imported successfully", "green"))
        return True
    except ImportError as e:
        print(colored(f"✗ Import error: {e}", "red"))
        return False

def test_ollama():
    """Test if Ollama is running"""
    print("\nTesting Ollama connection...")
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(colored("✓ Ollama is running", "green"))
            if models:
                print("Available models:")
                for model in models:
                    print(f"  - {model['name']}")
            else:
                print(colored("⚠ No models found. Run: docker exec ollama ollama pull llama3.2:1b", "yellow"))
            return True
        else:
            print(colored("✗ Ollama returned error status", "red"))
            return False
    except requests.exceptions.ConnectionError:
        print(colored("✗ Cannot connect to Ollama at http://localhost:11434", "red"))
        print("Run: docker start ollama")
        return False

def test_directories():
    """Test if required directories exist"""
    print("\nChecking directories...")
    import os
    dirs_ok = True
    
    for dir_name in ['pdfFiles', 'vectorDB']:
        if os.path.exists(dir_name):
            print(colored(f"✓ Directory '{dir_name}' exists", "green"))
        else:
            print(colored(f"⚠ Directory '{dir_name}' will be created on first run", "yellow"))
    
    return dirs_ok

def main():
    """Run all tests"""
    print(colored("=== LLM RAG Chatbot Setup Test ===\n", "blue", attrs=['bold']))
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test Ollama
    if not test_ollama():
        all_tests_passed = False
    
    # Test directories
    if not test_directories():
        all_tests_passed = False
    
    print("\n" + "="*40)
    if all_tests_passed:
        print(colored("✓ All tests passed! You can run: streamlit run app.py", "green", attrs=['bold']))
    else:
        print(colored("✗ Some tests failed. Please fix the issues above.", "red", attrs=['bold']))
        sys.exit(1)

if __name__ == "__main__":
    # Install termcolor if not available
    try:
        from termcolor import colored
    except ImportError:
        print("Installing termcolor...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "termcolor"])
        from termcolor import colored
    
    main()