"""
Example Usage - LangChain Agent with Memory and Tools
Class 06: LangChain Deep Dive

This file demonstrates the key concepts:
1. LangChain Architecture: Chains, Tools, Agents, Memory
2. PromptTemplate, LLMChain, ConversationalChain
3. Types of Memory (short-term, long-term, vector memory)
4. Building Custom Tools
5. Tool Integration: Web search, calculator, Python functions
6. Execution & Feedback Loop

Run: python example_agent_usage.py
"""

import logging
from termcolor import colored

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_memory_types():
    """Demonstrate different memory types."""
    print(colored("\n" + "="*60, "cyan"))
    print(colored("DEMO 1: Memory Types", "cyan", attrs=["bold"]))
    print(colored("="*60, "cyan"))

    from memory_manager import MemoryManager, MemoryType

    # 1. Short-term Buffer Memory
    print(colored("\n1. Buffer Memory (Short-term):", "yellow"))
    buffer_memory = MemoryManager(memory_type=MemoryType.BUFFER)
    memory = buffer_memory.get_memory()

    # Simulate conversation
    memory.save_context({"input": "Hello, my name is Alice"}, {"output": "Hello Alice! Nice to meet you."})
    memory.save_context({"input": "What's the weather like?"}, {"output": "I don't have real-time weather data."})

    history = memory.load_memory_variables({})
    print(f"   Stored history: {history}")
    print(f"   Memory info: {buffer_memory.get_memory_info()}")

    # 2. Buffer Window Memory
    print(colored("\n2. Buffer Window Memory (Last K messages):", "yellow"))
    window_memory = MemoryManager(memory_type=MemoryType.BUFFER_WINDOW, k=2)
    memory = window_memory.get_memory()

    memory.save_context({"input": "Message 1"}, {"output": "Response 1"})
    memory.save_context({"input": "Message 2"}, {"output": "Response 2"})
    memory.save_context({"input": "Message 3"}, {"output": "Response 3"})

    history = memory.load_memory_variables({})
    print(f"   Only last 2 messages kept: {history}")

    # 3. Summary Memory (requires LLM)
    print(colored("\n3. Summary Memory (Long-term):", "yellow"))
    print("   Creates summaries of long conversations using LLM")
    print("   Best for very long conversations where full history is too large")

    # 4. Vector Memory
    print(colored("\n4. Vector Memory (Semantic Search):", "yellow"))
    print("   Stores conversation in vector database")
    print("   Retrieves relevant past conversations based on semantic similarity")

    # 5. Combined Memory
    print(colored("\n5. Combined Memory (Buffer + Vector):", "yellow"))
    print("   Combines recent buffer with semantic vector search")
    print("   Provides both recent context and relevant historical context")


def demo_tools():
    """Demonstrate custom tools."""
    print(colored("\n" + "="*60, "cyan"))
    print(colored("DEMO 2: Custom Tools", "cyan", attrs=["bold"]))
    print(colored("="*60, "cyan"))

    from tools import (
        CalculatorTool,
        DateTimeTool,
        WebSearchTool,
        SafePythonREPLTool,
        ToolFactory,
    )

    # 1. Calculator Tool
    print(colored("\n1. Calculator Tool:", "yellow"))
    calc = CalculatorTool()
    print(f"   Tool name: {calc.name}")
    print(f"   Description: {calc.description[:50]}...")

    # Test calculations
    expressions = ["2 + 2", "sqrt(16)", "sin(pi/2)", "10 ** 2"]
    for expr in expressions:
        result = calc._run(expr)
        print(f"   {expr} = {result}")

    # 2. DateTime Tool
    print(colored("\n2. DateTime Tool:", "yellow"))
    dt = DateTimeTool()
    operations = ["now", "date", "time", "weekday"]
    for op in operations:
        result = dt._run(op)
        print(f"   {op}: {result}")

    # 3. Web Search Tool
    print(colored("\n3. Web Search Tool:", "yellow"))
    print("   Uses DuckDuckGo for web searches")
    print("   No API key required!")
    # Uncomment to test (requires internet):
    # ws = WebSearchTool()
    # result = ws._run("latest Python version")
    # print(f"   Search result: {result[:200]}...")

    # 4. Python REPL Tool
    print(colored("\n4. Python REPL Tool:", "yellow"))
    repl = SafePythonREPLTool()
    code = "print(sum([1, 2, 3, 4, 5]))"
    result = repl._run(code)
    print(f"   Code: {code}")
    print(f"   Output: {result}")

    # 5. Tool Factory
    print(colored("\n5. Tool Factory:", "yellow"))
    factory = ToolFactory()
    tools = factory.get_all_tools()
    print(f"   Available tools: {[t.name for t in tools]}")


def demo_agent():
    """Demonstrate the conversational agent."""
    print(colored("\n" + "="*60, "cyan"))
    print(colored("DEMO 3: Conversational Agent with ReAct Pattern", "cyan", attrs=["bold"]))
    print(colored("="*60, "cyan"))

    from agent import LangChainAgent, ConversationalAgent, create_agent

    print(colored("\nCreating agent with tools and memory...", "yellow"))
    print("Note: This requires Ollama running locally or Azure OpenAI configured.\n")

    # Create agent using factory function
    agent = create_agent(
        memory_type="buffer",
        use_tools=True,
        vector_store=None,
    )

    print(f"Available tools: {agent.get_tools()}")
    print(f"Memory info: {agent.memory_info}")

    # Example queries (uncomment to test with running LLM)
    example_queries = [
        "What is 25 * 4?",
        "What time is it?",
        "Calculate the square root of 144",
        "What day of the week is it?",
    ]

    print(colored("\nExample queries to try:", "green"))
    for query in example_queries:
        print(f"  - {query}")

    print(colored("\nTo test the agent, uncomment the code below:", "yellow"))
    print("""
    # Test with actual LLM:
    response = agent.chat("What is 25 * 4?")
    print(f"Response: {response}")

    # Get full response with thinking steps:
    result = agent.get_full_response("What time is it right now?")
    print(f"Output: {result['output']}")
    print(f"Steps: {result['intermediate_steps']}")
    """)


def demo_chains():
    """Demonstrate LangChain chains."""
    print(colored("\n" + "="*60, "cyan"))
    print(colored("DEMO 4: LangChain Chains", "cyan", attrs=["bold"]))
    print(colored("="*60, "cyan"))

    print(colored("\n1. PromptTemplate:", "yellow"))
    from langchain_core.prompts import PromptTemplate

    template = PromptTemplate(
        input_variables=["topic"],
        template="Tell me a joke about {topic}."
    )
    print(f"   Template: {template.template}")
    print(f"   Formatted: {template.format(topic='programming')}")

    print(colored("\n2. LLMChain:", "yellow"))
    print("   Combines PromptTemplate + LLM for simple Q&A")
    print("   Example: template -> LLM -> response")

    print(colored("\n3. ConversationChain:", "yellow"))
    print("   LLMChain with memory for multi-turn conversations")
    print("   Maintains context across multiple interactions")

    print(colored("\n4. RetrievalQA Chain:", "yellow"))
    print("   Combines retriever + LLM for RAG applications")
    print("   Used in your existing RAG chatbot!")


def demo_react_pattern():
    """Explain the ReAct pattern."""
    print(colored("\n" + "="*60, "cyan"))
    print(colored("DEMO 5: ReAct Pattern (Reasoning + Acting)", "cyan", attrs=["bold"]))
    print(colored("="*60, "cyan"))

    print(colored("\nReAct Pattern Flow:", "yellow"))
    print("""
    1. QUESTION: User asks a question

    2. THOUGHT: Agent thinks about what to do
       "I need to calculate this math expression..."

    3. ACTION: Agent chooses a tool
       "calculator"

    4. ACTION INPUT: Agent provides tool input
       "25 * 4"

    5. OBSERVATION: Tool returns result
       "100"

    6. THOUGHT: Agent processes the result
       "I now know the answer..."

    7. FINAL ANSWER: Agent provides response
       "25 * 4 equals 100"
    """)

    print(colored("This loop can repeat multiple times for complex queries!", "green"))


def main():
    """Run all demos."""
    print(colored("\n" + "="*60, "magenta", attrs=["bold"]))
    print(colored("  LangChain Deep Dive - Class 06 Examples", "magenta", attrs=["bold"]))
    print(colored("  Memory, Tools & Agents", "magenta", attrs=["bold"]))
    print(colored("="*60 + "\n", "magenta", attrs=["bold"]))

    try:
        demo_memory_types()
        demo_tools()
        demo_chains()
        demo_react_pattern()
        demo_agent()

        print(colored("\n" + "="*60, "green"))
        print(colored("All demos completed successfully!", "green", attrs=["bold"]))
        print(colored("="*60 + "\n", "green"))

        print(colored("Next Steps:", "cyan"))
        print("1. Run the Streamlit app: streamlit run agent_app.py")
        print("2. Make sure Ollama is running: ollama serve")
        print("3. Try different memory types and tools in the UI")
        print("")

    except ImportError as e:
        print(colored(f"\nImport Error: {e}", "red"))
        print("Make sure to install dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(colored(f"\nError: {e}", "red"))
        logger.exception("Demo error")


if __name__ == "__main__":
    main()
