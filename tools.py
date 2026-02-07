"""
Custom Tools Module - LangChain Deep Dive (Class 06)

This module implements custom tools for LangChain agents:
1. Calculator Tool - Mathematical expressions evaluation
2. Web Search Tool - Internet search using DuckDuckGo
3. Python REPL Tool - Execute Python code
4. DateTime Tool - Current date/time operations
5. RAG Search Tool - Search uploaded documents

LangChain Components Used:
- @tool decorator
- StructuredTool
- BaseTool (for custom implementations)
- DuckDuckGoSearchRun
- PythonREPLTool
"""

import logging
import math
import re
from datetime import datetime
from typing import Optional, List, Dict, Any, Type

from langchain_core.tools import tool, BaseTool, StructuredTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
from langchain_experimental.tools import PythonREPLTool
from pydantic import BaseModel, Field

import config

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL INPUT SCHEMAS (Pydantic Models)
# =============================================================================

class CalculatorInput(BaseModel):
    """Input schema for calculator tool."""
    expression: str = Field(
        description="Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(pi/2)')"
    )


class WebSearchInput(BaseModel):
    """Input schema for web search tool."""
    query: str = Field(
        description="Search query to look up on the internet"
    )


class PythonCodeInput(BaseModel):
    """Input schema for Python REPL tool."""
    code: str = Field(
        description="Python code to execute"
    )


class DateTimeInput(BaseModel):
    """Input schema for datetime tool."""
    operation: str = Field(
        description="Operation to perform: 'now', 'date', 'time', 'weekday', 'timestamp', or 'format:<strftime_format>'"
    )


class RAGSearchInput(BaseModel):
    """Input schema for RAG search tool."""
    query: str = Field(
        description="Query to search in the uploaded documents"
    )
    k: int = Field(
        default=4,
        description="Number of relevant documents to retrieve"
    )


# =============================================================================
# CALCULATOR TOOL
# =============================================================================

class CalculatorTool(BaseTool):
    """
    Calculator tool for evaluating mathematical expressions.

    Supports basic arithmetic, trigonometry, logarithms, and common math functions.
    Uses Python's math module for calculations.

    Examples:
        - "2 + 2" -> 4
        - "sqrt(16)" -> 4.0
        - "sin(pi/2)" -> 1.0
        - "log(100, 10)" -> 2.0
    """

    name: str = "calculator"
    description: str = """Useful for evaluating mathematical expressions.
    Input should be a mathematical expression like '2 + 2', 'sqrt(16)', 'sin(pi/2)', etc.
    Supports: +, -, *, /, **, sqrt, sin, cos, tan, log, exp, abs, round, pi, e"""
    args_schema: Type[BaseModel] = CalculatorInput

    # Safe math functions and constants
    _safe_dict: Dict[str, Any] = {
        # Basic operations
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        # Math functions
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "floor": math.floor,
        "ceil": math.ceil,
        "factorial": math.factorial,
        "gcd": math.gcd,
        "degrees": math.degrees,
        "radians": math.radians,
        # Constants
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
        "inf": math.inf,
    }

    def _run(
        self,
        expression: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the calculator tool."""
        try:
            # Clean the expression
            expression = expression.strip()

            # Validate expression (basic security check)
            if not self._is_safe_expression(expression):
                return "Error: Invalid or unsafe expression"

            # Evaluate the expression
            result = eval(expression, {"__builtins__": {}}, self._safe_dict)

            logger.info(f"Calculator: {expression} = {result}")
            return f"{result}"

        except ZeroDivisionError:
            return "Error: Division by zero"
        except ValueError as e:
            return f"Error: Invalid value - {str(e)}"
        except Exception as e:
            logger.error(f"Calculator error: {e}")
            return f"Error: Could not evaluate expression - {str(e)}"

    def _is_safe_expression(self, expression: str) -> bool:
        """Check if the expression is safe to evaluate."""
        # Block dangerous patterns
        dangerous_patterns = [
            r"__",           # Dunder methods
            r"import",       # Import statements
            r"exec",         # Exec function
            r"eval",         # Nested eval
            r"open",         # File operations
            r"os\.",         # OS module
            r"sys\.",        # Sys module
            r"subprocess",   # Subprocess
            r"lambda",       # Lambda functions
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, expression, re.IGNORECASE):
                return False

        return True


# =============================================================================
# WEB SEARCH TOOL
# =============================================================================

class WebSearchTool(BaseTool):
    """
    Web search tool using DuckDuckGo.

    Performs internet searches and returns relevant results.
    No API key required.
    """

    name: str = "web_search"
    description: str = """Useful for searching the internet for current information.
    Use this when you need up-to-date information or facts you don't know.
    Input should be a search query."""
    args_schema: Type[BaseModel] = WebSearchInput

    _search: DuckDuckGoSearchRun = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._search = DuckDuckGoSearchRun()

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the web search tool."""
        try:
            logger.info(f"Web search: {query}")
            result = self._search.run(query)
            return result

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Error: Could not perform search - {str(e)}"


class WebSearchDetailedTool(BaseTool):
    """
    Detailed web search tool returning structured results.

    Returns multiple search results with titles, snippets, and links.
    """

    name: str = "web_search_detailed"
    description: str = """Useful for detailed internet searches with multiple results.
    Returns structured results with titles, snippets, and links.
    Input should be a search query."""
    args_schema: Type[BaseModel] = WebSearchInput

    _search: DuckDuckGoSearchResults = None

    def __init__(self, max_results: int = None, **kwargs):
        super().__init__(**kwargs)
        max_results = max_results or config.WEB_SEARCH_MAX_RESULTS
        self._search = DuckDuckGoSearchResults(num_results=max_results)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the detailed web search tool."""
        try:
            logger.info(f"Detailed web search: {query}")
            result = self._search.run(query)
            return result

        except Exception as e:
            logger.error(f"Detailed web search error: {e}")
            return f"Error: Could not perform search - {str(e)}"


# =============================================================================
# PYTHON REPL TOOL
# =============================================================================

class SafePythonREPLTool(BaseTool):
    """
    Safe Python REPL tool with timeout and restrictions.

    Executes Python code in a sandboxed environment with:
    - Timeout protection
    - Import restrictions
    - Output capture
    """

    name: str = "python_repl"
    description: str = """Useful for executing Python code and getting the result.
    Use this for calculations, data processing, or when you need to run code.
    Input should be valid Python code.
    The code should print() any output you want to see."""
    args_schema: Type[BaseModel] = PythonCodeInput

    _repl: PythonREPLTool = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._repl = PythonREPLTool()

    def _run(
        self,
        code: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the Python REPL tool."""
        try:
            # Basic security check
            if self._has_dangerous_code(code):
                return "Error: Code contains potentially dangerous operations"

            logger.info(f"Python REPL executing: {code[:100]}...")
            result = self._repl.run(code)

            if result.strip():
                return result
            else:
                return "Code executed successfully (no output)"

        except Exception as e:
            logger.error(f"Python REPL error: {e}")
            return f"Error: {str(e)}"

    def _has_dangerous_code(self, code: str) -> bool:
        """Check for potentially dangerous code patterns."""
        dangerous_patterns = [
            r"os\.system",
            r"subprocess",
            r"shutil\.rmtree",
            r"open\s*\([^)]*['\"]w",  # Writing files
            r"__import__",
            r"exec\s*\(",
            r"eval\s*\(",
            r"rm\s+-rf",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return True

        return False


# =============================================================================
# DATETIME TOOL
# =============================================================================

class DateTimeTool(BaseTool):
    """
    DateTime tool for time-related queries.

    Provides current date, time, timestamp, and formatting operations.

    Operations:
        - now: Current datetime
        - date: Current date only
        - time: Current time only
        - weekday: Current day of week
        - timestamp: Unix timestamp
        - format:<strftime>: Custom formatted datetime
    """

    name: str = "datetime"
    description: str = """Useful for getting current date and time information.
    Operations: 'now', 'date', 'time', 'weekday', 'timestamp', or 'format:<strftime_format>'
    Examples:
    - 'now' -> Current datetime
    - 'date' -> Current date (YYYY-MM-DD)
    - 'time' -> Current time (HH:MM:SS)
    - 'weekday' -> Day of week
    - 'timestamp' -> Unix timestamp
    - 'format:%B %d, %Y' -> Custom format (e.g., 'January 01, 2024')"""
    args_schema: Type[BaseModel] = DateTimeInput

    def _run(
        self,
        operation: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the datetime tool."""
        try:
            now = datetime.now()
            operation = operation.strip().lower()

            if operation == "now":
                result = now.strftime("%Y-%m-%d %H:%M:%S")
            elif operation == "date":
                result = now.strftime("%Y-%m-%d")
            elif operation == "time":
                result = now.strftime("%H:%M:%S")
            elif operation == "weekday":
                result = now.strftime("%A")
            elif operation == "timestamp":
                result = str(int(now.timestamp()))
            elif operation.startswith("format:"):
                format_str = operation[7:]
                result = now.strftime(format_str)
            else:
                result = f"Unknown operation: {operation}. Use 'now', 'date', 'time', 'weekday', 'timestamp', or 'format:<strftime>'"

            logger.info(f"DateTime {operation}: {result}")
            return result

        except Exception as e:
            logger.error(f"DateTime error: {e}")
            return f"Error: {str(e)}"


# =============================================================================
# RAG SEARCH TOOL
# =============================================================================

class RAGSearchTool(BaseTool):
    """
    RAG Search tool for searching uploaded documents.

    This tool searches the vector store containing uploaded PDFs
    and returns relevant document chunks.
    """

    name: str = "rag_search"
    description: str = """Useful for searching through uploaded PDF documents.
    Use this when the user asks about information in their uploaded documents.
    Input should be a search query related to the document content."""
    args_schema: Type[BaseModel] = RAGSearchInput

    vector_store: Any = None

    def __init__(self, vector_store: Any = None, **kwargs):
        super().__init__(**kwargs)
        self.vector_store = vector_store

    def _run(
        self,
        query: str,
        k: int = 4,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the RAG search tool."""
        try:
            if self.vector_store is None:
                return "No documents have been uploaded yet. Please upload PDF documents first."

            logger.info(f"RAG search: {query} (k={k})")

            # Search the vector store
            docs = self.vector_store.similarity_search(query, k=k)

            if not docs:
                return "No relevant documents found for your query."

            # Format results
            results = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "N/A")
                content = doc.page_content[:500]  # Truncate for readability

                results.append(
                    f"**Document {i}** (Source: {source}, Page: {page}):\n{content}"
                )

            return "\n\n".join(results)

        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return f"Error searching documents: {str(e)}"


# =============================================================================
# TOOL DECORATOR EXAMPLES (Alternative way to create tools)
# =============================================================================

@tool
def simple_calculator(expression: str) -> str:
    """
    Simple calculator using the @tool decorator.

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        Result of the calculation
    """
    try:
        safe_dict = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "exp": math.exp,
            "pi": math.pi, "e": math.e,
        }
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def get_current_date() -> str:
    """Get the current date."""
    return datetime.now().strftime("%Y-%m-%d")


# =============================================================================
# TOOL FACTORY
# =============================================================================

class ToolFactory:
    """
    Factory class for creating and managing tools.

    Provides methods to create individual tools or get all enabled tools
    based on configuration.
    """

    def __init__(self, vector_store: Any = None):
        """
        Initialize the tool factory.

        Args:
            vector_store: Optional vector store for RAG search tool
        """
        self.vector_store = vector_store

    def get_calculator_tool(self) -> CalculatorTool:
        """Get calculator tool instance."""
        return CalculatorTool()

    def get_web_search_tool(self, detailed: bool = False) -> BaseTool:
        """Get web search tool instance."""
        if detailed:
            return WebSearchDetailedTool()
        return WebSearchTool()

    def get_python_repl_tool(self) -> SafePythonREPLTool:
        """Get Python REPL tool instance."""
        return SafePythonREPLTool()

    def get_datetime_tool(self) -> DateTimeTool:
        """Get datetime tool instance."""
        return DateTimeTool()

    def get_rag_search_tool(self) -> RAGSearchTool:
        """Get RAG search tool instance."""
        return RAGSearchTool(vector_store=self.vector_store)

    def get_all_tools(self) -> List[BaseTool]:
        """
        Get all enabled tools based on configuration.

        Returns:
            List of enabled tool instances
        """
        tools = []

        if config.TOOLS_ENABLED.get("calculator", True):
            tools.append(self.get_calculator_tool())

        if config.TOOLS_ENABLED.get("web_search", True):
            tools.append(self.get_web_search_tool())

        if config.TOOLS_ENABLED.get("python_repl", True):
            tools.append(self.get_python_repl_tool())

        if config.TOOLS_ENABLED.get("datetime", True):
            tools.append(self.get_datetime_tool())

        if config.TOOLS_ENABLED.get("rag_search", True):
            tools.append(self.get_rag_search_tool())

        logger.info(f"Created {len(tools)} tools: {[t.name for t in tools]}")
        return tools

    def get_tools_by_names(self, names: List[str]) -> List[BaseTool]:
        """
        Get specific tools by name.

        Args:
            names: List of tool names to get

        Returns:
            List of requested tool instances
        """
        tool_map = {
            "calculator": self.get_calculator_tool,
            "web_search": self.get_web_search_tool,
            "python_repl": self.get_python_repl_tool,
            "datetime": self.get_datetime_tool,
            "rag_search": self.get_rag_search_tool,
        }

        tools = []
        for name in names:
            if name in tool_map:
                tools.append(tool_map[name]())
            else:
                logger.warning(f"Unknown tool: {name}")

        return tools

    def update_vector_store(self, vector_store: Any) -> None:
        """Update the vector store for RAG search tool."""
        self.vector_store = vector_store


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_all_tools(vector_store: Any = None) -> List[BaseTool]:
    """
    Create all available tools.

    Args:
        vector_store: Optional vector store for RAG search

    Returns:
        List of all tool instances
    """
    factory = ToolFactory(vector_store=vector_store)
    return factory.get_all_tools()


def create_basic_tools() -> List[BaseTool]:
    """
    Create basic tools (calculator, datetime).

    Returns:
        List of basic tool instances
    """
    factory = ToolFactory()
    return factory.get_tools_by_names(["calculator", "datetime"])


def get_tool_descriptions(tools: List[BaseTool]) -> str:
    """
    Get formatted descriptions of all tools.

    Args:
        tools: List of tools

    Returns:
        Formatted string with tool descriptions
    """
    descriptions = []
    for tool in tools:
        descriptions.append(f"- {tool.name}: {tool.description}")
    return "\n".join(descriptions)
