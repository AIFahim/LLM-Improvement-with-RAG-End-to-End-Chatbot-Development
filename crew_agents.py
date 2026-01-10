"""
Class 07: Multi-Agent Systems - CrewAI Agents

This module defines multi-role agents for collaborative report writing:
- Planner: Plans the report structure and research strategy
- Researcher: Gathers information and facts
- Writer: Writes the report content
- Critic: Reviews and provides feedback

Planner -> Executor -> Critic model implementation
"""

from crewai import Agent, LLM
from typing import Optional, List, Any
from config import (
    LLM_PROVIDER,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
)


def get_llm() -> LLM:
    """Get the LLM instance based on configuration."""
    if LLM_PROVIDER == "azure":
        return LLM(
            model=f"azure/{AZURE_OPENAI_DEPLOYMENT}",
            api_key=AZURE_OPENAI_API_KEY,
            base_url=AZURE_OPENAI_ENDPOINT,
        )
    else:
        # Ollama provider
        return LLM(
            model=f"ollama/{OLLAMA_MODEL}",
            base_url=OLLAMA_BASE_URL,
        )


def create_planner_agent(
    llm: Optional[LLM] = None,
    tools: Optional[List[Any]] = None,
    verbose: bool = True,
) -> Agent:
    """
    Create a Planner agent that designs report structure and research strategy.

    The Planner is responsible for:
    - Understanding the report topic and requirements
    - Creating an outline/structure for the report
    - Identifying key areas that need research
    - Coordinating the overall workflow
    """
    if llm is None:
        llm = get_llm()

    return Agent(
        role="Report Planner",
        goal="Create a comprehensive plan and outline for the report, identifying key sections, research areas, and the overall structure.",
        backstory="""You are an expert report planner with years of experience in
        structuring complex documents. You excel at breaking down topics into
        logical sections and identifying the key areas that need investigation.
        You think systematically and create clear, actionable plans for the team.""",
        llm=llm,
        tools=tools or [],
        verbose=verbose,
        allow_delegation=True,
        memory=False,  # Disabled - requires OpenAI API key
    )


def create_researcher_agent(
    llm: Optional[LLM] = None,
    tools: Optional[List[Any]] = None,
    verbose: bool = True,
) -> Agent:
    """
    Create a Researcher agent that gathers information and facts.

    The Researcher is responsible for:
    - Searching for relevant information
    - Gathering facts, statistics, and evidence
    - Synthesizing information from multiple sources
    - Providing well-researched content for the writer
    """
    if llm is None:
        llm = get_llm()

    return Agent(
        role="Research Analyst",
        goal="Conduct thorough research on the assigned topics, gathering accurate and relevant information, facts, statistics, and examples to support the report.",
        backstory="""You are a skilled research analyst with expertise in finding
        and synthesizing information from various sources. You have a keen eye for
        detail and can distinguish between reliable and unreliable sources.
        You provide comprehensive research that forms the foundation of quality reports.""",
        llm=llm,
        tools=tools or [],
        verbose=verbose,
        allow_delegation=False,
        memory=False,  # Disabled - requires OpenAI API key
    )


def create_writer_agent(
    llm: Optional[LLM] = None,
    tools: Optional[List[Any]] = None,
    verbose: bool = True,
) -> Agent:
    """
    Create a Writer agent that composes the report content.

    The Writer is responsible for:
    - Writing clear, engaging content
    - Following the report structure from the Planner
    - Incorporating research from the Researcher
    - Creating cohesive narratives
    """
    if llm is None:
        llm = get_llm()

    return Agent(
        role="Content Writer",
        goal="Write clear, engaging, and well-structured report content based on the research provided, following the outline from the planner.",
        backstory="""You are a professional content writer with expertise in
        creating compelling reports and documents. You have a talent for
        transforming complex information into clear, readable prose.
        You follow structure guidelines while maintaining engaging writing style.""",
        llm=llm,
        tools=tools or [],
        verbose=verbose,
        allow_delegation=False,
        memory=False,  # Disabled - requires OpenAI API key
    )


def create_critic_agent(
    llm: Optional[LLM] = None,
    tools: Optional[List[Any]] = None,
    verbose: bool = True,
) -> Agent:
    """
    Create a Critic agent that reviews and provides feedback.

    The Critic is responsible for:
    - Reviewing the report for quality and accuracy
    - Identifying areas for improvement
    - Checking for logical flow and coherence
    - Providing constructive feedback
    """
    if llm is None:
        llm = get_llm()

    return Agent(
        role="Quality Reviewer",
        goal="Review the report critically, checking for accuracy, clarity, logical flow, and completeness. Provide specific, actionable feedback for improvement.",
        backstory="""You are a meticulous quality reviewer with high standards for
        written content. You have a sharp eye for inconsistencies, errors, and
        areas that need improvement. You provide constructive criticism that
        helps elevate the quality of reports to professional standards.""",
        llm=llm,
        tools=tools or [],
        verbose=verbose,
        allow_delegation=True,
        memory=False,  # Disabled - requires OpenAI API key
    )


def create_summarizer_agent(
    llm: Optional[LLM] = None,
    tools: Optional[List[Any]] = None,
    verbose: bool = True,
) -> Agent:
    """
    Create a Summarizer agent that creates executive summaries.

    The Summarizer is responsible for:
    - Creating concise executive summaries
    - Highlighting key findings and conclusions
    - Distilling complex content into digestible formats
    """
    if llm is None:
        llm = get_llm()

    return Agent(
        role="Executive Summarizer",
        goal="Create a concise and impactful executive summary that captures the key findings, conclusions, and recommendations from the report.",
        backstory="""You are an expert at distilling complex information into
        clear, concise summaries. You understand what executives and decision-makers
        need to know and can present it in a compelling, easy-to-digest format.
        Your summaries save readers time while ensuring they grasp the essential points.""",
        llm=llm,
        tools=tools or [],
        verbose=verbose,
        allow_delegation=False,
        memory=False,  # Disabled - requires OpenAI API key
    )


class AgentFactory:
    """Factory class to create and manage CrewAI agents."""

    def __init__(
        self,
        llm: Optional[LLM] = None,
        tools: Optional[List[Any]] = None,
        verbose: bool = True,
    ):
        """
        Initialize the AgentFactory.

        Args:
            llm: Optional LLM instance (uses default if not provided)
            tools: Optional list of tools for agents
            verbose: Whether agents should log their actions
        """
        self.llm = llm or get_llm()
        self.tools = tools or []
        self.verbose = verbose

    def create_planner(self) -> Agent:
        """Create a Planner agent."""
        return create_planner_agent(self.llm, self.tools, self.verbose)

    def create_researcher(self) -> Agent:
        """Create a Researcher agent."""
        return create_researcher_agent(self.llm, self.tools, self.verbose)

    def create_writer(self) -> Agent:
        """Create a Writer agent."""
        return create_writer_agent(self.llm, self.tools, self.verbose)

    def create_critic(self) -> Agent:
        """Create a Critic agent."""
        return create_critic_agent(self.llm, self.tools, self.verbose)

    def create_summarizer(self) -> Agent:
        """Create a Summarizer agent."""
        return create_summarizer_agent(self.llm, self.tools, self.verbose)

    def create_all_agents(self) -> dict:
        """
        Create all agents for the report writing crew.

        Returns:
            Dictionary with agent names as keys and Agent instances as values
        """
        return {
            "planner": self.create_planner(),
            "researcher": self.create_researcher(),
            "writer": self.create_writer(),
            "critic": self.create_critic(),
            "summarizer": self.create_summarizer(),
        }

    def create_minimal_crew(self) -> dict:
        """
        Create a minimal crew with essential agents.

        Returns:
            Dictionary with planner, researcher, and writer agents
        """
        return {
            "planner": self.create_planner(),
            "researcher": self.create_researcher(),
            "writer": self.create_writer(),
        }


# Predefined agent configurations for different use cases
AGENT_CONFIGS = {
    "report_writing": {
        "agents": ["planner", "researcher", "writer", "critic", "summarizer"],
        "description": "Full team for comprehensive report writing",
    },
    "quick_report": {
        "agents": ["researcher", "writer"],
        "description": "Minimal team for quick reports",
    },
    "research_only": {
        "agents": ["planner", "researcher"],
        "description": "Team focused on research and planning",
    },
    "review_team": {
        "agents": ["writer", "critic"],
        "description": "Team for writing and reviewing content",
    },
}


def get_agent_info() -> dict:
    """
    Get information about available agents.

    Returns:
        Dictionary with agent roles, goals, and descriptions
    """
    return {
        "planner": {
            "role": "Report Planner",
            "description": "Designs report structure and coordinates workflow",
            "can_delegate": True,
        },
        "researcher": {
            "role": "Research Analyst",
            "description": "Gathers information, facts, and evidence",
            "can_delegate": False,
        },
        "writer": {
            "role": "Content Writer",
            "description": "Writes clear, engaging report content",
            "can_delegate": False,
        },
        "critic": {
            "role": "Quality Reviewer",
            "description": "Reviews and provides feedback for improvement",
            "can_delegate": True,
        },
        "summarizer": {
            "role": "Executive Summarizer",
            "description": "Creates concise executive summaries",
            "can_delegate": False,
        },
    }
