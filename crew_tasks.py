"""
Class 07: Multi-Agent Systems - CrewAI Tasks

This module defines tasks for the collaborative report writing workflow:
1. Planning Task: Create report structure and outline
2. Research Task: Gather information and facts
3. Writing Task: Compose the report content
4. Review Task: Critique and provide feedback
5. Summary Task: Create executive summary

Tasks follow the Planner -> Executor -> Critic model
"""

from crewai import Task, Agent
from typing import Optional, List


def create_planning_task(
    agent: Agent,
    topic: str,
    requirements: Optional[str] = None,
    context: Optional[List[Task]] = None,
) -> Task:
    """
    Create a planning task for the report.

    Args:
        agent: The Planner agent to assign this task
        topic: The main topic of the report
        requirements: Additional requirements or constraints
        context: Previous tasks to use as context

    Returns:
        Task instance for planning the report
    """
    description = f"""
    Create a comprehensive plan and outline for a report on the following topic:

    TOPIC: {topic}

    {"REQUIREMENTS: " + requirements if requirements else ""}

    Your task is to:
    1. Analyze the topic and identify the key aspects to cover
    2. Create a detailed outline with sections and subsections
    3. Identify specific questions that need to be researched
    4. Suggest the approximate length and depth for each section
    5. Highlight any potential challenges or areas requiring special attention

    Provide a structured plan that the research and writing team can follow.
    """

    expected_output = """
    A detailed report plan including:
    1. Executive Summary section description
    2. Main sections with subsections (minimum 3-5 main sections)
    3. Key research questions for each section
    4. Recommended sources or types of information needed
    5. Suggested word count per section
    6. Notes on tone, style, and target audience
    """

    return Task(
        description=description.strip(),
        expected_output=expected_output.strip(),
        agent=agent,
        context=context,
    )


def create_research_task(
    agent: Agent,
    topic: str,
    research_areas: Optional[str] = None,
    context: Optional[List[Task]] = None,
) -> Task:
    """
    Create a research task to gather information.

    Args:
        agent: The Researcher agent to assign this task
        topic: The main topic to research
        research_areas: Specific areas to focus research on
        context: Previous tasks (like planning) to use as context

    Returns:
        Task instance for conducting research
    """
    description = f"""
    Conduct thorough research on the following topic:

    TOPIC: {topic}

    {"FOCUS AREAS: " + research_areas if research_areas else ""}

    Your task is to:
    1. Gather relevant facts, statistics, and information
    2. Find examples and case studies if applicable
    3. Identify different perspectives or viewpoints on the topic
    4. Note any controversies or debates in the field
    5. Compile sources and references

    If a plan was provided, follow the research questions outlined there.
    Focus on accuracy and relevance of information.
    """

    expected_output = """
    Comprehensive research findings including:
    1. Key facts and statistics with sources
    2. Important concepts and definitions
    3. Relevant examples or case studies
    4. Different perspectives on the topic
    5. Recent developments or trends
    6. List of references and sources used
    """

    return Task(
        description=description.strip(),
        expected_output=expected_output.strip(),
        agent=agent,
        context=context,
    )


def create_writing_task(
    agent: Agent,
    topic: str,
    style: str = "professional",
    word_count: Optional[int] = None,
    context: Optional[List[Task]] = None,
) -> Task:
    """
    Create a writing task to compose the report.

    Args:
        agent: The Writer agent to assign this task
        topic: The main topic of the report
        style: Writing style (professional, academic, casual)
        word_count: Target word count for the report
        context: Previous tasks (planning, research) to use as context

    Returns:
        Task instance for writing the report
    """
    word_guidance = f"Target approximately {word_count} words." if word_count else "Write a comprehensive report."

    description = f"""
    Write a complete report on the following topic:

    TOPIC: {topic}

    STYLE: {style}
    {word_guidance}

    Your task is to:
    1. Follow the outline provided by the planner
    2. Incorporate the research findings into the content
    3. Write clear, engaging, and well-structured prose
    4. Ensure logical flow between sections
    5. Include an introduction and conclusion
    6. Use appropriate headings and formatting

    Create a cohesive, readable report that effectively communicates the information.
    """

    expected_output = """
    A complete, well-structured report including:
    1. Title
    2. Introduction that sets context and objectives
    3. Main body sections following the outline
    4. Supporting evidence and examples from research
    5. Conclusion with key takeaways
    6. References section if applicable

    The report should be engaging, clear, and professionally written.
    """

    return Task(
        description=description.strip(),
        expected_output=expected_output.strip(),
        agent=agent,
        context=context,
    )


def create_review_task(
    agent: Agent,
    focus_areas: Optional[str] = None,
    context: Optional[List[Task]] = None,
) -> Task:
    """
    Create a review task to critique the report.

    Args:
        agent: The Critic agent to assign this task
        focus_areas: Specific areas to focus the review on
        context: Previous tasks (including the written report) to review

    Returns:
        Task instance for reviewing the report
    """
    description = f"""
    Review the report and provide detailed feedback.

    {"FOCUS AREAS: " + focus_areas if focus_areas else ""}

    Your task is to evaluate:
    1. ACCURACY: Are the facts and information correct?
    2. CLARITY: Is the writing clear and easy to understand?
    3. STRUCTURE: Is the report well-organized and logical?
    4. COMPLETENESS: Are all necessary topics covered?
    5. ENGAGEMENT: Is the writing engaging and appropriate for the audience?
    6. FORMATTING: Are headings, paragraphs, and sections well-formatted?

    Provide specific, actionable feedback for improvement.
    Be constructive but thorough in your critique.
    """

    expected_output = """
    A detailed review report including:
    1. Overall assessment (score or rating)
    2. Strengths of the report
    3. Areas for improvement with specific suggestions
    4. Any factual errors or inconsistencies found
    5. Suggestions for better flow or organization
    6. Final recommendations

    The review should be constructive and help improve the report quality.
    """

    return Task(
        description=description.strip(),
        expected_output=expected_output.strip(),
        agent=agent,
        context=context,
    )


def create_summary_task(
    agent: Agent,
    max_length: Optional[int] = None,
    context: Optional[List[Task]] = None,
) -> Task:
    """
    Create a summary task for executive summary.

    Args:
        agent: The Summarizer agent to assign this task
        max_length: Maximum length in words for the summary
        context: Previous tasks (the full report) to summarize

    Returns:
        Task instance for creating the summary
    """
    length_guidance = f"Keep the summary under {max_length} words." if max_length else "Keep the summary concise (200-300 words)."

    description = f"""
    Create an executive summary of the report.

    {length_guidance}

    Your task is to:
    1. Capture the key findings and insights
    2. Highlight the most important conclusions
    3. Include any critical recommendations
    4. Make it accessible to readers who won't read the full report
    5. Maintain the essence and impact of the original content

    The summary should stand alone and give readers a complete overview.
    """

    expected_output = """
    An executive summary including:
    1. Brief context/background (1-2 sentences)
    2. Key findings (3-5 bullet points)
    3. Main conclusions
    4. Recommendations (if applicable)
    5. Call to action or next steps (if applicable)

    The summary should be impactful and capture the report's essence.
    """

    return Task(
        description=description.strip(),
        expected_output=expected_output.strip(),
        agent=agent,
        context=context,
    )


def create_revision_task(
    agent: Agent,
    feedback: str,
    context: Optional[List[Task]] = None,
) -> Task:
    """
    Create a revision task based on feedback.

    Args:
        agent: The Writer agent to assign this task
        feedback: The feedback to address in the revision
        context: Previous tasks including original report and review

    Returns:
        Task instance for revising the report
    """
    description = f"""
    Revise the report based on the following feedback:

    FEEDBACK:
    {feedback}

    Your task is to:
    1. Address each point in the feedback
    2. Improve clarity and flow where suggested
    3. Fix any errors or inconsistencies identified
    4. Enhance sections that were marked as weak
    5. Maintain the overall structure while making improvements

    Create an improved version of the report that addresses all feedback.
    """

    expected_output = """
    A revised report that:
    1. Addresses all feedback points
    2. Shows clear improvements from the original
    3. Maintains good structure and flow
    4. Is ready for final review or publication

    Include a brief note on what changes were made.
    """

    return Task(
        description=description.strip(),
        expected_output=expected_output.strip(),
        agent=agent,
        context=context,
    )


class TaskFactory:
    """Factory class to create and manage CrewAI tasks."""

    def __init__(self, agents: dict):
        """
        Initialize the TaskFactory.

        Args:
            agents: Dictionary of agents (from AgentFactory.create_all_agents())
        """
        self.agents = agents

    def create_report_workflow(
        self,
        topic: str,
        requirements: Optional[str] = None,
        style: str = "professional",
        word_count: Optional[int] = None,
        include_review: bool = True,
        include_summary: bool = True,
    ) -> List[Task]:
        """
        Create a complete report writing workflow.

        Args:
            topic: The report topic
            requirements: Additional requirements
            style: Writing style
            word_count: Target word count
            include_review: Whether to include review task
            include_summary: Whether to include summary task

        Returns:
            List of tasks in execution order
        """
        tasks = []

        # 1. Planning Task
        if "planner" in self.agents:
            planning_task = create_planning_task(
                agent=self.agents["planner"],
                topic=topic,
                requirements=requirements,
            )
            tasks.append(planning_task)

        # 2. Research Task
        if "researcher" in self.agents:
            research_task = create_research_task(
                agent=self.agents["researcher"],
                topic=topic,
                context=[tasks[-1]] if tasks else None,
            )
            tasks.append(research_task)

        # 3. Writing Task
        if "writer" in self.agents:
            writing_task = create_writing_task(
                agent=self.agents["writer"],
                topic=topic,
                style=style,
                word_count=word_count,
                context=tasks.copy() if tasks else None,
            )
            tasks.append(writing_task)

        # 4. Review Task (optional)
        if include_review and "critic" in self.agents:
            review_task = create_review_task(
                agent=self.agents["critic"],
                context=[tasks[-1]] if tasks else None,
            )
            tasks.append(review_task)

        # 5. Summary Task (optional)
        if include_summary and "summarizer" in self.agents:
            # Use the writing task as context (not the review)
            writing_context = [t for t in tasks if "write" in t.description.lower()]
            summary_task = create_summary_task(
                agent=self.agents["summarizer"],
                context=writing_context if writing_context else None,
            )
            tasks.append(summary_task)

        return tasks

    def create_quick_report_workflow(
        self,
        topic: str,
        style: str = "professional",
    ) -> List[Task]:
        """
        Create a minimal report workflow (research + write only).

        Args:
            topic: The report topic
            style: Writing style

        Returns:
            List of tasks for quick report
        """
        tasks = []

        if "researcher" in self.agents:
            research_task = create_research_task(
                agent=self.agents["researcher"],
                topic=topic,
            )
            tasks.append(research_task)

        if "writer" in self.agents:
            writing_task = create_writing_task(
                agent=self.agents["writer"],
                topic=topic,
                style=style,
                context=[tasks[-1]] if tasks else None,
            )
            tasks.append(writing_task)

        return tasks

    def create_research_workflow(
        self,
        topic: str,
        research_areas: Optional[str] = None,
    ) -> List[Task]:
        """
        Create a research-focused workflow.

        Args:
            topic: The research topic
            research_areas: Specific areas to focus on

        Returns:
            List of tasks for research workflow
        """
        tasks = []

        if "planner" in self.agents:
            planning_task = create_planning_task(
                agent=self.agents["planner"],
                topic=topic,
            )
            tasks.append(planning_task)

        if "researcher" in self.agents:
            research_task = create_research_task(
                agent=self.agents["researcher"],
                topic=topic,
                research_areas=research_areas,
                context=[tasks[-1]] if tasks else None,
            )
            tasks.append(research_task)

        return tasks


# Predefined workflow templates
WORKFLOW_TEMPLATES = {
    "full_report": {
        "name": "Full Report Workflow",
        "description": "Complete workflow: Plan -> Research -> Write -> Review -> Summary",
        "tasks": ["planning", "research", "writing", "review", "summary"],
    },
    "quick_report": {
        "name": "Quick Report",
        "description": "Minimal workflow: Research -> Write",
        "tasks": ["research", "writing"],
    },
    "research_plan": {
        "name": "Research Plan",
        "description": "Planning and research only",
        "tasks": ["planning", "research"],
    },
    "write_review": {
        "name": "Write and Review",
        "description": "Writing with review feedback",
        "tasks": ["writing", "review"],
    },
}


def get_workflow_info() -> dict:
    """
    Get information about available workflows.

    Returns:
        Dictionary with workflow templates information
    """
    return WORKFLOW_TEMPLATES
