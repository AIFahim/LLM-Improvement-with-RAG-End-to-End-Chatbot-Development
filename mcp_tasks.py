"""
MCP Chatbot — Task templates.

One place for the description / expected_output strings used by each
Crew. Three tasks:

- Chat task: the per-turn user reply
- Running summary task: rolls the conversation summary forward
- Recap task: the on-demand 'Summarize' button output
"""

from crewai import Agent, Task


def create_chat_task(
    agent: Agent, user_message: str, running_summary: str
) -> Task:
    """Per-turn task. The running_summary supplies conversation context."""
    description = (
        f"Summary of the conversation so far:\n{running_summary}\n\n"
        if running_summary else ""
    ) + f"User: {user_message}\nAssistant:"
    return Task(
        description=description.strip(),
        expected_output="A direct, conversational reply to the user.",
        agent=agent,
    )


def create_running_summary_task(
    agent: Agent,
    prev_summary: str,
    user_msg: str,
    assistant_reply: str,
) -> Task:
    """Folds one new (user, assistant) exchange into the running summary."""
    description = (
        f"Previous summary of the conversation:\n"
        f"{prev_summary or '(no prior turns yet)'}\n\n"
        f"New turn just completed:\n"
        f"User: {user_msg}\n"
        f"Assistant: {assistant_reply}\n\n"
        f"Produce an UPDATED summary that covers everything the user "
        f"and assistant have discussed so far, including the new turn. "
        f"Keep it under 5 sentences. If a tool was called, mention it "
        f"briefly. This summary will be passed to the assistant as "
        f"context for the NEXT user message, so preserve any facts, "
        f"preferences, or identifiers the user has shared."
    )
    return Task(
        description=description,
        expected_output="Updated rolling summary, under 5 sentences.",
        agent=agent,
    )


def create_recap_task(agent: Agent, transcript: str) -> Task:
    """One-shot conversation recap (the Summarize sidebar button)."""
    description = (
        "Summarize the conversation below as 3-5 bullet points covering "
        "the main topics discussed and any conclusions reached. If a "
        "tool was called, mention which tool and what it returned at a "
        "high level.\n\n"
        f"---\n{transcript}\n---"
    )
    return Task(
        description=description,
        expected_output="A short bulleted summary.",
        agent=agent,
    )
