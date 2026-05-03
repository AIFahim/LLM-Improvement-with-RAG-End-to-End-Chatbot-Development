"""
MCP Chatbot — agent definitions.

One place for the role / goal / backstory of every Agent the chatbot
creates. Three agents:

- Helpful Assistant: the user-facing chat agent (uses MCP tools)
- Conversation Summarizer: rolls the chat summary forward each turn
- Recap Writer: produces the on-demand "Summarize" button output
"""

from crewai import Agent
from crewai.llm import LLM


CHATBOT_BACKSTORY = (
    "You are a friendly conversational assistant. You have access to "
    "MCP tools (which may include math, date/time, web search, file/"
    "report operations, or repo Q&A depending on the connected server) "
    "— but you only reach for a tool when the user actually asks for "
    "something a tool can answer. Most messages (greetings, opinions, "
    "casual chat, general questions you already know) need NO tool "
    "calls — just reply naturally. If the user says 'hi', say 'hi' "
    "back. If they ask 'what's 17 * 23', use the calculator. The "
    "principle: tools serve the user's actual request; never call "
    "tools to demonstrate capability or fill silence."
)

SUMMARIZER_BACKSTORY = (
    "You produce neutral, faithful, compact summaries that preserve "
    "the user's stated facts and preferences across turns."
)

RECAP_BACKSTORY = (
    "You write faithful, neutral summaries. You don't add facts that "
    "weren't in the conversation, and you don't omit major topics that "
    "were."
)


class AgentFactory:
    """Build the chatbot's agents.

    Pass in a fresh LLM each time — the factory itself is stateless so
    swapping models in the UI mid-session doesn't require rebuilding
    anything else.
    """

    def __init__(self, llm: LLM, verbose: bool = True):
        self.llm = llm
        self.verbose = verbose

    def create_chatbot(self, tools: list) -> Agent:
        """The user-facing chat agent. Tools come from the MCP server."""
        return Agent(
            role="Helpful Assistant",
            goal=(
                "Have a useful conversation with the user, using tools "
                "only when they directly answer the user's request."
            ),
            backstory=CHATBOT_BACKSTORY,
            tools=tools,
            llm=self.llm,
            verbose=self.verbose,
            allow_delegation=False,
        )

    def create_summarizer(self) -> Agent:
        """Updates the rolling per-turn summary. No tools."""
        return Agent(
            role="Conversation Summarizer",
            goal="Maintain a short rolling summary of an ongoing conversation.",
            backstory=SUMMARIZER_BACKSTORY,
            tools=[],
            llm=self.llm,
            verbose=False,
            allow_delegation=False,
        )

    def create_recap_writer(self) -> Agent:
        """Produces the explicit Summarize-button output. No tools."""
        return Agent(
            role="Summarizer",
            goal="Summarize a chatbot conversation accurately and concisely.",
            backstory=RECAP_BACKSTORY,
            tools=[],
            llm=self.llm,
            verbose=False,
            allow_delegation=False,
        )
