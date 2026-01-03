"""
LLM handler module for managing Ollama and Azure OpenAI interactions
"""
from typing import Optional, Dict, Any, Union
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
import config
import logging

logger = logging.getLogger(__name__)


class LLMHandler:
    """Handles interactions with LLM (Ollama or Azure OpenAI)"""

    def __init__(self,
                 provider: str = config.LLM_PROVIDER,
                 temperature: float = config.LLM_TEMPERATURE):
        """
        Initialize the LLM handler

        Args:
            provider: LLM provider ("ollama" or "azure")
            temperature: Temperature for generation
        """
        self.provider = provider.lower()
        self.temperature = temperature
        self._llm = None
        self._memory = None
        self._qa_chain = None

        logger.info(f"LLM Handler initialized with provider: {self.provider}")

    def get_llm(self) -> BaseLanguageModel:
        """
        Get or create the LLM instance based on provider

        Returns:
            LLM instance (Ollama or Azure OpenAI)
        """
        if self._llm is None:
            if self.provider == "azure":
                self._llm = self._create_azure_llm()
            else:
                self._llm = self._create_ollama_llm()
        return self._llm

    def _create_ollama_llm(self):
        """Create Ollama LLM instance"""
        from langchain_ollama import OllamaLLM

        llm = OllamaLLM(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=self.temperature
        )
        logger.info(f"Initialized Ollama LLM with model {config.OLLAMA_MODEL}")
        return llm

    def _create_azure_llm(self):
        """Create Azure OpenAI LLM instance"""
        from langchain_openai import AzureChatOpenAI

        if not config.AZURE_OPENAI_API_KEY or not config.AZURE_OPENAI_ENDPOINT:
            raise ValueError(
                "Azure OpenAI credentials not configured. "
                "Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables."
            )

        llm = AzureChatOpenAI(
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            azure_deployment=config.AZURE_OPENAI_DEPLOYMENT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
            temperature=self.temperature
        )
        logger.info(f"Initialized Azure OpenAI with deployment {config.AZURE_OPENAI_DEPLOYMENT}")
        return llm

    def get_memory(self) -> ConversationBufferMemory:
        """
        Get or create conversation memory

        Returns:
            ConversationBufferMemory instance
        """
        if self._memory is None:
            self._memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="result"
            )
            logger.info("Initialized conversation memory")
        return self._memory

    def create_qa_chain(self, retriever) -> RetrievalQA:
        """
        Create a QA chain with retriever

        Args:
            retriever: Document retriever instance

        Returns:
            RetrievalQA chain instance
        """
        # Define the prompt template
        prompt_template = """You are a helpful AI assistant that ONLY answers based on the provided context.

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer ONLY using information from the CONTEXT above
- Be specific and quote relevant details from the context
- If the context mentions a class name, topic, or subject, state it clearly
- If the answer is not in the context, say "I don't have that information in the provided documents"

ANSWER: """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        self._qa_chain = RetrievalQA.from_chain_type(
            llm=self.get_llm(),
            chain_type="stuff",
            retriever=retriever,
            memory=self.get_memory(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        logger.info(f"Created QA chain with retriever (provider: {self.provider})")
        return self._qa_chain

    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the QA chain

        Args:
            question: User question

        Returns:
            Dictionary with result and source documents
        """
        if self._qa_chain is None:
            raise ValueError("QA chain not initialized. Call create_qa_chain first.")

        try:
            response = self._qa_chain({"query": question})
            logger.info(f"Generated response for question: {question[:50]}...")
            return response
        except Exception as e:
            logger.error(f"Error during query: {e}")
            raise

    def generate_response(self, prompt: str) -> str:
        """
        Generate response without retrieval (direct LLM call)

        Args:
            prompt: Input prompt

        Returns:
            Generated response
        """
        llm = self.get_llm()
        try:
            response = llm.invoke(prompt)
            # Handle different response types
            if hasattr(response, 'content'):
                return response.content  # Azure returns AIMessage
            return str(response)  # Ollama returns string
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def clear_memory(self):
        """Clear conversation memory"""
        if self._memory:
            self._memory.clear()
            logger.info("Cleared conversation memory")

    def get_conversation_history(self) -> list:
        """
        Get conversation history

        Returns:
            List of conversation messages
        """
        if self._memory is None:
            return []

        return self._memory.chat_memory.messages

    def switch_provider(self, provider: str):
        """
        Switch LLM provider

        Args:
            provider: New provider ("ollama" or "azure")
        """
        self.provider = provider.lower()
        self._llm = None  # Reset LLM to force recreation
        self._qa_chain = None  # Reset chain
        logger.info(f"Switched LLM provider to: {self.provider}")

    @property
    def current_provider(self) -> str:
        """Get current LLM provider"""
        return self.provider

    @property
    def model_info(self) -> Dict[str, str]:
        """Get current model information"""
        if self.provider == "azure":
            return {
                "provider": "Azure OpenAI",
                "model": config.AZURE_OPENAI_DEPLOYMENT,
                "endpoint": config.AZURE_OPENAI_ENDPOINT
            }
        else:
            return {
                "provider": "Ollama",
                "model": config.OLLAMA_MODEL,
                "endpoint": config.OLLAMA_BASE_URL
            }
