"""
LLM handler module for managing Ollama and Azure OpenAI interactions
Enhanced with retry logic, streaming, caching, and monitoring
"""
import time
from typing import Optional, Dict, Any, Generator
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
import config
import logging
from error_handler import retry_decorator, RetryHandler, error_handler
from monitoring import langfuse_monitor, prometheus_metrics, trace_function
from performance import response_cache, timed_execution

logger = logging.getLogger(__name__)


class LLMHandler:
    """
    Handles interactions with LLM (Ollama or Azure OpenAI)
    Enhanced with error handling, monitoring, and performance optimizations
    """

    def __init__(
        self,
        provider: str = config.LLM_PROVIDER,
        temperature: float = config.LLM_TEMPERATURE
    ):
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
        self._retry_handler = RetryHandler()

        # Monitoring
        self._enable_monitoring = config.LANGFUSE_ENABLED
        self._enable_caching = config.CACHE_ENABLED

        logger.info(f"LLM Handler initialized with provider: {self.provider}")

    @trace_function("get_llm")
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
        """Create Ollama LLM instance with retry logic"""
        from langchain_ollama import OllamaLLM

        def create():
            return OllamaLLM(
                model=config.OLLAMA_MODEL,
                base_url=config.OLLAMA_BASE_URL,
                temperature=self.temperature
            )

        try:
            llm = self._retry_handler.with_retry(
                create,
                retryable_exceptions=(ConnectionError, TimeoutError)
            )
            logger.info(f"Initialized Ollama LLM with model {config.OLLAMA_MODEL}")
            return llm
        except Exception as e:
            logger.error(f"Failed to create Ollama LLM: {e}")
            raise

    def _create_azure_llm(self):
        """Create Azure OpenAI LLM instance with retry logic"""
        from langchain_openai import AzureChatOpenAI

        if not config.AZURE_OPENAI_API_KEY or not config.AZURE_OPENAI_ENDPOINT:
            raise ValueError(
                "Azure OpenAI credentials not configured. "
                "Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables."
            )

        def create():
            return AzureChatOpenAI(
                azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                azure_deployment=config.AZURE_OPENAI_DEPLOYMENT,
                api_key=config.AZURE_OPENAI_API_KEY,
                api_version=config.AZURE_OPENAI_API_VERSION,
                temperature=self.temperature
            )

        try:
            llm = self._retry_handler.with_retry(
                create,
                retryable_exceptions=(ConnectionError, TimeoutError)
            )
            logger.info(f"Initialized Azure OpenAI with deployment {config.AZURE_OPENAI_DEPLOYMENT}")
            return llm
        except Exception as e:
            logger.error(f"Failed to create Azure OpenAI LLM: {e}")
            raise

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

    @trace_function("create_qa_chain")
    def create_qa_chain(self, retriever) -> RetrievalQA:
        """
        Create a QA chain with retriever

        Args:
            retriever: Document retriever instance

        Returns:
            RetrievalQA chain instance
        """
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

    @timed_execution
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the QA chain with caching and monitoring

        Args:
            question: User question

        Returns:
            Dictionary with result and source documents
        """
        if self._qa_chain is None:
            raise ValueError("QA chain not initialized. Call create_qa_chain first.")

        start_time = time.time()
        trace_id = None

        # Check cache first
        if self._enable_caching:
            cached = response_cache.get(question)
            if cached is not None:
                logger.debug(f"Cache hit for: {question[:50]}...")
                prometheus_metrics.record_cache_hit("response")
                return {
                    "result": cached.response,
                    "source_documents": [],
                    "cached": True
                }
            prometheus_metrics.record_cache_miss("response")

        # Create monitoring trace
        if self._enable_monitoring:
            trace_id = langfuse_monitor.create_trace(
                name="chat_query",
                metadata={"provider": self.provider}
            )

        try:
            # Execute query with retry
            response = self._retry_handler.with_retry(
                lambda: self._qa_chain({"query": question}),
                retryable_exceptions=(ConnectionError, TimeoutError)
            )

            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"Generated response in {elapsed_ms:.2f}ms for: {question[:50]}...")

            # Cache the response
            if self._enable_caching:
                contexts = [
                    doc.page_content
                    for doc in response.get("source_documents", [])
                ]
                response_cache.set(question, response["result"], contexts)

            # Record monitoring data
            if self._enable_monitoring and trace_id:
                contexts = [
                    doc.page_content
                    for doc in response.get("source_documents", [])
                ]
                langfuse_monitor.trace_chat(
                    trace_id=trace_id,
                    query=question,
                    response=response["result"],
                    contexts=contexts,
                    metadata={"latency_ms": elapsed_ms}
                )

            # Record metrics
            prometheus_metrics.increment_request("chat", "success")
            prometheus_metrics.observe_latency("chat", elapsed_ms / 1000)

            # Add trace_id to response for evaluation
            response["trace_id"] = trace_id

            return response

        except Exception as e:
            logger.error(f"Error during query: {e}")
            prometheus_metrics.increment_request("chat", "error")
            prometheus_metrics.increment_error("llm")

            if trace_id:
                langfuse_monitor.log_error(trace_id, str(e))

            # Handle error and potentially get suggestion
            error_context = error_handler.handle_error(e, "LLM query", question)

            raise

    @retry_decorator(max_retries=3)
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

    def stream_response(self, question: str) -> Generator[str, None, None]:
        """
        Stream a response in chunks

        Args:
            question: User question

        Yields:
            Response chunks
        """
        if self._qa_chain is None:
            raise ValueError("QA chain not initialized. Call create_qa_chain first.")

        try:
            # Get full response first (streaming from chain requires different setup)
            response = self.query(question)
            full_text = response.get("result", "")

            # Yield in chunks
            chunk_size = 50
            for i in range(0, len(full_text), chunk_size):
                yield full_text[i:i + chunk_size]
                time.sleep(0.02)  # Small delay for streaming effect

        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            yield f"Error: {str(e)}"

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

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        cache_stats = response_cache.get_stats()
        return {
            "provider": self.provider,
            "model": self.model_info["model"],
            "cache_enabled": self._enable_caching,
            "monitoring_enabled": self._enable_monitoring,
            "cache_stats": cache_stats
        }
