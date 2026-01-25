"""
Monitoring and observability module using Langfuse and Prometheus
Provides full tracing, metrics, and debugging capabilities
"""
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime
import config

logger = logging.getLogger(__name__)

# =============================================================================
# Data Classes for Monitoring
# =============================================================================


@dataclass
class TraceInfo:
    """Information about a single trace"""
    trace_id: str
    session_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    input_data: Optional[str] = None
    output_data: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None


@dataclass
class RetrievalTrace:
    """Trace information for document retrieval"""
    query: str
    num_results: int
    retrieval_time_ms: float
    documents: List[Dict[str, Any]]
    scores: Optional[List[float]] = None


@dataclass
class PerformanceProfile:
    """Performance profiling results"""
    total_time_ms: float
    retrieval_time_ms: float
    llm_time_ms: float
    embedding_time_ms: Optional[float] = None
    tokens_used: Optional[int] = None


# =============================================================================
# Langfuse Monitor
# =============================================================================


class LangfuseMonitor:
    """
    Langfuse integration for LLM observability
    Provides tracing, scoring, and session management
    """

    def __init__(self):
        """Initialize Langfuse client if enabled"""
        self._langfuse = None
        self._enabled = config.LANGFUSE_ENABLED

        if self._enabled:
            try:
                from langfuse import Langfuse
                self._langfuse = Langfuse(
                    public_key=config.LANGFUSE_PUBLIC_KEY,
                    secret_key=config.LANGFUSE_SECRET_KEY,
                    host=config.LANGFUSE_HOST
                )
                logger.info(f"Langfuse initialized with host: {config.LANGFUSE_HOST}")
            except ImportError:
                logger.warning("Langfuse not installed. Install with: pip install langfuse")
                self._enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize Langfuse: {e}")
                self._enabled = False

        self._traces: Dict[str, TraceInfo] = {}

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self._langfuse is not None

    def create_trace(
        self,
        name: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Create a new trace for tracking an interaction

        Args:
            name: Name of the trace (e.g., "chat", "retrieval")
            session_id: Optional session identifier
            user_id: Optional user identifier
            metadata: Optional metadata dictionary

        Returns:
            Trace ID or None if not enabled
        """
        import uuid
        trace_id = str(uuid.uuid4())

        trace_info = TraceInfo(
            trace_id=trace_id,
            session_id=session_id,
            metadata=metadata or {}
        )
        self._traces[trace_id] = trace_info

        if self.is_enabled:
            try:
                trace = self._langfuse.trace(
                    id=trace_id,
                    name=name,
                    session_id=session_id,
                    user_id=user_id,
                    metadata=metadata
                )
                logger.debug(f"Created Langfuse trace: {trace_id}")
            except Exception as e:
                logger.error(f"Failed to create Langfuse trace: {e}")

        return trace_id

    def trace_chat(
        self,
        trace_id: str,
        query: str,
        response: str,
        contexts: List[str],
        metadata: Optional[Dict] = None
    ):
        """
        Trace a chat interaction

        Args:
            trace_id: The trace identifier
            query: User query
            response: Assistant response
            contexts: Retrieved contexts
            metadata: Optional additional metadata
        """
        if trace_id in self._traces:
            trace_info = self._traces[trace_id]
            trace_info.input_data = query
            trace_info.output_data = response
            trace_info.end_time = time.time()
            if metadata:
                trace_info.metadata.update(metadata)

        if self.is_enabled:
            try:
                self._langfuse.generation(
                    trace_id=trace_id,
                    name="chat_response",
                    input=query,
                    output=response,
                    metadata={
                        "contexts": contexts[:3],  # Limit context size
                        "num_contexts": len(contexts),
                        **(metadata or {})
                    }
                )
            except Exception as e:
                logger.error(f"Failed to trace chat: {e}")

    def trace_retrieval(
        self,
        trace_id: str,
        query: str,
        documents: List[Any],
        retrieval_time_ms: float
    ):
        """
        Trace a retrieval operation

        Args:
            trace_id: The trace identifier
            query: Search query
            documents: Retrieved documents
            retrieval_time_ms: Time taken for retrieval
        """
        if self.is_enabled:
            try:
                self._langfuse.span(
                    trace_id=trace_id,
                    name="retrieval",
                    input=query,
                    output={"num_documents": len(documents)},
                    metadata={
                        "retrieval_time_ms": retrieval_time_ms,
                        "num_results": len(documents)
                    }
                )
            except Exception as e:
                logger.error(f"Failed to trace retrieval: {e}")

    def score_response(self, trace_id: str, scores: Dict[str, float]):
        """
        Add evaluation scores to a trace

        Args:
            trace_id: The trace identifier
            scores: Dictionary of score names to values
        """
        if trace_id in self._traces:
            self._traces[trace_id].scores.update(scores)

        if self.is_enabled:
            try:
                for name, value in scores.items():
                    self._langfuse.score(
                        trace_id=trace_id,
                        name=name,
                        value=value
                    )
                logger.debug(f"Added scores to trace {trace_id}: {scores}")
            except Exception as e:
                logger.error(f"Failed to score response: {e}")

    def log_error(self, trace_id: str, error: str, error_type: str = "error"):
        """
        Log an error for a trace

        Args:
            trace_id: The trace identifier
            error: Error message
            error_type: Type of error
        """
        if trace_id in self._traces:
            self._traces[trace_id].error = error

        if self.is_enabled:
            try:
                self._langfuse.event(
                    trace_id=trace_id,
                    name=error_type,
                    metadata={"error": error}
                )
            except Exception as e:
                logger.error(f"Failed to log error: {e}")

    def get_trace_info(self, trace_id: str) -> Optional[TraceInfo]:
        """Get trace information by ID"""
        return self._traces.get(trace_id)

    def flush(self):
        """Flush pending traces to Langfuse"""
        if self.is_enabled:
            try:
                self._langfuse.flush()
            except Exception as e:
                logger.error(f"Failed to flush Langfuse: {e}")


# =============================================================================
# Prometheus Metrics
# =============================================================================


class PrometheusMetrics:
    """
    Prometheus metrics for monitoring
    Provides counters, histograms, and gauges for key metrics
    """

    def __init__(self):
        """Initialize Prometheus metrics"""
        self._enabled = False
        self._metrics = {}

        try:
            from prometheus_client import Counter, Histogram, Gauge, Info

            self._metrics = {
                "request_counter": Counter(
                    'chatbot_requests_total',
                    'Total number of chatbot requests',
                    ['endpoint', 'status']
                ),
                "response_latency": Histogram(
                    'chatbot_response_latency_seconds',
                    'Response latency in seconds',
                    ['endpoint'],
                    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
                ),
                "token_usage": Gauge(
                    'chatbot_tokens_used',
                    'Number of tokens used',
                    ['type']  # input, output, total
                ),
                "error_counter": Counter(
                    'chatbot_errors_total',
                    'Total number of errors',
                    ['type']  # retrieval, llm, evaluation
                ),
                "active_sessions": Gauge(
                    'chatbot_active_sessions',
                    'Number of active sessions'
                ),
                "documents_processed": Counter(
                    'chatbot_documents_processed_total',
                    'Total documents processed'
                ),
                "cache_hits": Counter(
                    'chatbot_cache_hits_total',
                    'Cache hit count',
                    ['cache_type']
                ),
                "cache_misses": Counter(
                    'chatbot_cache_misses_total',
                    'Cache miss count',
                    ['cache_type']
                ),
                "evaluation_scores": Histogram(
                    'chatbot_evaluation_scores',
                    'Evaluation score distribution',
                    ['metric'],
                    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
                ),
                "info": Info(
                    'chatbot_info',
                    'Chatbot application information'
                )
            }

            # Set application info
            self._metrics["info"].info({
                'version': '1.0.0',
                'llm_provider': config.LLM_PROVIDER,
                'model': config.OLLAMA_MODEL if config.LLM_PROVIDER == 'ollama'
                else config.AZURE_OPENAI_DEPLOYMENT
            })

            self._enabled = True
            logger.info("Prometheus metrics initialized")

        except ImportError:
            logger.warning("prometheus_client not installed. Metrics disabled.")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def increment_request(self, endpoint: str, status: str = "success"):
        """Increment request counter"""
        if self._enabled:
            self._metrics["request_counter"].labels(
                endpoint=endpoint, status=status
            ).inc()

    def observe_latency(self, endpoint: str, latency_seconds: float):
        """Record response latency"""
        if self._enabled:
            self._metrics["response_latency"].labels(
                endpoint=endpoint
            ).observe(latency_seconds)

    def set_token_usage(self, input_tokens: int, output_tokens: int):
        """Set token usage metrics"""
        if self._enabled:
            self._metrics["token_usage"].labels(type="input").set(input_tokens)
            self._metrics["token_usage"].labels(type="output").set(output_tokens)
            self._metrics["token_usage"].labels(type="total").set(
                input_tokens + output_tokens
            )

    def increment_error(self, error_type: str):
        """Increment error counter"""
        if self._enabled:
            self._metrics["error_counter"].labels(type=error_type).inc()

    def set_active_sessions(self, count: int):
        """Set active session count"""
        if self._enabled:
            self._metrics["active_sessions"].set(count)

    def increment_documents_processed(self, count: int = 1):
        """Increment documents processed counter"""
        if self._enabled:
            self._metrics["documents_processed"].inc(count)

    def record_cache_hit(self, cache_type: str = "response"):
        """Record a cache hit"""
        if self._enabled:
            self._metrics["cache_hits"].labels(cache_type=cache_type).inc()

    def record_cache_miss(self, cache_type: str = "response"):
        """Record a cache miss"""
        if self._enabled:
            self._metrics["cache_misses"].labels(cache_type=cache_type).inc()

    def record_evaluation_score(self, metric: str, score: float):
        """Record an evaluation score"""
        if self._enabled:
            self._metrics["evaluation_scores"].labels(metric=metric).observe(score)

    def get_metrics_output(self) -> str:
        """Get Prometheus metrics in text format"""
        if self._enabled:
            try:
                from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
                return generate_latest().decode('utf-8')
            except Exception as e:
                logger.error(f"Failed to generate metrics: {e}")
        return ""


# =============================================================================
# Debug Tools
# =============================================================================


class DebugTools:
    """
    Debugging utilities for development and troubleshooting
    """

    def __init__(self):
        self._profiling_data: List[PerformanceProfile] = []

    def trace_retrieval(
        self,
        query: str,
        documents: List[Any],
        retrieval_time_ms: float,
        scores: Optional[List[float]] = None
    ) -> RetrievalTrace:
        """
        Create a detailed retrieval trace

        Args:
            query: Search query
            documents: Retrieved documents
            retrieval_time_ms: Time taken for retrieval
            scores: Optional relevance scores

        Returns:
            RetrievalTrace with detailed information
        """
        doc_info = []
        for i, doc in enumerate(documents):
            info = {
                "index": i,
                "content_preview": doc.page_content[:200] if hasattr(doc, 'page_content') else str(doc)[:200],
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
            }
            if scores and i < len(scores):
                info["score"] = scores[i]
            doc_info.append(info)

        return RetrievalTrace(
            query=query,
            num_results=len(documents),
            retrieval_time_ms=retrieval_time_ms,
            documents=doc_info,
            scores=scores
        )

    def profile_query(
        self,
        total_time_ms: float,
        retrieval_time_ms: float,
        llm_time_ms: float,
        embedding_time_ms: Optional[float] = None,
        tokens_used: Optional[int] = None
    ) -> PerformanceProfile:
        """
        Create a performance profile for a query

        Args:
            total_time_ms: Total query time
            retrieval_time_ms: Time for retrieval
            llm_time_ms: Time for LLM generation
            embedding_time_ms: Optional embedding time
            tokens_used: Optional token count

        Returns:
            PerformanceProfile with timing breakdown
        """
        profile = PerformanceProfile(
            total_time_ms=total_time_ms,
            retrieval_time_ms=retrieval_time_ms,
            llm_time_ms=llm_time_ms,
            embedding_time_ms=embedding_time_ms,
            tokens_used=tokens_used
        )
        self._profiling_data.append(profile)
        return profile

    def get_average_profile(self) -> Optional[PerformanceProfile]:
        """Get average performance profile"""
        if not self._profiling_data:
            return None

        n = len(self._profiling_data)
        return PerformanceProfile(
            total_time_ms=sum(p.total_time_ms for p in self._profiling_data) / n,
            retrieval_time_ms=sum(p.retrieval_time_ms for p in self._profiling_data) / n,
            llm_time_ms=sum(p.llm_time_ms for p in self._profiling_data) / n
        )

    def clear_profiling_data(self):
        """Clear all profiling data"""
        self._profiling_data.clear()


# =============================================================================
# Monitoring Decorators
# =============================================================================


def monitor_latency(endpoint: str):
    """
    Decorator to monitor function latency

    Args:
        endpoint: Name of the endpoint/function
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                latency = time.time() - start_time
                logger.debug(f"{endpoint} latency: {latency:.3f}s")
        return wrapper
    return decorator


def trace_function(name: str):
    """
    Decorator to trace function execution

    Args:
        name: Name for the trace
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"Entering {name}")
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = (time.time() - start_time) * 1000
                logger.debug(f"Exiting {name} (took {duration:.2f}ms)")
                return result
            except Exception as e:
                logger.error(f"Error in {name}: {e}")
                raise
        return wrapper
    return decorator


# =============================================================================
# Global Monitor Instances
# =============================================================================

# Create singleton instances
langfuse_monitor = LangfuseMonitor()
prometheus_metrics = PrometheusMetrics()
debug_tools = DebugTools()
