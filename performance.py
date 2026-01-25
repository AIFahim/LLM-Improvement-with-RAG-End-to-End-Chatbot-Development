"""
Performance optimization module
Provides caching, batching, streaming, and retrieval optimization
"""
import time
import hashlib
import logging
from typing import List, Dict, Optional, Any, Generator, Callable, TypeVar
from dataclasses import dataclass, field
from functools import wraps
from collections import OrderedDict
import threading
import config

logger = logging.getLogger(__name__)

T = TypeVar('T')

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CachedResponse:
    """A cached response with metadata"""
    response: str
    contexts: List[str]
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def is_expired(self, ttl: int) -> bool:
        """Check if the cached response has expired"""
        return (time.time() - self.created_at) > ttl


@dataclass
class CachedEmbedding:
    """A cached embedding with metadata"""
    text: str
    embedding: List[float]
    created_at: float = field(default_factory=time.time)


@dataclass
class OptimizationStats:
    """Statistics about optimization performance"""
    cache_hits: int = 0
    cache_misses: int = 0
    total_requests: int = 0
    avg_response_time_ms: float = 0.0
    tokens_saved: int = 0


# =============================================================================
# Response Cache
# =============================================================================


class ResponseCache:
    """
    In-memory cache for LLM responses
    Uses LRU eviction and TTL expiration
    """

    def __init__(
        self,
        max_size: int = config.CACHE_MAX_SIZE,
        ttl_seconds: int = config.CACHE_TTL_SECONDS
    ):
        """
        Initialize the response cache

        Args:
            max_size: Maximum number of cached responses
            ttl_seconds: Time-to-live in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CachedResponse] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = OptimizationStats()

    def _generate_key(self, query: str, contexts: Optional[List[str]] = None) -> str:
        """Generate a cache key from query and contexts"""
        content = query
        if contexts:
            content += "||" + "||".join(sorted(contexts))
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, query: str, contexts: Optional[List[str]] = None) -> Optional[CachedResponse]:
        """
        Get a cached response

        Args:
            query: The query to look up
            contexts: Optional contexts for key generation

        Returns:
            CachedResponse if found and not expired, None otherwise
        """
        key = self._generate_key(query, contexts)

        with self._lock:
            self._stats.total_requests += 1

            if key not in self._cache:
                self._stats.cache_misses += 1
                return None

            cached = self._cache[key]

            if cached.is_expired(self.ttl_seconds):
                del self._cache[key]
                self._stats.cache_misses += 1
                return None

            # Update access stats and move to end (LRU)
            cached.access_count += 1
            cached.last_accessed = time.time()
            self._cache.move_to_end(key)

            self._stats.cache_hits += 1
            return cached

    def set(
        self,
        query: str,
        response: str,
        contexts: List[str],
        ttl: Optional[int] = None
    ):
        """
        Cache a response

        Args:
            query: The query
            response: The response to cache
            contexts: The contexts used
            ttl: Optional custom TTL (uses default if None)
        """
        key = self._generate_key(query, contexts)

        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)  # Remove oldest

            self._cache[key] = CachedResponse(
                response=response,
                contexts=contexts
            )

    def invalidate(self, pattern: Optional[str] = None) -> int:
        """
        Invalidate cache entries

        Args:
            pattern: Optional pattern to match queries (invalidates all if None)

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            if pattern is None:
                count = len(self._cache)
                self._cache.clear()
                return count

            # Pattern matching (simple substring match)
            to_remove = []
            for key, cached in self._cache.items():
                # We don't store the original query, so just clear matching keys
                # In production, you might want to store the query for pattern matching
                if pattern in key:
                    to_remove.append(key)

            for key in to_remove:
                del self._cache[key]

            return len(to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            hit_rate = (
                self._stats.cache_hits / self._stats.total_requests
                if self._stats.total_requests > 0 else 0.0
            )
            return {
                "cache_size": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "cache_hits": self._stats.cache_hits,
                "cache_misses": self._stats.cache_misses,
                "total_requests": self._stats.total_requests,
                "hit_rate": hit_rate
            }


# =============================================================================
# Embedding Cache
# =============================================================================


class EmbeddingCache:
    """
    Cache for document embeddings
    Prevents redundant embedding computations
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize the embedding cache

        Args:
            max_size: Maximum number of cached embeddings
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, CachedEmbedding] = OrderedDict()
        self._lock = threading.RLock()

    def _generate_key(self, text: str) -> str:
        """Generate a cache key from text"""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def get(self, text: str) -> Optional[List[float]]:
        """Get a cached embedding"""
        key = self._generate_key(text)

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key].embedding
            return None

    def set(self, text: str, embedding: List[float]):
        """Cache an embedding"""
        key = self._generate_key(text)

        with self._lock:
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)

            self._cache[key] = CachedEmbedding(
                text=text,
                embedding=embedding
            )

    def cache_documents(
        self,
        documents: List[Any],
        embed_func: Callable[[str], List[float]]
    ) -> List[List[float]]:
        """
        Cache embeddings for multiple documents

        Args:
            documents: List of documents to embed
            embed_func: Function to compute embeddings

        Returns:
            List of embeddings
        """
        embeddings = []

        for doc in documents:
            text = doc.page_content if hasattr(doc, 'page_content') else str(doc)

            cached = self.get(text)
            if cached is not None:
                embeddings.append(cached)
            else:
                embedding = embed_func(text)
                self.set(text, embedding)
                embeddings.append(embedding)

        return embeddings


# =============================================================================
# Performance Optimizer
# =============================================================================


class PerformanceOptimizer:
    """
    Main performance optimization class
    Coordinates caching, batching, and other optimizations
    """

    def __init__(self):
        """Initialize the performance optimizer"""
        self.response_cache = ResponseCache()
        self.embedding_cache = EmbeddingCache()
        self._response_times: List[float] = []

    def cache_embeddings(
        self,
        documents: List[Any],
        embed_func: Callable[[str], List[float]]
    ) -> List[List[float]]:
        """
        Cache embeddings for documents

        Args:
            documents: Documents to embed
            embed_func: Embedding function

        Returns:
            List of embeddings
        """
        return self.embedding_cache.cache_documents(documents, embed_func)

    def batch_process(
        self,
        items: List[T],
        process_func: Callable[[List[T]], List[Any]],
        batch_size: int = 10
    ) -> List[Any]:
        """
        Process items in batches

        Args:
            items: Items to process
            process_func: Function that processes a batch
            batch_size: Size of each batch

        Returns:
            Combined results from all batches
        """
        results = []

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = process_func(batch)
            results.extend(batch_results)
            logger.debug(f"Processed batch {i // batch_size + 1}")

        return results

    def stream_response(
        self,
        generate_func: Callable[[], str],
        chunk_size: int = 50
    ) -> Generator[str, None, None]:
        """
        Stream a response in chunks (simulated for non-streaming LLMs)

        Args:
            generate_func: Function that generates the full response
            chunk_size: Number of characters per chunk

        Yields:
            Response chunks
        """
        full_response = generate_func()

        for i in range(0, len(full_response), chunk_size):
            yield full_response[i:i + chunk_size]
            time.sleep(0.01)  # Small delay for streaming effect

    def optimize_retrieval(
        self,
        retriever: Any,
        k: int = 4,
        score_threshold: float = 0.5
    ) -> Any:
        """
        Create an optimized retriever with score filtering

        Args:
            retriever: Base retriever
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score

        Returns:
            Optimized retriever
        """
        try:
            # Try to create a retriever with score threshold
            return retriever.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": k,
                    "score_threshold": score_threshold
                }
            )
        except Exception:
            # Fall back to regular retriever
            return retriever.as_retriever(search_kwargs={"k": k})

    def record_response_time(self, time_ms: float):
        """Record a response time for statistics"""
        self._response_times.append(time_ms)
        # Keep only last 100 times
        if len(self._response_times) > 100:
            self._response_times = self._response_times[-100:]

    def get_avg_response_time(self) -> float:
        """Get average response time"""
        if not self._response_times:
            return 0.0
        return sum(self._response_times) / len(self._response_times)

    def get_optimization_report(self) -> Dict[str, Any]:
        """Get a full optimization report"""
        return {
            "response_cache": self.response_cache.get_stats(),
            "embedding_cache_size": len(self.embedding_cache._cache),
            "avg_response_time_ms": self.get_avg_response_time(),
            "total_responses_tracked": len(self._response_times)
        }


# =============================================================================
# Decorators for Performance
# =============================================================================


def cached_response(cache: ResponseCache = None):
    """
    Decorator to cache function responses

    Args:
        cache: ResponseCache instance (creates new if None)
    """
    _cache = cache or ResponseCache()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(query: str, *args, **kwargs):
            # Try to get from cache
            cached = _cache.get(query)
            if cached is not None:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return {
                    "result": cached.response,
                    "source_documents": [],
                    "cached": True
                }

            # Execute function
            result = func(query, *args, **kwargs)

            # Cache the result
            if isinstance(result, dict) and "result" in result:
                contexts = [
                    doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    for doc in result.get("source_documents", [])
                ]
                _cache.set(query, result["result"], contexts)

            return result
        return wrapper
    return decorator


def timed_execution(func: Callable) -> Callable:
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(f"{func.__name__} took {elapsed_ms:.2f}ms")
        return result
    return wrapper


def batch_enabled(batch_size: int = 10):
    """
    Decorator to enable automatic batching for list inputs

    Args:
        batch_size: Size of each batch
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(items: List, *args, **kwargs):
            if len(items) <= batch_size:
                return func(items, *args, **kwargs)

            results = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                batch_results = func(batch, *args, **kwargs)
                if isinstance(batch_results, list):
                    results.extend(batch_results)
                else:
                    results.append(batch_results)
            return results
        return wrapper
    return decorator


# =============================================================================
# Global Optimizer Instance
# =============================================================================

# Create singleton instance
performance_optimizer = PerformanceOptimizer()
response_cache = performance_optimizer.response_cache
embedding_cache = performance_optimizer.embedding_cache
