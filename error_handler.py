"""
Error handling module with retry logic, fallbacks, and self-reflection
Provides robust error recovery for LLM applications
"""
import time
import logging
from typing import Callable, Optional, Any, TypeVar, List
from functools import wraps
from dataclasses import dataclass
from enum import Enum
import config

logger = logging.getLogger(__name__)

T = TypeVar('T')

# =============================================================================
# Error Types
# =============================================================================


class ErrorType(Enum):
    """Types of errors that can occur"""
    CONNECTION = "connection"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    INVALID_REQUEST = "invalid_request"
    MODEL_ERROR = "model_error"
    RETRIEVAL_ERROR = "retrieval_error"
    VISION_ERROR = "vision_error"
    AUDIO_ERROR = "audio_error"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information about an error"""
    error_type: ErrorType
    message: str
    original_error: Optional[Exception] = None
    retry_count: int = 0
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class ReflectionResult:
    """Result of error reflection/analysis"""
    error_analysis: str
    suggested_fix: str
    should_retry: bool
    modified_query: Optional[str] = None


# =============================================================================
# Retry Handler
# =============================================================================


class RetryHandler:
    """
    Handles retry logic with exponential backoff
    """

    def __init__(
        self,
        max_retries: int = config.MAX_RETRIES,
        base_delay: float = config.RETRY_DELAY_SECONDS,
        max_delay: float = 30.0,
        exponential_base: float = 2.0
    ):
        """
        Initialize the retry handler

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries
            exponential_base: Base for exponential backoff
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt using exponential backoff"""
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)

    def with_retry(
        self,
        func: Callable[..., T],
        *args,
        retryable_exceptions: tuple = (Exception,),
        on_retry: Optional[Callable[[int, Exception], None]] = None,
        **kwargs
    ) -> T:
        """
        Execute a function with retry logic

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            retryable_exceptions: Tuple of exceptions that should trigger retry
            on_retry: Optional callback called on each retry
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function

        Raises:
            Last exception if all retries fail
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)

            except retryable_exceptions as e:
                last_exception = e

                if attempt < self.max_retries:
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    if on_retry:
                        on_retry(attempt, e)

                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {self.max_retries + 1} attempts failed. Last error: {e}"
                    )

        raise last_exception

    def with_fallback(
        self,
        primary_func: Callable[..., T],
        fallback_func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """
        Execute primary function with fallback

        Args:
            primary_func: Primary function to try first
            fallback_func: Fallback function if primary fails
            *args: Arguments for both functions
            **kwargs: Keyword arguments for both functions

        Returns:
            Result from either primary or fallback function
        """
        try:
            return self.with_retry(primary_func, *args, **kwargs)
        except Exception as e:
            logger.warning(f"Primary function failed: {e}. Trying fallback...")
            try:
                return fallback_func(*args, **kwargs)
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                raise


def retry_decorator(
    max_retries: int = config.MAX_RETRIES,
    base_delay: float = config.RETRY_DELAY_SECONDS,
    retryable_exceptions: tuple = (Exception,)
):
    """
    Decorator for adding retry logic to functions

    Args:
        max_retries: Maximum retry attempts
        base_delay: Initial delay between retries
        retryable_exceptions: Exceptions that should trigger retry
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        handler = RetryHandler(max_retries=max_retries, base_delay=base_delay)

        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return handler.with_retry(
                func, *args,
                retryable_exceptions=retryable_exceptions,
                **kwargs
            )
        return wrapper
    return decorator


# =============================================================================
# Circuit Breaker
# =============================================================================


class CircuitBreaker:
    """
    Circuit breaker pattern for external services
    Prevents repeated calls to failing services
    """

    class State(Enum):
        CLOSED = "closed"  # Normal operation
        OPEN = "open"  # Failing, reject requests
        HALF_OPEN = "half_open"  # Testing if service recovered

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2
    ):
        """
        Initialize circuit breaker

        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before trying again
            success_threshold: Successes needed to close circuit
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self._state = self.State.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0

    @property
    def state(self) -> State:
        if self._state == self.State.OPEN:
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = self.State.HALF_OPEN
                self._success_count = 0
        return self._state

    def record_success(self):
        """Record a successful call"""
        if self._state == self.State.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state = self.State.CLOSED
                self._failure_count = 0
                logger.info("Circuit breaker closed")
        elif self._state == self.State.CLOSED:
            self._failure_count = 0

    def record_failure(self):
        """Record a failed call"""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == self.State.HALF_OPEN:
            self._state = self.State.OPEN
            logger.warning("Circuit breaker opened (half-open test failed)")
        elif self._failure_count >= self.failure_threshold:
            self._state = self.State.OPEN
            logger.warning(
                f"Circuit breaker opened after {self._failure_count} failures"
            )

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute a function through the circuit breaker

        Args:
            func: Function to execute
            *args, **kwargs: Arguments for the function

        Returns:
            Function result

        Raises:
            CircuitBreakerError if circuit is open
        """
        if self.state == self.State.OPEN:
            raise CircuitBreakerError("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass


# =============================================================================
# Self Reflector
# =============================================================================


class SelfReflector:
    """
    LLM-based error analysis and self-reflection
    Uses the LLM to analyze errors and suggest fixes
    """

    def __init__(self, llm_handler=None):
        """
        Initialize the self-reflector

        Args:
            llm_handler: Optional LLMHandler instance for reflection
        """
        self._llm_handler = llm_handler

    def classify_error(self, error: Exception) -> ErrorType:
        """
        Classify an error into an error type

        Args:
            error: The exception to classify

        Returns:
            ErrorType classification
        """
        error_str = str(error).lower()

        if "connection" in error_str or "network" in error_str:
            return ErrorType.CONNECTION
        elif "timeout" in error_str:
            return ErrorType.TIMEOUT
        elif "rate" in error_str or "limit" in error_str or "429" in error_str:
            return ErrorType.RATE_LIMIT
        elif "auth" in error_str or "key" in error_str or "401" in error_str:
            return ErrorType.AUTHENTICATION
        elif "invalid" in error_str or "400" in error_str:
            return ErrorType.INVALID_REQUEST
        elif "model" in error_str or "llm" in error_str:
            return ErrorType.MODEL_ERROR
        elif "retrieval" in error_str or "vector" in error_str:
            return ErrorType.RETRIEVAL_ERROR
        elif "vision" in error_str or "image" in error_str or "llava" in error_str:
            return ErrorType.VISION_ERROR
        elif "audio" in error_str or "whisper" in error_str or "transcri" in error_str:
            return ErrorType.AUDIO_ERROR
        else:
            return ErrorType.UNKNOWN

    def reflect_on_error(
        self,
        error: Exception,
        context: Optional[str] = None
    ) -> ReflectionResult:
        """
        Analyze an error and provide reflection/suggestions

        Args:
            error: The exception to analyze
            context: Optional context about what was happening

        Returns:
            ReflectionResult with analysis and suggestions
        """
        error_type = self.classify_error(error)

        # Default reflections based on error type
        reflections = {
            ErrorType.CONNECTION: ReflectionResult(
                error_analysis="Network connection issue detected",
                suggested_fix="Check network connectivity and service availability",
                should_retry=True
            ),
            ErrorType.TIMEOUT: ReflectionResult(
                error_analysis="Request timed out",
                suggested_fix="Consider reducing request complexity or increasing timeout",
                should_retry=True
            ),
            ErrorType.RATE_LIMIT: ReflectionResult(
                error_analysis="Rate limit exceeded",
                suggested_fix="Wait before retrying, consider implementing backoff",
                should_retry=True
            ),
            ErrorType.AUTHENTICATION: ReflectionResult(
                error_analysis="Authentication failed",
                suggested_fix="Verify API keys and credentials are correct",
                should_retry=False
            ),
            ErrorType.INVALID_REQUEST: ReflectionResult(
                error_analysis="Invalid request parameters",
                suggested_fix="Check input format and required parameters",
                should_retry=False
            ),
            ErrorType.MODEL_ERROR: ReflectionResult(
                error_analysis="LLM model error",
                suggested_fix="Try a different model or simplify the request",
                should_retry=True
            ),
            ErrorType.RETRIEVAL_ERROR: ReflectionResult(
                error_analysis="Document retrieval failed",
                suggested_fix="Check vector store connection and data availability",
                should_retry=True
            ),
            ErrorType.VISION_ERROR: ReflectionResult(
                error_analysis="Vision model processing failed",
                suggested_fix="Check that LLaVA model is pulled in Ollama and image format is supported",
                should_retry=True
            ),
            ErrorType.AUDIO_ERROR: ReflectionResult(
                error_analysis="Audio processing failed",
                suggested_fix="Check audio format, file size, and that ffmpeg is installed",
                should_retry=False
            ),
            ErrorType.UNKNOWN: ReflectionResult(
                error_analysis=f"Unknown error: {str(error)}",
                suggested_fix="Review error details and logs for more information",
                should_retry=False
            )
        }

        result = reflections.get(error_type, reflections[ErrorType.UNKNOWN])

        # If LLM is available, use it for deeper analysis
        if self._llm_handler and error_type == ErrorType.UNKNOWN:
            try:
                analysis = self._analyze_with_llm(error, context)
                if analysis:
                    result = analysis
            except Exception as e:
                logger.debug(f"LLM analysis failed: {e}")

        return result

    def _analyze_with_llm(
        self,
        error: Exception,
        context: Optional[str]
    ) -> Optional[ReflectionResult]:
        """Use LLM to analyze error (if available)"""
        if not self._llm_handler:
            return None

        prompt = f"""Analyze this error and provide a brief suggestion:

Error: {str(error)}
Context: {context or 'Not provided'}

Provide:
1. Brief analysis (1 sentence)
2. Suggested fix (1 sentence)
3. Should retry? (yes/no)"""

        try:
            response = self._llm_handler.generate_response(prompt)
            # Parse response (simplified)
            lines = response.strip().split('\n')
            return ReflectionResult(
                error_analysis=lines[0] if lines else str(error),
                suggested_fix=lines[1] if len(lines) > 1 else "Check logs",
                should_retry="yes" in response.lower()
            )
        except Exception:
            return None

    def suggest_query_modification(
        self,
        original_query: str,
        error: Exception
    ) -> Optional[str]:
        """
        Suggest a modified query based on the error

        Args:
            original_query: The original user query
            error: The exception that occurred

        Returns:
            Modified query suggestion or None
        """
        error_type = self.classify_error(error)

        # Simple query modifications based on error type
        if error_type == ErrorType.INVALID_REQUEST:
            # Simplify the query
            if len(original_query) > 500:
                return original_query[:500] + "..."

        if error_type == ErrorType.MODEL_ERROR:
            # Try to simplify
            return f"Briefly answer: {original_query[:200]}"

        return None

    def auto_correct(
        self,
        response: str,
        feedback: str
    ) -> str:
        """
        Auto-correct a response based on feedback

        Args:
            response: Original response
            feedback: Feedback about what was wrong

        Returns:
            Corrected response or original if correction fails
        """
        if not self._llm_handler:
            return response

        prompt = f"""Correct this response based on the feedback:

Original response: {response}
Feedback: {feedback}

Provide only the corrected response:"""

        try:
            corrected = self._llm_handler.generate_response(prompt)
            return corrected
        except Exception as e:
            logger.warning(f"Auto-correction failed: {e}")
            return response


# =============================================================================
# Error Handler Facade
# =============================================================================


class ErrorHandler:
    """
    Facade for all error handling functionality
    """

    def __init__(self, llm_handler=None):
        """
        Initialize the error handler

        Args:
            llm_handler: Optional LLMHandler for self-reflection
        """
        self.retry_handler = RetryHandler()
        self.reflector = SelfReflector(llm_handler)
        self._circuit_breakers: dict = {}

    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a service"""
        if service_name not in self._circuit_breakers:
            self._circuit_breakers[service_name] = CircuitBreaker()
        return self._circuit_breakers[service_name]

    def handle_error(
        self,
        error: Exception,
        context: Optional[str] = None,
        query: Optional[str] = None
    ) -> ErrorContext:
        """
        Handle an error with full analysis

        Args:
            error: The exception to handle
            context: Optional context description
            query: Optional original query

        Returns:
            ErrorContext with full error information
        """
        error_type = self.reflector.classify_error(error)
        reflection = self.reflector.reflect_on_error(error, context)

        error_context = ErrorContext(
            error_type=error_type,
            message=str(error),
            original_error=error
        )

        # Log the error appropriately
        if error_type in [ErrorType.AUTHENTICATION, ErrorType.INVALID_REQUEST]:
            logger.error(f"Error ({error_type.value}): {error}")
        else:
            logger.warning(f"Error ({error_type.value}): {error}")

        return error_context

    def safe_execute(
        self,
        func: Callable[..., T],
        *args,
        fallback_value: Optional[T] = None,
        service_name: Optional[str] = None,
        **kwargs
    ) -> T:
        """
        Safely execute a function with all error handling

        Args:
            func: Function to execute
            *args: Function arguments
            fallback_value: Value to return if all recovery fails
            service_name: Optional service name for circuit breaker
            **kwargs: Function keyword arguments

        Returns:
            Function result or fallback value
        """
        try:
            # Use circuit breaker if service name provided
            if service_name:
                breaker = self.get_circuit_breaker(service_name)
                return breaker.call(
                    self.retry_handler.with_retry,
                    func, *args, **kwargs
                )
            else:
                return self.retry_handler.with_retry(func, *args, **kwargs)

        except CircuitBreakerError as e:
            logger.warning(f"Circuit breaker open for {service_name}: {e}")
            if fallback_value is not None:
                return fallback_value
            raise

        except Exception as e:
            self.handle_error(e)
            if fallback_value is not None:
                return fallback_value
            raise


# Create a global error handler instance
error_handler = ErrorHandler()
