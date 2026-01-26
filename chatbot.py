"""
Main chatbot module that orchestrates all components
Enhanced with error handling, monitoring, and evaluation
"""
import time
from typing import Optional, Dict, Any, List
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from llm_handler import LLMHandler
from error_handler import error_handler, ErrorType
from monitoring import langfuse_monitor, prometheus_metrics, debug_tools
from evaluator import evaluator, simple_evaluator, EvaluationResult
from llm_judge import llm_judge, JudgmentCriteria
import config
import logging

logger = logging.getLogger(__name__)


class RAGChatbot:
    """
    Main chatbot class that integrates all components
    Enhanced with evaluation, monitoring, and error handling
    """

    def __init__(self, enable_evaluation: bool = None, use_fast_evaluation: bool = False):
        """
        Initialize the RAG chatbot with all components

        Args:
            enable_evaluation: Whether to enable response evaluation
            use_fast_evaluation: Use fast heuristic evaluation instead of RAGAS
        """
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.llm_handler = LLMHandler()
        self._is_initialized = False

        # Evaluation settings
        self._enable_evaluation = (
            enable_evaluation if enable_evaluation is not None
            else config.EVALUATION_ENABLED
        )
        self._use_fast_evaluation = use_fast_evaluation
        self._last_evaluation: Optional[EvaluationResult] = None
        self._evaluation_history: List[Dict[str, Any]] = []

        # Session tracking
        self._session_id = self._generate_session_id()
        self._query_count = 0

        eval_type = "fast" if use_fast_evaluation else "RAGAS"
        logger.info(f"Initialized RAG Chatbot (evaluation: {self._enable_evaluation}, type: {eval_type})")

    def _generate_session_id(self) -> str:
        """Generate a unique session ID"""
        import uuid
        return str(uuid.uuid4())[:8]

    def process_pdfs(self, uploaded_files: List) -> bool:
        """
        Process uploaded PDF files and create vector store

        Args:
            uploaded_files: List of uploaded PDF files

        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()

        try:
            # Process PDFs into chunks
            if isinstance(uploaded_files, list):
                chunks = self.document_processor.process_multiple_pdfs(uploaded_files)
            else:
                chunks = self.document_processor.process_pdf(uploaded_files)

            # Create vector store
            self.vector_store_manager.create_vector_store(chunks)

            # Initialize QA chain
            retriever = self.vector_store_manager.get_retriever()
            self.llm_handler.create_qa_chain(retriever)

            self._is_initialized = True

            # Record metrics
            elapsed_ms = (time.time() - start_time) * 1000
            prometheus_metrics.increment_documents_processed(len(uploaded_files) if isinstance(uploaded_files, list) else 1)
            prometheus_metrics.observe_latency("pdf_processing", elapsed_ms / 1000)

            logger.info(f"Successfully processed PDFs in {elapsed_ms:.2f}ms")
            return True

        except Exception as e:
            logger.error(f"Error processing PDFs: {e}")
            error_handler.handle_error(e, "PDF processing")
            prometheus_metrics.increment_error("document_processing")
            return False

    def add_pdfs(self, uploaded_files: List) -> bool:
        """
        Add additional PDFs to existing vector store

        Args:
            uploaded_files: List of uploaded PDF files

        Returns:
            True if successful, False otherwise
        """
        try:
            # Process new PDFs
            if isinstance(uploaded_files, list):
                chunks = self.document_processor.process_multiple_pdfs(uploaded_files)
            else:
                chunks = self.document_processor.process_pdf(uploaded_files)

            # Add to vector store
            self.vector_store_manager.add_documents(chunks)

            # Reinitialize QA chain with updated retriever
            retriever = self.vector_store_manager.get_retriever()
            self.llm_handler.create_qa_chain(retriever)

            prometheus_metrics.increment_documents_processed(
                len(uploaded_files) if isinstance(uploaded_files, list) else 1
            )

            logger.info("Successfully added new PDFs to chatbot")
            return True

        except Exception as e:
            logger.error(f"Error adding PDFs: {e}")
            error_handler.handle_error(e, "Adding PDFs")
            return False

    def chat(
        self,
        question: str,
        evaluate: bool = None
    ) -> Dict[str, Any]:
        """
        Chat with the bot about the uploaded documents

        Args:
            question: User question
            evaluate: Whether to evaluate this response (overrides default)

        Returns:
            Dictionary with response, source documents, and optional evaluation
        """
        if not self._is_initialized:
            return {
                "result": "Please upload a PDF file first to start chatting.",
                "source_documents": [],
                "evaluation": None
            }

        self._query_count += 1
        start_time = time.time()
        should_evaluate = evaluate if evaluate is not None else self._enable_evaluation

        try:
            # Get response from LLM
            response = self.llm_handler.query(question)

            # Extract contexts for evaluation
            contexts = [
                doc.page_content
                for doc in response.get("source_documents", [])
            ]

            # Evaluate if enabled
            evaluation_result = None
            if should_evaluate and contexts:
                evaluation_result = self._evaluate_response(
                    question, response["result"], contexts
                )
                response["evaluation"] = evaluation_result

                # Record evaluation metrics
                if evaluation_result and evaluation_result.overall_score:
                    prometheus_metrics.record_evaluation_score(
                        "overall", evaluation_result.overall_score
                    )

                # Add scores to monitoring trace
                if response.get("trace_id") and evaluation_result:
                    scores = {}
                    if evaluation_result.faithfulness:
                        scores["faithfulness"] = evaluation_result.faithfulness
                    if evaluation_result.answer_relevancy:
                        scores["answer_relevancy"] = evaluation_result.answer_relevancy
                    if scores:
                        langfuse_monitor.score_response(response["trace_id"], scores)

            # Record debug info
            elapsed_ms = (time.time() - start_time) * 1000
            debug_tools.profile_query(
                total_time_ms=elapsed_ms,
                retrieval_time_ms=elapsed_ms * 0.3,  # Estimated
                llm_time_ms=elapsed_ms * 0.7  # Estimated
            )

            return response

        except Exception as e:
            logger.error(f"Error during chat: {e}")
            error_context = error_handler.handle_error(e, "Chat", question)

            return {
                "result": f"Sorry, I encountered an error: {str(e)}",
                "source_documents": [],
                "evaluation": None,
                "error": str(e)
            }

    def _evaluate_response(
        self,
        query: str,
        response: str,
        contexts: List[str]
    ) -> Optional[EvaluationResult]:
        """
        Evaluate a response using RAGAS metrics or fast heuristics

        Args:
            query: User query
            response: Generated response
            contexts: Retrieved contexts

        Returns:
            EvaluationResult or None if evaluation fails
        """
        try:
            # Choose evaluator based on setting
            eval_to_use = simple_evaluator if self._use_fast_evaluation else evaluator
            result = eval_to_use.evaluate_response(query, response, contexts)

            # If RAGAS evaluation failed or returned all None, fall back to simple
            if not self._use_fast_evaluation:
                if result.error or (result.faithfulness is None and result.answer_relevancy is None):
                    logger.info("RAGAS evaluation failed, falling back to simple evaluator")
                    result = simple_evaluator.evaluate_response(query, response, contexts)

            self._last_evaluation = result

            # Store in history
            self._evaluation_history.append({
                "query": query[:100],
                "overall_score": result.overall_score,
                "faithfulness": result.faithfulness,
                "relevancy": result.answer_relevancy,
                "timestamp": time.time()
            })

            # Keep only last 100 evaluations
            if len(self._evaluation_history) > 100:
                self._evaluation_history = self._evaluation_history[-100:]

            logger.debug(f"Evaluation complete: score={result.overall_score}")
            return result

        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            # Last resort: try simple evaluator
            try:
                result = simple_evaluator.evaluate_response(query, response, contexts)
                self._last_evaluation = result
                return result
            except Exception:
                return None

    def judge_response(
        self,
        question: str,
        response: str,
        contexts: List[str],
        criteria: JudgmentCriteria = JudgmentCriteria.CORRECTNESS
    ) -> Dict[str, Any]:
        """
        Judge a response using LLM-as-a-Judge

        Args:
            question: User question
            response: Generated response
            contexts: Retrieved contexts
            criteria: Evaluation criteria

        Returns:
            Dictionary with judgment results
        """
        try:
            result = llm_judge.judge_response(question, response, contexts, criteria)
            return {
                "score": result.score,
                "reason": result.reason,
                "criteria": result.criteria
            }
        except Exception as e:
            logger.warning(f"Judgment failed: {e}")
            return {
                "score": None,
                "reason": str(e),
                "criteria": criteria.value
            }

    def compare_responses(
        self,
        question: str,
        response_a: str,
        response_b: str
    ) -> Dict[str, Any]:
        """
        Compare two responses using LLM-as-a-Judge

        Args:
            question: User question
            response_a: First response
            response_b: Second response

        Returns:
            Dictionary with comparison results
        """
        # Get contexts for comparison
        contexts = []
        if self._is_initialized:
            docs = self.search_documents(question, k=4)
            contexts = [doc.page_content for doc in docs]

        try:
            result = llm_judge.compare_responses(
                question, response_a, response_b, contexts
            )
            return {
                "winner": result.winner,
                "score_a": result.score_a,
                "score_b": result.score_b,
                "reason": result.reason
            }
        except Exception as e:
            logger.warning(f"Comparison failed: {e}")
            return {
                "winner": "tie",
                "reason": str(e)
            }

    def search_documents(self, query: str, k: int = 4) -> List:
        """
        Search for relevant documents

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of relevant documents
        """
        return self.vector_store_manager.search(query, k)

    def clear_chat_history(self):
        """Clear the conversation history"""
        self.llm_handler.clear_memory()
        logger.info("Cleared chat history")

    def get_chat_history(self) -> list:
        """
        Get conversation history

        Returns:
            List of conversation messages
        """
        return self.llm_handler.get_conversation_history()

    def reset(self):
        """Reset the entire chatbot state"""
        self.vector_store_manager.clear_vector_store()
        self.llm_handler.clear_memory()
        self._is_initialized = False
        self._evaluation_history = []
        self._last_evaluation = None
        self._query_count = 0
        logger.info("Reset chatbot state")

    @property
    def is_ready(self) -> bool:
        """Check if chatbot is ready for queries"""
        return self._is_initialized

    @property
    def evaluation_enabled(self) -> bool:
        """Check if evaluation is enabled"""
        return self._enable_evaluation

    @evaluation_enabled.setter
    def evaluation_enabled(self, value: bool):
        """Set evaluation enabled status"""
        self._enable_evaluation = value
        logger.info(f"Evaluation {'enabled' if value else 'disabled'}")

    @property
    def fast_evaluation(self) -> bool:
        """Check if fast evaluation is enabled"""
        return self._use_fast_evaluation

    @fast_evaluation.setter
    def fast_evaluation(self, value: bool):
        """Set fast evaluation mode"""
        self._use_fast_evaluation = value
        logger.info(f"Fast evaluation {'enabled' if value else 'disabled (using RAGAS)'}")

    @property
    def last_evaluation(self) -> Optional[EvaluationResult]:
        """Get the last evaluation result"""
        return self._last_evaluation

    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Get evaluation history"""
        return self._evaluation_history

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of evaluation history"""
        if not self._evaluation_history:
            return {
                "total_evaluations": 0,
                "avg_overall_score": None,
                "avg_faithfulness": None,
                "avg_relevancy": None
            }

        def safe_avg(key: str) -> Optional[float]:
            values = [e[key] for e in self._evaluation_history if e.get(key) is not None]
            return sum(values) / len(values) if values else None

        return {
            "total_evaluations": len(self._evaluation_history),
            "avg_overall_score": safe_avg("overall_score"),
            "avg_faithfulness": safe_avg("faithfulness"),
            "avg_relevancy": safe_avg("relevancy")
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive chatbot statistics"""
        return {
            "session_id": self._session_id,
            "is_ready": self._is_initialized,
            "query_count": self._query_count,
            "evaluation_enabled": self._enable_evaluation,
            "evaluation_summary": self.get_evaluation_summary(),
            "llm_stats": self.llm_handler.get_performance_stats(),
            "debug_profile": debug_tools.get_average_profile()
        }

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information for troubleshooting"""
        return {
            "session_id": self._session_id,
            "provider": self.llm_handler.current_provider,
            "model_info": self.llm_handler.model_info,
            "is_initialized": self._is_initialized,
            "cache_stats": self.llm_handler.get_performance_stats().get("cache_stats", {}),
            "evaluation_count": len(self._evaluation_history),
            "last_evaluation_score": (
                self._last_evaluation.overall_score if self._last_evaluation else None
            )
        }
