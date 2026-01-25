"""
LLM Response Evaluation module using RAGAS
Provides industry-standard metrics for RAG evaluation
"""
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import config

logger = logging.getLogger(__name__)

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EvaluationResult:
    """Result of a single response evaluation"""
    query: str
    response: str
    contexts: List[str]
    ground_truth: Optional[str] = None

    # RAGAS Metrics (0-1 scale)
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None

    # Aggregate score
    overall_score: Optional[float] = None

    # Metadata
    evaluation_time_ms: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "query": self.query,
            "response": self.response[:200] + "..." if len(self.response) > 200 else self.response,
            "num_contexts": len(self.contexts),
            "has_ground_truth": self.ground_truth is not None,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "overall_score": self.overall_score,
            "evaluation_time_ms": self.evaluation_time_ms,
            "error": self.error
        }


@dataclass
class BatchEvaluationResult:
    """Result of batch evaluation"""
    results: List[EvaluationResult] = field(default_factory=list)
    avg_faithfulness: Optional[float] = None
    avg_answer_relevancy: Optional[float] = None
    avg_context_precision: Optional[float] = None
    avg_context_recall: Optional[float] = None
    avg_overall_score: Optional[float] = None
    total_time_ms: Optional[float] = None

    def compute_averages(self):
        """Compute average scores from individual results"""
        if not self.results:
            return

        def safe_avg(values: List[Optional[float]]) -> Optional[float]:
            valid = [v for v in values if v is not None]
            return sum(valid) / len(valid) if valid else None

        self.avg_faithfulness = safe_avg([r.faithfulness for r in self.results])
        self.avg_answer_relevancy = safe_avg([r.answer_relevancy for r in self.results])
        self.avg_context_precision = safe_avg([r.context_precision for r in self.results])
        self.avg_context_recall = safe_avg([r.context_recall for r in self.results])
        self.avg_overall_score = safe_avg([r.overall_score for r in self.results])


@dataclass
class Sample:
    """A sample for evaluation"""
    query: str
    response: str
    contexts: List[str]
    ground_truth: Optional[str] = None


# =============================================================================
# RAGAS Evaluator
# =============================================================================


class RAGASEvaluator:
    """
    RAGAS-based evaluation for RAG responses
    Uses industry-standard metrics for faithfulness, relevancy, and context quality
    """

    def __init__(self, model_name: str = None):
        """
        Initialize the RAGAS evaluator

        Args:
            model_name: Optional model name for evaluation LLM
        """
        self.model_name = model_name or config.EVALUATION_MODEL
        self._ragas_available = False
        self._metrics = None
        self._llm = None
        self._embeddings = None

        self._initialize_ragas()

    def _initialize_ragas(self):
        """Initialize RAGAS components"""
        try:
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            )
            from ragas import evaluate

            self._metrics = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall
            }

            # Initialize LLM for evaluation
            self._initialize_llm()

            self._ragas_available = True
            logger.info("RAGAS evaluator initialized successfully")

        except ImportError as e:
            logger.warning(f"RAGAS not available: {e}. Install with: pip install ragas")
            self._ragas_available = False
        except Exception as e:
            logger.error(f"Failed to initialize RAGAS: {e}")
            self._ragas_available = False

    def _initialize_llm(self):
        """Initialize LLM for RAGAS evaluation"""
        try:
            from langchain_ollama import OllamaLLM, OllamaEmbeddings

            self._llm = OllamaLLM(
                model=self.model_name,
                base_url=config.OLLAMA_BASE_URL,
                temperature=0.1  # Low temperature for evaluation
            )

            self._embeddings = OllamaEmbeddings(
                model=self.model_name,
                base_url=config.OLLAMA_BASE_URL
            )

            logger.debug(f"Initialized evaluation LLM: {self.model_name}")

        except Exception as e:
            logger.warning(f"Could not initialize evaluation LLM: {e}")

    @property
    def is_available(self) -> bool:
        """Check if RAGAS is available"""
        return self._ragas_available

    def evaluate_response(
        self,
        query: str,
        response: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate a single RAG response using RAGAS metrics

        Args:
            query: User query
            response: Generated response
            contexts: Retrieved context documents
            ground_truth: Optional ground truth answer

        Returns:
            EvaluationResult with metric scores
        """
        import time
        start_time = time.time()

        result = EvaluationResult(
            query=query,
            response=response,
            contexts=contexts,
            ground_truth=ground_truth
        )

        if not self._ragas_available:
            result.error = "RAGAS not available"
            result.evaluation_time_ms = (time.time() - start_time) * 1000
            return result

        try:
            from datasets import Dataset
            from ragas import evaluate

            # Prepare dataset for RAGAS
            data = {
                "question": [query],
                "answer": [response],
                "contexts": [contexts]
            }

            # Add ground truth if available (needed for context_recall)
            if ground_truth:
                data["ground_truth"] = [ground_truth]

            dataset = Dataset.from_dict(data)

            # Select metrics based on available data
            metrics_to_use = [
                self._metrics["faithfulness"],
                self._metrics["answer_relevancy"],
                self._metrics["context_precision"]
            ]

            if ground_truth:
                metrics_to_use.append(self._metrics["context_recall"])

            # Run evaluation
            eval_result = evaluate(
                dataset,
                metrics=metrics_to_use,
                llm=self._llm,
                embeddings=self._embeddings
            )

            # Extract scores
            result.faithfulness = float(eval_result.get("faithfulness", 0))
            result.answer_relevancy = float(eval_result.get("answer_relevancy", 0))
            result.context_precision = float(eval_result.get("context_precision", 0))

            if ground_truth:
                result.context_recall = float(eval_result.get("context_recall", 0))

            # Compute overall score (weighted average)
            scores = [
                result.faithfulness,
                result.answer_relevancy,
                result.context_precision
            ]
            if result.context_recall is not None:
                scores.append(result.context_recall)

            valid_scores = [s for s in scores if s is not None]
            result.overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            result.error = str(e)

        result.evaluation_time_ms = (time.time() - start_time) * 1000
        return result

    def batch_evaluate(self, samples: List[Sample]) -> BatchEvaluationResult:
        """
        Batch evaluation for multiple samples

        Args:
            samples: List of Sample objects to evaluate

        Returns:
            BatchEvaluationResult with individual and aggregate scores
        """
        import time
        start_time = time.time()

        batch_result = BatchEvaluationResult()

        if not self._ragas_available:
            logger.warning("RAGAS not available for batch evaluation")
            return batch_result

        try:
            from datasets import Dataset
            from ragas import evaluate

            # Prepare batch dataset
            data = {
                "question": [s.query for s in samples],
                "answer": [s.response for s in samples],
                "contexts": [s.contexts for s in samples]
            }

            # Check if any samples have ground truth
            has_ground_truth = any(s.ground_truth for s in samples)
            if has_ground_truth:
                data["ground_truth"] = [
                    s.ground_truth or "" for s in samples
                ]

            dataset = Dataset.from_dict(data)

            # Select metrics
            metrics_to_use = [
                self._metrics["faithfulness"],
                self._metrics["answer_relevancy"],
                self._metrics["context_precision"]
            ]

            if has_ground_truth:
                metrics_to_use.append(self._metrics["context_recall"])

            # Run batch evaluation
            eval_result = evaluate(
                dataset,
                metrics=metrics_to_use,
                llm=self._llm,
                embeddings=self._embeddings
            )

            # Process results
            for i, sample in enumerate(samples):
                result = EvaluationResult(
                    query=sample.query,
                    response=sample.response,
                    contexts=sample.contexts,
                    ground_truth=sample.ground_truth
                )

                # RAGAS returns DatasetDict, access by index
                if hasattr(eval_result, 'to_pandas'):
                    df = eval_result.to_pandas()
                    if i < len(df):
                        result.faithfulness = float(df.iloc[i].get("faithfulness", 0))
                        result.answer_relevancy = float(df.iloc[i].get("answer_relevancy", 0))
                        result.context_precision = float(df.iloc[i].get("context_precision", 0))
                        if has_ground_truth:
                            result.context_recall = float(df.iloc[i].get("context_recall", 0))

                # Compute overall score
                scores = [s for s in [
                    result.faithfulness,
                    result.answer_relevancy,
                    result.context_precision,
                    result.context_recall
                ] if s is not None]

                result.overall_score = sum(scores) / len(scores) if scores else None
                batch_result.results.append(result)

            # Compute averages
            batch_result.compute_averages()

        except Exception as e:
            logger.error(f"Batch evaluation failed: {e}")

        batch_result.total_time_ms = (time.time() - start_time) * 1000
        return batch_result


# =============================================================================
# Simple Evaluator (Fallback without RAGAS)
# =============================================================================


class SimpleEvaluator:
    """
    Simple heuristic-based evaluator as fallback when RAGAS is not available
    """

    def __init__(self):
        logger.info("Using SimpleEvaluator (RAGAS not available)")

    def evaluate_response(
        self,
        query: str,
        response: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> EvaluationResult:
        """
        Simple heuristic evaluation

        Args:
            query: User query
            response: Generated response
            contexts: Retrieved contexts
            ground_truth: Optional ground truth

        Returns:
            EvaluationResult with heuristic scores
        """
        result = EvaluationResult(
            query=query,
            response=response,
            contexts=contexts,
            ground_truth=ground_truth
        )

        # Simple heuristics
        # Faithfulness: Check if response words appear in context
        context_text = " ".join(contexts).lower()
        response_words = set(response.lower().split())
        context_words = set(context_text.split())

        if response_words:
            overlap = len(response_words & context_words)
            result.faithfulness = min(overlap / len(response_words), 1.0)
        else:
            result.faithfulness = 0.0

        # Answer relevancy: Check query-response word overlap
        query_words = set(query.lower().split())
        if query_words:
            relevancy_overlap = len(query_words & response_words)
            result.answer_relevancy = min(relevancy_overlap / len(query_words), 1.0)
        else:
            result.answer_relevancy = 0.0

        # Context precision: Simple length-based heuristic
        if contexts:
            avg_context_len = sum(len(c) for c in contexts) / len(contexts)
            # Assume longer contexts are more informative (simple heuristic)
            result.context_precision = min(avg_context_len / 500, 1.0)
        else:
            result.context_precision = 0.0

        # Context recall: Compare with ground truth if available
        if ground_truth:
            gt_words = set(ground_truth.lower().split())
            if gt_words:
                recall_overlap = len(gt_words & context_words)
                result.context_recall = recall_overlap / len(gt_words)
            else:
                result.context_recall = 0.0

        # Overall score
        scores = [
            result.faithfulness,
            result.answer_relevancy,
            result.context_precision
        ]
        if result.context_recall is not None:
            scores.append(result.context_recall)

        result.overall_score = sum(scores) / len(scores) if scores else 0.0
        result.evaluation_time_ms = 0.1  # Nearly instant

        return result


# =============================================================================
# Evaluator Factory
# =============================================================================


def get_evaluator() -> Any:
    """
    Get the appropriate evaluator based on availability

    Returns:
        RAGASEvaluator if available, otherwise SimpleEvaluator
    """
    evaluator = RAGASEvaluator()
    if evaluator.is_available:
        return evaluator

    logger.info("Falling back to SimpleEvaluator")
    return SimpleEvaluator()


# Create a global evaluator instance
evaluator = get_evaluator()
