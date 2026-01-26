"""
LLM-as-a-Judge implementation using DeepEval with G-Eval
Provides reliable LLM-based scoring with Chain-of-Thought reasoning
"""
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import config

logger = logging.getLogger(__name__)

# =============================================================================
# Data Classes
# =============================================================================


class JudgmentCriteria(Enum):
    """Predefined evaluation criteria"""
    CORRECTNESS = "correctness"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    HELPFULNESS = "helpfulness"
    HARMLESSNESS = "harmlessness"
    COMPLETENESS = "completeness"


@dataclass
class GEvalResult:
    """Result from G-Eval scoring"""
    score: float  # 0-1 scale
    reason: str
    criteria: str
    evaluation_steps: List[str]
    raw_score: Optional[int] = None  # Original 1-10 scale


@dataclass
class ComparisonResult:
    """Result from pairwise comparison"""
    winner: str  # "A", "B", or "tie"
    score_a: float
    score_b: float
    reason: str


@dataclass
class JudgmentResult:
    """Complete judgment result"""
    query: str
    response: str
    scores: Dict[str, GEvalResult]
    overall_score: float
    recommendation: str


# =============================================================================
# Ollama Judge Model Wrapper
# =============================================================================


class OllamaJudge:
    """
    Custom Ollama wrapper for DeepEval
    Implements the DeepEvalBaseLLM interface
    """

    def __init__(self, model: str = None):
        """
        Initialize the Ollama judge

        Args:
            model: Model name for judging
        """
        self.model = model or config.EVALUATION_MODEL
        self.base_url = config.OLLAMA_BASE_URL
        self._llm = None

        self._initialize()

    def _initialize(self):
        """Initialize the Ollama LLM"""
        try:
            from langchain_ollama import OllamaLLM

            self._llm = OllamaLLM(
                model=self.model,
                base_url=self.base_url,
                temperature=0.1  # Low temperature for consistent judging
            )
            logger.info(f"Ollama judge initialized with model: {self.model}")

        except ImportError:
            logger.warning("langchain_ollama not available")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama judge: {e}")

    def generate(self, prompt: str) -> str:
        """
        Generate response from the judge model

        Args:
            prompt: Input prompt

        Returns:
            Generated response
        """
        if self._llm is None:
            raise RuntimeError("Ollama LLM not initialized")

        return self._llm.invoke(prompt)

    def get_model_name(self) -> str:
        """Get the model name"""
        return self.model

    async def a_generate(self, prompt: str) -> str:
        """Async generation (wraps sync for compatibility)"""
        return self.generate(prompt)


# =============================================================================
# LLM Judge
# =============================================================================


class LLMJudge:
    """
    LLM-as-a-Judge implementation using G-Eval methodology
    Provides stable, explainable scoring with Chain-of-Thought reasoning
    """

    def __init__(self, model: str = None):
        """
        Initialize the LLM judge

        Args:
            model: Model name for judging
        """
        self.model = model or config.EVALUATION_MODEL
        self._judge_model = OllamaJudge(self.model)
        self._deepeval_available = False

        self._check_deepeval()

    def _check_deepeval(self):
        """Check if DeepEval is available"""
        try:
            from deepeval.metrics import GEval
            from deepeval.test_case import LLMTestCase
            self._deepeval_available = True
            logger.info("DeepEval available for LLM judging")
        except ImportError:
            logger.warning("DeepEval not available. Install with: pip install deepeval")
            self._deepeval_available = False

    @property
    def is_available(self) -> bool:
        """Check if judge is available"""
        return self._judge_model._llm is not None

    def judge_response(
        self,
        query: str,
        response: str,
        context: List[str],
        criteria: JudgmentCriteria = JudgmentCriteria.CORRECTNESS
    ) -> GEvalResult:
        """
        Judge a response using G-Eval methodology

        Args:
            query: User query
            response: Generated response
            context: Retrieved context
            criteria: Evaluation criteria

        Returns:
            GEvalResult with score and reasoning
        """
        if self._deepeval_available:
            return self._judge_with_deepeval(query, response, context, criteria)
        else:
            return self._judge_with_prompt(query, response, context, criteria)

    def _judge_with_deepeval(
        self,
        query: str,
        response: str,
        context: List[str],
        criteria: JudgmentCriteria
    ) -> GEvalResult:
        """Use DeepEval's G-Eval for scoring"""
        try:
            from deepeval.metrics import GEval
            from deepeval.test_case import LLMTestCase, LLMTestCaseParams

            # Define evaluation criteria and steps
            criteria_configs = {
                JudgmentCriteria.CORRECTNESS: {
                    "name": "Correctness",
                    "criteria": "Determine if the response correctly answers the question based on the provided context.",
                    "steps": [
                        "Check if the response directly addresses the question",
                        "Verify factual accuracy against the context",
                        "Assess if key information is included",
                        "Check for contradictions with context"
                    ]
                },
                JudgmentCriteria.RELEVANCE: {
                    "name": "Relevance",
                    "criteria": "Evaluate how relevant the response is to the user's question.",
                    "steps": [
                        "Identify the main intent of the question",
                        "Check if response addresses that intent",
                        "Assess if response stays on topic",
                        "Check for unnecessary information"
                    ]
                },
                JudgmentCriteria.COHERENCE: {
                    "name": "Coherence",
                    "criteria": "Evaluate the logical flow and clarity of the response.",
                    "steps": [
                        "Check for logical structure",
                        "Assess sentence-level clarity",
                        "Verify transitions between ideas",
                        "Check overall readability"
                    ]
                },
                JudgmentCriteria.HELPFULNESS: {
                    "name": "Helpfulness",
                    "criteria": "Determine how helpful the response is for the user.",
                    "steps": [
                        "Check if response provides actionable information",
                        "Assess completeness of the answer",
                        "Evaluate practical usefulness",
                        "Check if it addresses user's likely needs"
                    ]
                },
                JudgmentCriteria.COMPLETENESS: {
                    "name": "Completeness",
                    "criteria": "Evaluate if the response fully addresses all aspects of the question.",
                    "steps": [
                        "Identify all parts of the question",
                        "Check if each part is addressed",
                        "Assess depth of coverage",
                        "Check for missing important details"
                    ]
                }
            }

            config_item = criteria_configs.get(criteria, criteria_configs[JudgmentCriteria.CORRECTNESS])

            # Create G-Eval metric with required evaluation_params
            metric = GEval(
                name=config_item["name"],
                criteria=config_item["criteria"],
                evaluation_steps=config_item["steps"],
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.CONTEXT
                ],
                model=self._judge_model.get_model_name(),
                async_mode=False  # Use sync mode for Ollama
            )

            # Create test case
            test_case = LLMTestCase(
                input=query,
                actual_output=response,
                context=context
            )

            # Run evaluation
            metric.measure(test_case)

            return GEvalResult(
                score=metric.score,
                reason=metric.reason if hasattr(metric, 'reason') else "No reason provided",
                criteria=config_item["name"],
                evaluation_steps=config_item["steps"],
                raw_score=int(metric.score * 10) if metric.score else None
            )

        except Exception as e:
            logger.error(f"DeepEval G-Eval failed: {e}")
            return self._judge_with_prompt(query, response, context, criteria)

    def _judge_with_prompt(
        self,
        query: str,
        response: str,
        context: List[str],
        criteria: JudgmentCriteria
    ) -> GEvalResult:
        """Fallback: Use direct prompting for evaluation"""
        context_text = "\n".join(context[:3])  # Limit context

        evaluation_steps = [
            "Read the question carefully",
            "Review the provided context",
            "Analyze the response for accuracy",
            "Determine a score from 1-10"
        ]

        prompt = f"""You are an expert evaluator. Evaluate the following response.

QUESTION: {query}

CONTEXT:
{context_text}

RESPONSE TO EVALUATE:
{response}

EVALUATION CRITERIA: {criteria.value}

Please follow these steps:
1. {evaluation_steps[0]}
2. {evaluation_steps[1]}
3. {evaluation_steps[2]}
4. {evaluation_steps[3]}

Provide your evaluation in this format:
SCORE: [1-10]
REASON: [Your detailed reasoning]"""

        try:
            result = self._judge_model.generate(prompt)

            # Parse result
            score = 5  # Default
            reason = result

            if "SCORE:" in result:
                try:
                    score_line = result.split("SCORE:")[1].split("\n")[0]
                    score = int(''.join(filter(str.isdigit, score_line[:5])))
                    score = min(max(score, 1), 10)
                except (ValueError, IndexError):
                    pass

            if "REASON:" in result:
                reason = result.split("REASON:")[1].strip()

            return GEvalResult(
                score=score / 10.0,  # Convert to 0-1
                reason=reason,
                criteria=criteria.value,
                evaluation_steps=evaluation_steps,
                raw_score=score
            )

        except Exception as e:
            logger.error(f"Prompt-based judging failed: {e}")
            return GEvalResult(
                score=0.5,
                reason=f"Evaluation failed: {str(e)}",
                criteria=criteria.value,
                evaluation_steps=evaluation_steps,
                raw_score=5
            )

    def judge_multiple_criteria(
        self,
        query: str,
        response: str,
        context: List[str],
        criteria_list: List[JudgmentCriteria] = None
    ) -> JudgmentResult:
        """
        Judge a response on multiple criteria

        Args:
            query: User query
            response: Generated response
            context: Retrieved context
            criteria_list: List of criteria to evaluate

        Returns:
            JudgmentResult with all scores
        """
        if criteria_list is None:
            criteria_list = [
                JudgmentCriteria.CORRECTNESS,
                JudgmentCriteria.RELEVANCE,
                JudgmentCriteria.HELPFULNESS
            ]

        scores = {}
        for criteria in criteria_list:
            scores[criteria.value] = self.judge_response(
                query, response, context, criteria
            )

        # Calculate overall score
        valid_scores = [s.score for s in scores.values() if s.score is not None]
        overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        # Generate recommendation
        if overall_score >= 0.8:
            recommendation = "Excellent response - meets quality standards"
        elif overall_score >= 0.6:
            recommendation = "Good response - minor improvements possible"
        elif overall_score >= 0.4:
            recommendation = "Acceptable response - consider revising"
        else:
            recommendation = "Poor response - significant revision needed"

        return JudgmentResult(
            query=query,
            response=response,
            scores=scores,
            overall_score=overall_score,
            recommendation=recommendation
        )

    def compare_responses(
        self,
        query: str,
        response_a: str,
        response_b: str,
        context: List[str]
    ) -> ComparisonResult:
        """
        Compare two responses using pairwise comparison

        Args:
            query: User query
            response_a: First response
            response_b: Second response
            context: Retrieved context

        Returns:
            ComparisonResult indicating which response is better
        """
        context_text = "\n".join(context[:3])

        prompt = f"""You are comparing two responses to the same question. Determine which is better.

QUESTION: {query}

CONTEXT:
{context_text}

RESPONSE A:
{response_a}

RESPONSE B:
{response_b}

Compare these responses on:
1. Correctness - which is more accurate?
2. Relevance - which better addresses the question?
3. Clarity - which is clearer?

Provide your judgment in this format:
WINNER: [A/B/TIE]
SCORE_A: [1-10]
SCORE_B: [1-10]
REASON: [Your explanation]"""

        try:
            result = self._judge_model.generate(prompt)

            # Parse result
            winner = "tie"
            score_a = 5
            score_b = 5
            reason = result

            if "WINNER:" in result:
                winner_line = result.split("WINNER:")[1].split("\n")[0].upper()
                if "A" in winner_line:
                    winner = "A"
                elif "B" in winner_line:
                    winner = "B"
                else:
                    winner = "tie"

            if "SCORE_A:" in result:
                try:
                    score_line = result.split("SCORE_A:")[1].split("\n")[0]
                    score_a = int(''.join(filter(str.isdigit, score_line[:5])))
                except (ValueError, IndexError):
                    pass

            if "SCORE_B:" in result:
                try:
                    score_line = result.split("SCORE_B:")[1].split("\n")[0]
                    score_b = int(''.join(filter(str.isdigit, score_line[:5])))
                except (ValueError, IndexError):
                    pass

            if "REASON:" in result:
                reason = result.split("REASON:")[1].strip()

            return ComparisonResult(
                winner=winner,
                score_a=score_a / 10.0,
                score_b=score_b / 10.0,
                reason=reason
            )

        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            return ComparisonResult(
                winner="tie",
                score_a=0.5,
                score_b=0.5,
                reason=f"Comparison failed: {str(e)}"
            )


# =============================================================================
# Quick Evaluation Functions
# =============================================================================


def quick_judge(
    query: str,
    response: str,
    context: List[str]
) -> Dict[str, Any]:
    """
    Quick evaluation of a response

    Args:
        query: User query
        response: Generated response
        context: Retrieved context

    Returns:
        Dictionary with score and basic feedback
    """
    judge = LLMJudge()

    if not judge.is_available:
        return {
            "score": None,
            "error": "Judge not available"
        }

    result = judge.judge_response(query, response, context)

    return {
        "score": result.score,
        "reason": result.reason,
        "criteria": result.criteria
    }


def quick_compare(
    query: str,
    response_a: str,
    response_b: str,
    context: List[str]
) -> str:
    """
    Quick comparison of two responses

    Args:
        query: User query
        response_a: First response
        response_b: Second response
        context: Retrieved context

    Returns:
        Winner string ("A", "B", or "tie")
    """
    judge = LLMJudge()

    if not judge.is_available:
        return "tie"

    result = judge.compare_responses(query, response_a, response_b, context)
    return result.winner


# Create a global judge instance
llm_judge = LLMJudge()
