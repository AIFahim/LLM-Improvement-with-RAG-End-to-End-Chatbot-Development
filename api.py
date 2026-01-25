"""
FastAPI deployment for the RAG Chatbot
Provides RESTful API endpoints for chat, upload, evaluation, and metrics
"""
import os
import time
import tempfile
import logging
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from chatbot import RAGChatbot
from monitoring import prometheus_metrics, langfuse_monitor
from evaluator import evaluator
from llm_judge import llm_judge, JudgmentCriteria
import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Pydantic Models
# =============================================================================


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    query: str = Field(..., min_length=1, max_length=5000, description="User query")
    evaluate: bool = Field(False, description="Whether to evaluate the response")
    session_id: Optional[str] = Field(None, description="Optional session ID")


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    result: str
    sources: List[Dict[str, Any]] = []
    cached: bool = False
    evaluation: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None
    latency_ms: Optional[float] = None


class UploadResponse(BaseModel):
    """Response model for upload endpoint"""
    success: bool
    message: str
    files_processed: int = 0
    chunks_created: int = 0


class EvaluationRequest(BaseModel):
    """Request model for evaluation endpoint"""
    query: str
    response: str
    contexts: List[str]
    ground_truth: Optional[str] = None


class EvaluationResponse(BaseModel):
    """Response model for evaluation endpoint"""
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    overall_score: Optional[float] = None
    evaluation_time_ms: Optional[float] = None
    error: Optional[str] = None


class JudgeRequest(BaseModel):
    """Request model for LLM judge endpoint"""
    query: str
    response: str
    contexts: List[str]
    criteria: str = "correctness"


class JudgeResponse(BaseModel):
    """Response model for LLM judge endpoint"""
    score: Optional[float] = None
    reason: str
    criteria: str


class CompareRequest(BaseModel):
    """Request model for comparison endpoint"""
    query: str
    response_a: str
    response_b: str
    contexts: List[str] = []


class CompareResponse(BaseModel):
    """Response model for comparison endpoint"""
    winner: str
    score_a: Optional[float] = None
    score_b: Optional[float] = None
    reason: str


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    chatbot_ready: bool
    provider: str
    model: str
    version: str = "1.0.0"


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint"""
    chatbot_stats: Dict[str, Any]
    cache_stats: Dict[str, Any]
    evaluation_summary: Dict[str, Any]


# =============================================================================
# Application Lifespan
# =============================================================================

# Global chatbot instance
chatbot: Optional[RAGChatbot] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global chatbot

    # Startup
    logger.info("Starting RAG Chatbot API...")
    chatbot = RAGChatbot()

    # Validate environment
    if not config.validate_environment():
        logger.warning("Environment validation failed, but continuing...")

    logger.info(f"API ready with provider: {chatbot.llm_handler.current_provider}")

    yield

    # Shutdown
    logger.info("Shutting down RAG Chatbot API...")
    if langfuse_monitor.is_enabled:
        langfuse_monitor.flush()


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="RAG Chatbot API",
    description="API for the RAG-based PDF chatbot with evaluation and monitoring",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "RAG Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    global chatbot

    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    model_info = chatbot.llm_handler.model_info

    return HealthResponse(
        status="healthy",
        chatbot_ready=chatbot.is_ready,
        provider=model_info["provider"],
        model=model_info["model"]
    )


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_pdf(files: List[UploadFile] = File(...)):
    """
    Upload PDF files for processing

    Accepts one or more PDF files and processes them for Q&A
    """
    global chatbot

    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Validate files
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.filename}. Only PDF files are accepted."
            )

    try:
        # Save files temporarily
        temp_files = []
        for file in files:
            # Create a temporary file-like object
            content = await file.read()

            # Create a simple object that mimics uploaded file
            class TempFile:
                def __init__(self, name, content):
                    self.name = name
                    self._content = content

                def read(self):
                    return self._content

                def seek(self, pos):
                    pass

            temp_files.append(TempFile(file.filename, content))

        # Process the PDFs
        success = chatbot.process_pdfs(temp_files)

        if success:
            return UploadResponse(
                success=True,
                message="PDFs processed successfully",
                files_processed=len(files)
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to process PDFs")

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Chat with the bot about uploaded documents

    Requires documents to be uploaded first via /upload
    """
    global chatbot

    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    if not chatbot.is_ready:
        raise HTTPException(
            status_code=400,
            detail="No documents loaded. Please upload PDFs first via /upload"
        )

    start_time = time.time()

    try:
        response = chatbot.chat(request.query, evaluate=request.evaluate)

        latency_ms = (time.time() - start_time) * 1000

        # Format sources
        sources = []
        for doc in response.get("source_documents", []):
            sources.append({
                "content": doc.page_content[:500],
                "page": doc.metadata.get("page", "Unknown"),
                "source": doc.metadata.get("source", "Unknown")
            })

        # Format evaluation if present
        evaluation = None
        if response.get("evaluation"):
            eval_result = response["evaluation"]
            evaluation = {
                "faithfulness": eval_result.faithfulness,
                "answer_relevancy": eval_result.answer_relevancy,
                "context_precision": eval_result.context_precision,
                "overall_score": eval_result.overall_score
            }

        return ChatResponse(
            result=response.get("result", ""),
            sources=sources,
            cached=response.get("cached", False),
            evaluation=evaluation,
            trace_id=response.get("trace_id"),
            latency_ms=latency_ms
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        prometheus_metrics.increment_error("api_chat")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate", response_model=EvaluationResponse, tags=["Evaluation"])
async def evaluate_response(request: EvaluationRequest):
    """
    Evaluate a response using RAGAS metrics

    Standalone evaluation endpoint for custom responses
    """
    try:
        result = evaluator.evaluate_response(
            query=request.query,
            response=request.response,
            contexts=request.contexts,
            ground_truth=request.ground_truth
        )

        return EvaluationResponse(
            faithfulness=result.faithfulness,
            answer_relevancy=result.answer_relevancy,
            context_precision=result.context_precision,
            context_recall=result.context_recall,
            overall_score=result.overall_score,
            evaluation_time_ms=result.evaluation_time_ms,
            error=result.error
        )

    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return EvaluationResponse(error=str(e))


@app.post("/judge", response_model=JudgeResponse, tags=["Evaluation"])
async def judge_response(request: JudgeRequest):
    """
    Judge a response using LLM-as-a-Judge

    Uses G-Eval methodology for reliable scoring
    """
    try:
        # Parse criteria
        criteria_map = {
            "correctness": JudgmentCriteria.CORRECTNESS,
            "relevance": JudgmentCriteria.RELEVANCE,
            "coherence": JudgmentCriteria.COHERENCE,
            "helpfulness": JudgmentCriteria.HELPFULNESS,
            "completeness": JudgmentCriteria.COMPLETENESS
        }
        criteria = criteria_map.get(request.criteria.lower(), JudgmentCriteria.CORRECTNESS)

        result = llm_judge.judge_response(
            query=request.query,
            response=request.response,
            context=request.contexts,
            criteria=criteria
        )

        return JudgeResponse(
            score=result.score,
            reason=result.reason,
            criteria=result.criteria
        )

    except Exception as e:
        logger.error(f"Judge error: {e}")
        return JudgeResponse(
            score=None,
            reason=str(e),
            criteria=request.criteria
        )


@app.post("/compare", response_model=CompareResponse, tags=["Evaluation"])
async def compare_responses(request: CompareRequest):
    """
    Compare two responses using LLM-as-a-Judge

    Useful for A/B testing different prompts or models
    """
    try:
        result = llm_judge.compare_responses(
            query=request.query,
            response_a=request.response_a,
            response_b=request.response_b,
            context=request.contexts
        )

        return CompareResponse(
            winner=result.winner,
            score_a=result.score_a,
            score_b=result.score_b,
            reason=result.reason
        )

    except Exception as e:
        logger.error(f"Compare error: {e}")
        return CompareResponse(
            winner="tie",
            reason=str(e)
        )


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """
    Get application metrics in JSON format

    Returns chatbot stats, cache stats, and evaluation summary
    """
    global chatbot

    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    stats = chatbot.get_stats()

    return MetricsResponse(
        chatbot_stats={
            "session_id": stats["session_id"],
            "is_ready": stats["is_ready"],
            "query_count": stats["query_count"],
            "evaluation_enabled": stats["evaluation_enabled"]
        },
        cache_stats=stats["llm_stats"].get("cache_stats", {}),
        evaluation_summary=stats["evaluation_summary"]
    )


@app.get("/prometheus", tags=["Monitoring"])
async def prometheus_metrics_endpoint():
    """
    Get Prometheus-compatible metrics

    For scraping by Prometheus/Grafana
    """
    metrics_output = prometheus_metrics.get_metrics_output()

    if not metrics_output:
        return PlainTextResponse(
            content="# Prometheus metrics not available\n",
            media_type="text/plain"
        )

    return PlainTextResponse(
        content=metrics_output,
        media_type="text/plain; version=0.0.4"
    )


@app.get("/debug", tags=["Debug"])
async def get_debug_info():
    """
    Get debug information for troubleshooting

    Only enable in development
    """
    global chatbot

    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    return chatbot.get_debug_info()


@app.post("/reset", tags=["Admin"])
async def reset_chatbot():
    """
    Reset the chatbot state

    Clears all documents and conversation history
    """
    global chatbot

    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    chatbot.reset()

    return {"message": "Chatbot reset successfully"}


@app.post("/clear-history", tags=["Admin"])
async def clear_history():
    """
    Clear conversation history only

    Keeps documents but resets chat history
    """
    global chatbot

    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    chatbot.clear_chat_history()

    return {"message": "Chat history cleared"}


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.API_RELOAD
    )
