# RAG Chatbot with Evaluation, Optimization & Deployment

A production-ready RAG (Retrieval-Augmented Generation) chatbot with **SOTA evaluation**, **monitoring**, and **deployment** capabilities. Supports both **Ollama** (local) and **Azure OpenAI** (cloud) as LLM providers.

## Features

- **RAG Evaluation**: RAGAS metrics (faithfulness, relevancy, context precision)
- **LLM-as-a-Judge**: G-Eval with Chain-of-Thought scoring + A/B Testing
- **Fast Evaluation**: Instant heuristic-based scoring (no LLM required)
- **Observability**: Langfuse tracing & Prometheus metrics
- **Performance**: Response caching, retry logic, circuit breaker
- **Deployment**: Streamlit UI + FastAPI REST endpoints
- **Multi-Provider**: Ollama (local) and Azure OpenAI (cloud)

---

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda create -n rag_chatbot python=3.11 -y
conda activate rag_chatbot

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings (optional - defaults work for Ollama)
```

### 3. Setup LLM Provider

#### Option A: Ollama (Local - Free)

```bash
# Start Ollama with Docker
docker run -d --name ollama -p 11434:11434 -v ollama:/root/.ollama ollama/ollama

# Pull a model
docker exec ollama ollama pull llama3.2:1b
```

#### Option B: Azure OpenAI (Cloud)

Edit `.env` with your Azure credentials:
```env
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4
```

### 4. Run the Application

```bash
# Run Streamlit app
streamlit run app.py

# OR run FastAPI server
uvicorn api:app --reload --port 8000

# OR use the CLI launcher
python run_app.py
```

- Streamlit UI: http://localhost:8501
- FastAPI docs: http://localhost:8000/docs

---

## Project Structure

```
.
├── app.py                 # Streamlit UI with evaluation dashboard
├── api.py                 # FastAPI REST endpoints
├── chatbot.py             # Main chatbot orchestrator
├── config.py              # Configuration with secure API key handling
├── document_processor.py  # PDF processing module
├── llm_handler.py         # LLM integration with retries & caching
├── vector_store.py        # ChromaDB vector store management
├── evaluator.py           # RAGAS-based evaluation metrics
├── llm_judge.py           # LLM-as-a-Judge with G-Eval
├── error_handler.py       # Retry logic & circuit breaker
├── monitoring.py          # Langfuse & Prometheus metrics
├── performance.py         # Response caching & optimization
├── utils.py               # Utility functions
├── run_app.py             # CLI launcher
├── .env.example           # Environment template
├── pdfFiles/              # Directory for uploaded PDFs
└── vectorDB/              # Directory for vector database
```

---

## Streamlit UI Features

### 4 Main Tabs

| Tab | Description |
|-----|-------------|
| **Chat** | Main chat interface with PDF upload |
| **Evaluation Dashboard** | RAGAS/Simple evaluation scores, history, charts |
| **LLM Judge** | Judge history + A/B Testing comparison tool |
| **Metrics** | Cache stats, response times, debug info |

### Sidebar Settings

| Setting | Description |
|---------|-------------|
| **Enable Response Evaluation** | Toggle evaluation on/off |
| **Use Fast Evaluation** | Switch between instant heuristics vs RAGAS (slower) |
| **Enable LLM Judge** | Use LLM to judge response quality |
| **Judge Criteria** | Select: Correctness, Relevance, Coherence, Helpfulness, Completeness |
| **Debug Mode** | Show response times and technical details |

---

## Evaluation Systems

### 1. Fast Evaluation (Instant)
Heuristic-based scoring without LLM calls:
- **Faithfulness**: Word overlap between response and context
- **Relevancy**: Query-response word matching
- **Overall Score**: Weighted average

### 2. RAGAS Evaluation (Accurate)
LLM-based metrics using RAGAS library:
- **Faithfulness**: Is response factually consistent with context? (0-1)
- **Answer Relevancy**: Does response address the query? (0-1)
- **Context Precision**: Are retrieved contexts relevant? (0-1)
- **Context Recall**: Does context contain required info? (0-1)

### 3. LLM-as-a-Judge (G-Eval)
Chain-of-Thought prompting for reliable scoring:
- Multiple criteria: correctness, relevance, coherence, helpfulness, completeness
- Detailed reasoning for each score
- Pairwise comparison for A/B testing

### A/B Testing (Response Comparison)
Compare two responses to the same question:
1. Go to **LLM Judge** tab
2. Enter question and two responses
3. Click "Compare Responses"
4. See winner, scores, and reasoning

---

## API Endpoints (FastAPI)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Chat with documents |
| `/upload` | POST | Upload PDF files |
| `/evaluate` | POST | Evaluate a response with RAGAS |
| `/judge` | POST | Judge response with LLM-as-a-Judge |
| `/compare` | POST | Compare two responses (A/B testing) |
| `/health` | GET | Health check |
| `/metrics` | GET | Application metrics (JSON) |
| `/prometheus` | GET | Prometheus metrics (for Grafana) |

### Example API Usage

```bash
# Upload a PDF
curl -X POST http://localhost:8000/upload \
  -F "files=@document.pdf"

# Chat with documents
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?", "evaluate": true}'

# Judge a response
curl -X POST http://localhost:8000/judge \
  -H "Content-Type: application/json" \
  -d '{"query": "What is AI?", "response": "AI is artificial intelligence.", "criteria": "correctness"}'

# Compare responses (A/B testing)
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"query": "What is AI?", "response_a": "AI is tech.", "response_b": "AI is artificial intelligence that mimics human cognition."}'

# Health check
curl http://localhost:8000/health
```

---

## Monitoring & Observability

### Langfuse Integration

Enable tracing in `.env`:
```env
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com
```

Langfuse tracks:
- Every chat interaction (query, response, contexts)
- Evaluation scores (faithfulness, relevancy)
- Error events
- Performance timing

View traces at: https://cloud.langfuse.com

### Prometheus Metrics

Scrape metrics at `/prometheus` endpoint for Grafana dashboards:
- `chatbot_requests_total` - Total requests by endpoint/status
- `chatbot_response_latency_seconds` - Response latency histogram
- `chatbot_errors_total` - Error count by type
- `chatbot_cache_hits_total` - Cache hit/miss counts
- `chatbot_evaluation_scores` - Evaluation score distribution

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | LLM provider: `ollama` or `azure` |
| `OLLAMA_MODEL` | `llama3.2:1b` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `AZURE_OPENAI_API_KEY` | - | Azure API key (required for Azure) |
| `AZURE_OPENAI_ENDPOINT` | - | Azure endpoint URL (required for Azure) |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4` | Azure deployment name |
| `EVALUATION_ENABLED` | `false` | Enable response evaluation |
| `CACHE_ENABLED` | `true` | Enable response caching |
| `CACHE_TTL_SECONDS` | `3600` | Cache time-to-live |
| `LANGFUSE_ENABLED` | `false` | Enable Langfuse tracing |
| `API_PORT` | `8000` | FastAPI port |

---

## Command Line Options

```bash
python run_app.py --help
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--provider` | `ollama` | LLM provider: `ollama` or `azure` |
| `--model` | `llama3.2:1b` | Ollama model name |
| `--port` | `8501` | Streamlit port |
| `--check` | - | Validate config only |

---

## Troubleshooting

### Ollama not running
```bash
docker start ollama
```

### Model not found
```bash
docker exec ollama ollama pull llama3.2:1b
```

### Import errors
```bash
pip install -r requirements.txt
```

### Azure authentication error
- Verify API key in `.env`
- Check endpoint URL format
- Confirm deployment name matches Azure portal

### Evaluation not working
- Ensure `EVALUATION_ENABLED=true` in `.env`
- Or enable via Streamlit sidebar toggle
- Check Ollama is running (needed for RAGAS evaluation)

### LLM Judge slow
- LLM Judge uses Ollama for evaluation which can be slow
- Consider using Fast Evaluation for quicker feedback
- Larger models (llama3.2:3b+) give better judgments but are slower

### A/B Testing not working
- Ensure Ollama is running
- Check the LLM Judge tab in Streamlit
- Works even without PDFs uploaded (uses empty context)

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│   Streamlit UI  │ ◄─── PDF Upload
│   or FastAPI    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Chatbot      │ ◄─── Orchestrates all components
│  (chatbot.py)   │
└────────┬────────┘
         │
    ┌────┴────┬─────────────┐
    ▼         ▼             ▼
┌───────┐ ┌───────┐   ┌──────────┐
│Vector │ │  LLM  │   │Evaluator │
│ Store │ │Handler│   │  Suite   │
└───┬───┘ └───┬───┘   └────┬─────┘
    │         │            │
    ▼         ▼            ▼
ChromaDB   Ollama/    RAGAS + G-Eval
           Azure      + Simple Eval
```

---

## License

MIT License
