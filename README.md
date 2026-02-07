# Multimodal RAG Chatbot — Voice, Vision & Tool-Calling Agent

A multimodal AI assistant that combines **image analysis**, **voice interaction**, and **tool-calling** with RAG (Retrieval-Augmented Generation) over PDF documents. Built with LangGraph ReAct agent, Ollama, and Streamlit.

## What It Does

- **Image Analysis** — Upload images and get descriptions, visual Q&A (moondream via Ollama)
- **Voice Input/Output** — Speech-to-text (Whisper) and text-to-speech (gTTS / Orpheus)
- **Tool-Calling Agent** — LangGraph ReAct agent with 10 tools: web search, calculator, datetime, Python REPL, RAG search, image analysis, and more
- **RAG over PDFs** — Upload PDFs, chunk & embed them, then ask questions with context retrieval
- **ChatGPT-Style UI** — Streamlit app with inline image upload, voice recording, and text chat

---

## Models

| Role | Model | Size | Purpose |
|------|-------|------|---------|
| **Agent LLM** | `qwen2.5:3b` | ~2 GB | Chat + tool calling |
| **Vision** | `moondream` | ~1.7 GB | Image analysis and visual Q&A |
| **TTS** | `legraphista/Orpheus:3b-ft-q4_k_m` | ~2.4 GB | Natural speech synthesis (optional) |

---

## Setup

### 1. Environment

```bash
conda create -n rag_chatbot python=3.11 -y
conda activate rag_chatbot
pip install -r requirements.txt
```

### 2. System Dependencies

```bash
# ffmpeg (required by Whisper for audio processing)
sudo apt install ffmpeg          # Ubuntu/Debian
brew install ffmpeg               # macOS
choco install ffmpeg              # Windows
```

### 3. Start Ollama & Pull Models

```bash
# Start Ollama (Docker)
docker run -d --name ollama -p 11434:11434 -v ollama:/root/.ollama ollama/ollama

# Pull models
docker exec ollama ollama pull qwen2.5:3b                          # Agent LLM (required)
docker exec ollama ollama pull moondream                            # Vision (required)
docker exec ollama ollama pull legraphista/Orpheus:3b-ft-q4_k_m    # TTS (optional)

# Verify
docker exec ollama ollama list
```

> If running Ollama natively, omit `docker exec ollama`.

### 4. Configure

```bash
cp .env.example .env
# Defaults work for local Ollama — edit only if needed
```

### 5. Run

```bash
streamlit run multimodal_app.py
```

Open **http://localhost:8501**

---

## App Tabs

| Tab | Description |
|-----|-------------|
| **Multimodal Chat** | ChatGPT-style chat — attach images, record voice, type text |
| **Vision Studio** | Upload images and analyze with custom prompts |
| **Voice Lab** | Test STT (Whisper) and TTS (gTTS / pyttsx3 / Orpheus) |
| **Agent Info** | View tools, models, config, memory status |

---

## Tools (LangGraph ReAct Agent)

| Tool | Description |
|------|-------------|
| `calculator` | Math expressions |
| `web_search` | DuckDuckGo web search |
| `python_repl` | Execute Python code |
| `datetime` | Current date/time |
| `rag_search` | Search uploaded PDFs |
| `image_analysis` | Analyze images with vision model |
| `image_question` | Visual Q&A on images |
| `voice_transcription` | Transcribe audio to text |
| `text_to_speech` | Convert text to speech |
| `image_to_rag` | Analyze image + store in vector DB |

---

## Architecture

```
User Input (Text / Image / Audio)
    │
    ▼
┌──────────────────────────┐
│   Multimodal App (UI)    │  Streamlit (4 tabs)
│   multimodal_app.py      │  ChatGPT-style chat
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   MultimodalAgent        │  Orchestrates all modalities
│   (LangGraph ReAct)      │  qwen2.5:3b with tool calling
└────────────┬─────────────┘
             │
    ┌────────┼────────┬───────────┐
    ▼        ▼        ▼           ▼
┌────────┐┌───────┐┌───────┐┌────────┐
│ Vision ││ Voice ││ Tools ││ Memory │
│moondream││Whisper││ 5 + 5 ││ Buffer │
│        ││gTTS   ││       ││        │
│        ││Orpheus││       ││        │
└───┬────┘└───┬───┘└───┬───┘└────────┘
    ▼         ▼        ▼
  Ollama   ffmpeg   DuckDuckGo /
  Vision   + SNAC   ChromaDB / etc.
```

---

## Project Structure

```
├── multimodal_app.py      # Streamlit UI (4 tabs)
├── multimodal_agent.py    # Central orchestrator (LangGraph ReAct)
├── multimodal_tools.py    # 5 multimodal LangChain tools
├── vision_handler.py      # Image analysis via Ollama (moondream)
├── voice_handler.py       # Whisper STT + gTTS/Orpheus TTS
├── agent.py               # LangGraph ReAct agent base
├── tools.py               # 5 standard LangChain tools
├── memory_manager.py      # Conversation memory (buffer/window/summary)
├── config.py              # All configuration + secure API key handling
├── chatbot.py             # RAG chatbot orchestrator
├── document_processor.py  # PDF chunking
├── vector_store.py        # ChromaDB vector store
├── llm_handler.py         # LLM integration with retries & caching
├── error_handler.py       # Retry logic & circuit breaker
├── app.py                 # Evaluation dashboard (legacy)
├── api.py                 # FastAPI REST endpoints (legacy)
├── .env.example           # Environment variable template
└── requirements.txt       # Python dependencies
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `qwen2.5:3b` | Agent LLM (needs tool calling support) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `VISION_MODEL` | `moondream` | Vision model |
| `VISION_ENABLED` | `true` | Enable image analysis |
| `VOICE_ENABLED` | `true` | Enable voice features |
| `STT_ENGINE` | `whisper` | STT backend (`whisper` / `google`) |
| `WHISPER_MODEL_SIZE` | `base` | Whisper model (`tiny`, `base`, `small`, `medium`, `large`) |
| `TTS_ENGINE` | `gtts` | TTS backend (`gtts` / `pyttsx3` / `orpheus`) |
| `ORPHEUS_MODEL` | `legraphista/Orpheus` | Orpheus TTS model |
| `ORPHEUS_VOICE` | `tara` | Voice: `tara` `leah` `jess` `leo` `dan` `mia` `zac` `zoe` |
| `MULTIMODAL_MODE` | `full` | Agent mode (`full` / `text` / `vision` / `voice`) |
| `MULTIMODAL_AUTO_TTS` | `false` | Auto-generate speech for responses |
| `LLM_PROVIDER` | `ollama` | `ollama` or `azure` |

---

## Troubleshooting

**Ollama not running**
```bash
docker start ollama
```

**Vision model not found**
```bash
docker exec ollama ollama pull moondream
```

**Agent not using tools / outputting raw JSON**
Use a 3B+ model with tool calling support:
```bash
docker exec ollama ollama pull qwen2.5:3b
# Set OLLAMA_MODEL=qwen2.5:3b in .env
```

**Whisper import error**
```bash
pip install openai-whisper
```

**ffmpeg not found**
```bash
sudo apt install ffmpeg
```

**TTS not working**
```bash
pip install gTTS                  # Google TTS (needs internet)
pip install pyttsx3               # Offline fallback
# Or use Orpheus via Ollama:
docker exec ollama ollama pull legraphista/Orpheus:3b-ft-q4_k_m
```

**OOM with Orpheus** — Use `TTS_ENGINE=gtts` in `.env` instead.

**Import errors** — `pip install -r requirements.txt`

---

## Other Apps (from previous classes)

| App | Command | Description |
|-----|---------|-------------|
| Evaluation Dashboard | `streamlit run app.py` | RAGAS eval, LLM Judge, A/B testing |
| FastAPI Server | `uvicorn api:app --reload` | REST API at `/chat`, `/upload`, `/evaluate` |
| CLI Launcher | `python run_app.py` | Command-line interface |

---

## License

MIT License
