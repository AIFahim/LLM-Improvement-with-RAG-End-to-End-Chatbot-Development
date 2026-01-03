# RAG Chatbot with Ollama & Azure OpenAI

A modular RAG (Retrieval-Augmented Generation) chatbot that supports both **Ollama** (local) and **Azure OpenAI** (cloud) as LLM providers.

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda create -n rag_chatbot python=3.11 -y
conda activate rag_chatbot

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup LLM Provider

#### Option A: Ollama (Local - Free)

```bash
# Start Ollama with Docker
docker run -d --name ollama -p 11434:11434 -v ollama:/root/.ollama ollama/ollama

# Pull a model
docker exec ollama ollama pull qwen2.5:1.5b
```

#### Option B: Azure OpenAI (Cloud)

Create a `.env` file:
```bash
cp .env.example .env
```

Edit `.env` with your Azure credentials:
```env
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_LLM_DEPLOYMENT_NAME=gpt-4
```

### 3. Run the Application

```bash
# Run with Ollama (default)
python run_app.py

# Run with Azure OpenAI
python run_app.py --provider azure

# Run with specific options
python run_app.py --provider ollama --model llama3.2:1b --port 8501
```

Open http://localhost:8501 in your browser.

---

## Command Line Options

```bash
python run_app.py --help
```

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--provider` | `-p` | `ollama` | LLM provider: `ollama` or `azure` |
| `--model` | `-m` | `qwen2.5:1.5b` | Ollama model name |
| `--ollama-url` | | `http://localhost:11434` | Ollama base URL |
| `--api-key` | | from .env | Azure OpenAI API key |
| `--endpoint` | | from .env | Azure OpenAI endpoint |
| `--deployment` | `-d` | `gpt-4` | Azure deployment name |
| `--temperature` | `-t` | `0.7` | LLM temperature |
| `--port` | | `8501` | Streamlit port |
| `--check` | | | Validate config only |

### Examples

```bash
# Check configuration without starting
python run_app.py --check

# Run with Ollama and specific model
python run_app.py --provider ollama --model llama3.2:3b

# Run with Azure OpenAI
python run_app.py --provider azure

# Run with Azure and custom deployment
python run_app.py --provider azure --deployment gpt-4-turbo

# Run on different port
python run_app.py --port 8502
```

---

## Project Structure

```
.
├── app.py                 # Streamlit UI application
├── chatbot.py             # Main chatbot orchestrator
├── config.py              # Configuration settings
├── document_processor.py  # PDF processing module
├── llm_handler.py         # LLM integration (Ollama + Azure)
├── vector_store.py        # ChromaDB vector store management
├── utils.py               # Utility functions
├── run_app.py             # CLI launcher with argparse
├── run.sh                 # Shell script launcher
├── .env.example           # Environment template
├── pdfFiles/              # Directory for uploaded PDFs
└── vectorDB/              # Directory for vector database
```

---

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `LLM_PROVIDER` | `ollama` or `azure` | No (default: ollama) |
| `OLLAMA_MODEL` | Ollama model name | No |
| `OLLAMA_BASE_URL` | Ollama API URL | No |
| `AZURE_OPENAI_API_KEY` | Azure API key | For Azure |
| `AZURE_OPENAI_ENDPOINT` | Azure endpoint URL | For Azure |
| `AZURE_LLM_DEPLOYMENT_NAME` | Azure deployment name | For Azure |
| `AZURE_OPENAI_API_VERSION` | Azure API version | No |
| `LLM_TEMPERATURE` | Generation temperature | No |

### Supported Ollama Models

```bash
# Small models (fast, low memory)
docker exec ollama ollama pull qwen2.5:1.5b
docker exec ollama ollama pull llama3.2:1b
docker exec ollama ollama pull phi3:mini

# Medium models (balanced)
docker exec ollama ollama pull llama3.2:3b
docker exec ollama ollama pull mistral

# List available models
docker exec ollama ollama list
```

---

## Module Descriptions

### config.py
Central configuration with environment variable support for both Ollama and Azure.

### llm_handler.py
Unified LLM handler supporting:
- Ollama (local models)
- Azure OpenAI (cloud models)
- Provider switching at runtime

### document_processor.py
PDF processing with:
- Text extraction
- Document chunking
- Multiple file support

### vector_store.py
ChromaDB vector database:
- Document embedding (via Ollama)
- Similarity search
- Persistent storage

### chatbot.py
Main orchestrator combining all components.

### app.py
Streamlit UI with:
- File upload
- Chat interface
- Provider info display

---

## Usage

1. Start the application with your preferred provider
2. Upload PDF files through the sidebar
3. Click "Process PDFs" to analyze documents
4. Start asking questions about your documents

---

## Troubleshooting

### Ollama not running
```bash
docker start ollama
# or
docker run -d --name ollama -p 11434:11434 -v ollama:/root/.ollama ollama/ollama
```

### Model not found
```bash
docker exec ollama ollama pull qwen2.5:1.5b
```

### Azure authentication error
- Verify API key in `.env`
- Check endpoint URL format
- Confirm deployment name matches Azure portal

### Port already in use
```bash
python run_app.py --port 8502
```

---

## License

MIT License
