#!/bin/bash

# Quick start script for LLM RAG Chatbot

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Starting LLM RAG Chatbot ===${NC}"

# Check if Ollama is running
echo -e "${YELLOW}Checking Ollama status...${NC}"
if docker ps | grep -q ollama; then
    echo -e "${GREEN}✓ Ollama is running${NC}"
else
    echo -e "${YELLOW}Starting Ollama...${NC}"
    docker start ollama || {
        echo -e "${RED}Failed to start Ollama. Run: ./setup_ollama.sh${NC}"
        exit 1
    }
    sleep 5
fi

# Activate conda environment
echo -e "${YELLOW}Activating conda environment...${NC}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm-rag || {
    echo -e "${RED}Failed to activate conda environment.${NC}"
    echo "Run: conda create -n llm-rag python=3.10 -y"
    echo "Then: conda activate llm-rag && pip install -r requirements.txt"
    exit 1
}

# Run the application
echo -e "${GREEN}Starting Streamlit application...${NC}"
echo -e "${YELLOW}Opening browser at http://localhost:8501${NC}"
streamlit run app.py