#!/bin/bash

# Ollama Setup Script for LLM RAG Chatbot
# This script sets up Ollama using Docker and downloads the required model

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Ollama Setup Script for LLM RAG Chatbot ===${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Docker is installed
echo -e "${YELLOW}Checking for Docker...${NC}"
if ! command_exists docker; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}Docker is not running. Please start Docker.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker is installed and running${NC}"

# Check if Ollama container exists
echo -e "${YELLOW}Checking for existing Ollama container...${NC}"
if docker ps -a | grep -q ollama; then
    echo -e "${YELLOW}Ollama container already exists. Removing old container...${NC}"
    docker stop ollama >/dev/null 2>&1 || true
    docker rm ollama >/dev/null 2>&1 || true
fi

# Pull Ollama Docker image
echo -e "${YELLOW}Pulling Ollama Docker image...${NC}"
docker pull ollama/ollama:latest

# Check for NVIDIA GPU
echo -e "${YELLOW}Checking for NVIDIA GPU...${NC}"
GPU_FLAGS=""
if command_exists nvidia-smi && nvidia-smi >/dev/null 2>&1; then
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    # Check if nvidia-docker is available
    if docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi >/dev/null 2>&1; then
        GPU_FLAGS="--gpus all"
        echo -e "${GREEN}✓ GPU support enabled${NC}"
    else
        echo -e "${YELLOW}⚠ GPU detected but Docker GPU support not available${NC}"
        echo "To enable GPU support, install nvidia-docker2"
    fi
else
    echo -e "${YELLOW}No NVIDIA GPU detected or nvidia-smi not available${NC}"
fi

# Run Ollama container
echo -e "${YELLOW}Starting Ollama container...${NC}"
docker run -d \
    $GPU_FLAGS \
    -v ollama:/root/.ollama \
    -p 11434:11434 \
    --name ollama \
    --restart unless-stopped \
    ollama/ollama

# Wait for Ollama to start
echo -e "${YELLOW}Waiting for Ollama to start...${NC}"
sleep 5

# Check if Ollama is running
max_attempts=30
attempt=0
while ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ $attempt -ge $max_attempts ]; then
        echo -e "${RED}Ollama failed to start. Please check Docker logs:${NC}"
        echo "docker logs ollama"
        exit 1
    fi
    echo -n "."
    sleep 2
done
echo ""
echo -e "${GREEN}✓ Ollama is running${NC}"

# Download the model
MODEL="llama3.2:1b"
echo -e "${YELLOW}Downloading model: $MODEL${NC}"
echo "This may take several minutes depending on your internet connection..."

docker exec ollama ollama pull $MODEL

# Verify model is downloaded
echo -e "${YELLOW}Verifying model installation...${NC}"
if docker exec ollama ollama list | grep -q "$MODEL"; then
    echo -e "${GREEN}✓ Model $MODEL successfully installed${NC}"
else
    echo -e "${RED}Failed to install model $MODEL${NC}"
    exit 1
fi

# Test the model
echo -e "${YELLOW}Testing the model...${NC}"
TEST_RESPONSE=$(curl -s http://localhost:11434/api/generate -d '{
  "model": "llama3.2:1b",
  "prompt": "Hello, respond with: I am working!",
  "stream": false
}' | grep -o '"response":"[^"]*"' | sed 's/"response":"//;s/"$//')

if [ ! -z "$TEST_RESPONSE" ]; then
    echo -e "${GREEN}✓ Model test successful${NC}"
    echo "Response: $TEST_RESPONSE"
else
    echo -e "${RED}Model test failed${NC}"
fi

echo ""
echo -e "${GREEN}=== Ollama Setup Complete ===${NC}"
echo ""
echo "Ollama is now running with the following configuration:"
echo "- Container name: ollama"
echo "- API endpoint: http://localhost:11434"
echo "- Model: $MODEL"
if [ ! -z "$GPU_FLAGS" ]; then
    echo "- GPU support: Enabled"
fi
echo ""
echo "To check Ollama status: docker ps | grep ollama"
echo "To view Ollama logs: docker logs ollama"
echo "To stop Ollama: docker stop ollama"
echo "To start Ollama: docker start ollama"
echo ""

# Optional: Download additional models
echo -e "${YELLOW}Would you like to download additional models? (y/n)${NC}"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Available models:"
    echo "1. llama3.2:3b (3B parameters, ~2GB)"
    echo "2. mistral:7b (7B parameters, ~4GB)"
    echo "3. phi3:3.8b (3.8B parameters, ~2.3GB)"
    echo "4. tinyllama:1.1b (1.1B parameters, ~637MB)"
    echo ""
    echo "Enter model numbers separated by spaces (e.g., '1 3'):"
    read -r model_choices
    
    for choice in $model_choices; do
        case $choice in
            1) docker exec ollama ollama pull llama3.2:3b ;;
            2) docker exec ollama ollama pull mistral:7b ;;
            3) docker exec ollama ollama pull phi3:3.8b ;;
            4) docker exec ollama ollama pull tinyllama:1.1b ;;
            *) echo "Invalid choice: $choice" ;;
        esac
    done
fi

echo -e "${GREEN}Setup complete! You can now run the chatbot application.${NC}"