# RAG Chatbot Ollama

## Project Structure

The project has been refactored into a modular architecture with clear separation of concerns:

```
.
├── app.py                 # Streamlit UI application
├── chatbot.py            # Main chatbot orchestrator
├── config.py             # Configuration settings
├── document_processor.py # PDF processing module
├── llm_handler.py        # Ollama LLM integration
├── vector_store.py       # ChromaDB vector store management
├── utils.py              # Utility functions
├── pdfFiles/             # Directory for uploaded PDFs
└── vectorDB/             # Directory for vector database
```

## Module Descriptions

### 1. **config.py**
Central configuration file containing:
- Directory paths
- LLM model settings
- Document processing parameters
- UI configurations
- Message templates

### 2. **document_processor.py**
Handles all PDF-related operations:
- `DocumentProcessor` class
- PDF file saving and loading
- Text extraction from PDFs
- Document chunking with configurable parameters
- Support for single and multiple PDF processing

### 3. **vector_store.py**
Manages ChromaDB vector database:
- `VectorStoreManager` class
- Vector store creation and persistence
- Document embedding generation
- Similarity search functionality
- Retriever creation for QA chains

### 4. **llm_handler.py**
Interfaces with Ollama LLM:
- `LLMHandler` class
- LLM initialization and configuration
- Conversation memory management
- QA chain creation with retrieval
- Direct response generation

### 5. **chatbot.py**
Main orchestrator that combines all components:
- `RAGChatbot` class
- PDF processing pipeline
- Chat functionality
- State management
- Error handling

### 6. **utils.py**
Utility functions including:
- Logging setup
- Typing animation effect
- File validation
- Source formatting
- Chat history management
- Various helper functions

### 7. **app.py**
Streamlit user interface:
- Session state management
- File upload interface
- Chat interface
- Status displays
- Configuration options

## Key Features

1. **Modular Design**: Each component has a single responsibility
2. **Error Handling**: Comprehensive error handling with logging
3. **Type Hints**: Full type annotations for better code clarity
4. **Logging**: Structured logging throughout the application
5. **Configuration**: Centralized configuration management
6. **Scalability**: Easy to extend with new features

## Usage

1. Start Ollama service:
   ```bash
   docker start ollama
   ```

2. Run the application:
   ```bash
   conda activate llm-rag
   streamlit run app.py
   ```

3. Upload PDFs through the sidebar
4. Click "Process PDFs" to analyze documents
5. Start chatting with your documents

## Benefits of Modular Architecture

1. **Maintainability**: Easy to update individual components
2. **Testability**: Each module can be tested independently
3. **Reusability**: Components can be reused in other projects
4. **Clarity**: Clear separation of concerns
5. **Scalability**: Easy to add new features or swap components

## Extending the Application

To add new features:
- New LLM providers: Modify `llm_handler.py`
- Different vector stores: Update `vector_store.py`
- Additional file formats: Extend `document_processor.py`
- UI improvements: Modify `app.py`
- New utilities: Add to `utils.py`