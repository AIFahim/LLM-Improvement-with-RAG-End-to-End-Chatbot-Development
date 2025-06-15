"""
LLM handler module for managing Ollama interactions
"""
from typing import Optional, Dict, Any
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import config
import logging

logger = logging.getLogger(__name__)


class LLMHandler:
    """Handles interactions with Ollama LLM"""
    
    def __init__(self, model_name: str = config.LLM_MODEL,
                 base_url: str = config.LLM_BASE_URL,
                 temperature: float = config.LLM_TEMPERATURE):
        """
        Initialize the LLM handler
        
        Args:
            model_name: Name of the Ollama model
            base_url: Base URL for Ollama API
            temperature: Temperature for generation
        """
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self._llm = None
        self._memory = None
        self._qa_chain = None
    
    def get_llm(self) -> OllamaLLM:
        """
        Get or create the LLM instance
        
        Returns:
            Ollama LLM instance
        """
        if self._llm is None:
            self._llm = OllamaLLM(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature
            )
            logger.info(f"Initialized Ollama LLM with model {self.model_name}")
        return self._llm
    
    def get_memory(self) -> ConversationBufferMemory:
        """
        Get or create conversation memory
        
        Returns:
            ConversationBufferMemory instance
        """
        if self._memory is None:
            self._memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="result"
            )
            logger.info("Initialized conversation memory")
        return self._memory
    
    def create_qa_chain(self, retriever) -> RetrievalQA:
        """
        Create a QA chain with retriever
        
        Args:
            retriever: Document retriever instance
            
        Returns:
            RetrievalQA chain instance
        """
        # Define the prompt template
        prompt_template = """You are a helpful AI assistant. Use the following context to answer the user's question.
If you cannot find the answer in the context, say so politely.

Context: {context}

Question: {question}

Answer: """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self._qa_chain = RetrievalQA.from_chain_type(
            llm=self.get_llm(),
            chain_type="stuff",
            retriever=retriever,
            memory=self.get_memory(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        logger.info("Created QA chain with retriever")
        return self._qa_chain
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the QA chain
        
        Args:
            question: User question
            
        Returns:
            Dictionary with result and source documents
        """
        if self._qa_chain is None:
            raise ValueError("QA chain not initialized. Call create_qa_chain first.")
        
        try:
            response = self._qa_chain({"query": question})
            logger.info(f"Generated response for question: {question[:50]}...")
            return response
        except Exception as e:
            logger.error(f"Error during query: {e}")
            raise
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate response without retrieval (direct LLM call)
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated response
        """
        llm = self.get_llm()
        try:
            response = llm.invoke(prompt)
            logger.info(f"Generated direct response for prompt: {prompt[:50]}...")
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def clear_memory(self):
        """Clear conversation memory"""
        if self._memory:
            self._memory.clear()
            logger.info("Cleared conversation memory")
    
    def get_conversation_history(self) -> list:
        """
        Get conversation history
        
        Returns:
            List of conversation messages
        """
        if self._memory is None:
            return []
        
        return self._memory.chat_memory.messages