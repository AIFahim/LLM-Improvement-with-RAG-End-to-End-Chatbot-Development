"""
Example usage of the modular RAG chatbot components.
"""

from pathlib import Path
from chatbot import RAGChatbot
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from llm_handler import LLMHandler


def example_basic_usage():
    """Example of basic chatbot usage."""
    print("=== Basic Chatbot Usage Example ===\n")
    
    # Initialize the chatbot
    chatbot = RAGChatbot()
    
    # Process a PDF (replace with your PDF path)
    pdf_path = Path("pdfFiles/example.pdf")
    if pdf_path.exists():
        success = chatbot.process_pdf(pdf_path)
        if success:
            print(f"Successfully processed: {pdf_path}")
            
            # Ask questions
            questions = [
                "What is the main topic of this document?",
                "Can you summarize the key points?",
                "What are the conclusions?"
            ]
            
            for question in questions:
                print(f"\nQ: {question}")
                response = chatbot.ask_question(question)
                if response:
                    print(f"A: {response['result']}")
    else:
        print(f"PDF file not found: {pdf_path}")


def example_component_usage():
    """Example of using individual components."""
    print("\n=== Component Usage Example ===\n")
    
    # Document Processor Example
    print("1. Document Processor:")
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=100)
    pdf_path = Path("pdfFiles/example.pdf")
    
    if pdf_path.exists():
        chunks = processor.process_pdf(pdf_path)
        if chunks:
            print(f"   - Processed {len(chunks)} chunks")
            print(f"   - First chunk preview: {chunks[0].page_content[:100]}...")
    
    # Vector Store Example
    print("\n2. Vector Store Manager:")
    vector_store = VectorStoreManager()
    
    if chunks:
        success = vector_store.add_documents(chunks)
        if success:
            print("   - Documents added to vector store")
            
            # Search example
            results = vector_store.similarity_search("main topic", k=3)
            print(f"   - Found {len(results)} similar documents")
    
    # LLM Handler Example
    print("\n3. LLM Handler:")
    llm_handler = LLMHandler()
    
    # Test connection
    if llm_handler.test_connection():
        print("   - LLM connection successful")
        
        # Generate response
        response = llm_handler.generate_response("What is artificial intelligence?")
        if response:
            print(f"   - Generated response: {response[:100]}...")


def example_custom_configuration():
    """Example of custom configuration."""
    print("\n=== Custom Configuration Example ===\n")
    
    # Custom prompt template
    custom_prompt = """You are an expert document analyst. 
    Based on the context provided, give detailed and technical answers.
    
    Context: {context}
    History: {history}
    
    Question: {question}
    Expert Analysis:"""
    
    # Initialize chatbot
    chatbot = RAGChatbot()
    
    # Update prompt template
    chatbot.update_prompt_template(custom_prompt)
    print("Custom prompt template applied")
    
    # Get system info
    info = chatbot.get_system_info()
    print(f"System Info: {info}")


if __name__ == "__main__":
    # Run examples
    try:
        # Uncomment the examples you want to run
        
        # example_basic_usage()
        # example_component_usage()
        # example_custom_configuration()
        
        print("To run the examples, uncomment the desired function calls above.")
        print("Make sure you have:")
        print("1. Ollama running locally")
        print("2. A PDF file in the pdfFiles directory")
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")