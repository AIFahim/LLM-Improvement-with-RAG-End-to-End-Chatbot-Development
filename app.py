"""
Streamlit application for the RAG Chatbot
"""
import streamlit as st
import logging
from chatbot import RAGChatbot
import config
import utils

# Setup logging
utils.setup_logging()
logger = logging.getLogger(__name__)


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
        logger.info("Initialized new chatbot instance")
    
    if config.SESSION_MESSAGES not in st.session_state:
        st.session_state[config.SESSION_MESSAGES] = []


def display_chat_messages():
    """Display chat messages from history"""
    for message in st.session_state[config.SESSION_MESSAGES]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main():
    """Main Streamlit application"""
    # Page configuration
    st.set_page_config(
        page_title=config.PAGE_TITLE,
        page_icon=config.PAGE_ICON,
        layout=config.PAGE_LAYOUT
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Application header
    st.title(config.PAGE_TITLE)
    st.markdown(config.WELCOME_MESSAGE)
    
    # Sidebar for PDF upload and settings
    with st.sidebar:
        st.header("📄 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            # Validate files
            valid_files = []
            for file in uploaded_files:
                if utils.validate_pdf_file(file):
                    valid_files.append(file)
                    st.success(f"✅ {file.name} ({utils.get_file_size_mb(file):.2f} MB)")
                else:
                    st.error(f"❌ {file.name} is not a valid PDF")
            
            # Process button
            if valid_files and st.button("Process PDFs", type="primary"):
                with st.spinner(config.PROCESSING_MESSAGE):
                    try:
                        success = st.session_state.chatbot.process_pdfs(valid_files)
                        if success:
                            st.success(config.SUCCESS_MESSAGE)
                            st.balloons()
                        else:
                            st.error("Failed to process PDFs. Please check the logs.")
                    except Exception as e:
                        st.error(config.ERROR_MESSAGE.format(str(e)))
                        logger.error(f"Error processing PDFs: {e}")
        
        # Additional options
        st.divider()
        st.header("⚙️ Options")
        
        # Clear chat history
        if st.button("Clear Chat History"):
            st.session_state.chatbot.clear_chat_history()
            st.session_state[config.SESSION_MESSAGES] = []
            st.success("Chat history cleared!")
        
        # Reset entire chatbot
        if st.button("Reset Chatbot", help="Clear all data and start fresh"):
            st.session_state.chatbot.reset()
            st.session_state[config.SESSION_MESSAGES] = []
            st.success("Chatbot reset successfully!")
            st.rerun()
        
        # Display chatbot status
        st.divider()
        if st.session_state.chatbot.is_ready:
            st.success("🟢 Chatbot is ready!")
        else:
            st.info("🔴 " + config.UPLOAD_PROMPT)
        
        # Model information
        with st.expander("Model Information"):
            st.write(f"**Model:** {config.LLM_MODEL}")
            st.write(f"**Base URL:** {config.LLM_BASE_URL}")
            st.write(f"**Temperature:** {config.LLM_TEMPERATURE}")
    
    # Main chat interface
    if st.session_state.chatbot.is_ready:
        # Display chat messages
        display_chat_messages()
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about your documents..."):
            # Add user message to chat history
            st.session_state[config.SESSION_MESSAGES].append({
                "role": "user",
                "content": prompt
            })
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.chatbot.chat(prompt)
                        
                        # Extract answer
                        answer = response.get('result', 'Sorry, I could not generate a response.')
                        
                        # Display with typing effect
                        utils.display_message_with_typing(answer)
                        
                        # Show sources if available
                        if response.get('source_documents'):
                            with st.expander("📚 Sources"):
                                sources = utils.format_sources(response['source_documents'])
                                st.markdown(sources)
                        
                        # Add assistant response to chat history
                        st.session_state[config.SESSION_MESSAGES].append({
                            "role": "assistant",
                            "content": answer
                        })
                        
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        logger.error(f"Chat error: {e}")
    
    else:
        # Welcome screen when no PDFs are loaded
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("👈 " + config.UPLOAD_PROMPT)
            
            # Instructions
            st.markdown("""
            ### How to use:
            1. Upload one or more PDF files using the sidebar
            2. Click "Process PDFs" to analyze the documents
            3. Start asking questions in the chat
            
            ### Features:
            - 🤖 AI-powered document Q&A
            - 📄 Support for multiple PDFs
            - 💬 Conversation memory
            - 📚 Source citations
            - 🔒 Local processing (your data stays private)
            """)


if __name__ == "__main__":
    main()