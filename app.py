"""
Streamlit application for the RAG Chatbot
Enhanced with evaluation UI, metrics dashboard, and debug mode
"""
import streamlit as st
import logging
import time
from chatbot import RAGChatbot
from evaluator import EvaluationResult
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

    if config.SESSION_EVALUATION_ENABLED not in st.session_state:
        st.session_state[config.SESSION_EVALUATION_ENABLED] = config.EVALUATION_ENABLED

    if config.SESSION_DEBUG_MODE not in st.session_state:
        st.session_state[config.SESSION_DEBUG_MODE] = False

    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = []


def display_chat_messages():
    """Display chat messages from history"""
    for message in st.session_state[config.SESSION_MESSAGES]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display evaluation if available
            if message.get("evaluation") and st.session_state[config.SESSION_EVALUATION_ENABLED]:
                display_inline_evaluation(message["evaluation"])


def display_inline_evaluation(evaluation: dict):
    """Display evaluation scores inline with the message"""
    if not evaluation:
        return

    with st.expander("Evaluation Scores", expanded=False):
        cols = st.columns(4)

        scores = [
            ("Faithfulness", evaluation.get("faithfulness")),
            ("Relevancy", evaluation.get("answer_relevancy")),
            ("Context Precision", evaluation.get("context_precision")),
            ("Overall", evaluation.get("overall_score"))
        ]

        for col, (label, score) in zip(cols, scores):
            if score is not None:
                # Color based on score
                if score >= 0.7:
                    color = "green"
                elif score >= 0.4:
                    color = "orange"
                else:
                    color = "red"

                col.metric(label, f"{score:.2f}")


def display_evaluation_dashboard():
    """Display the evaluation dashboard tab"""
    st.header("Evaluation Dashboard")

    chatbot = st.session_state.chatbot
    eval_history = chatbot.get_evaluation_history()
    eval_summary = chatbot.get_evaluation_summary()

    # Summary metrics
    st.subheader("Summary")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Evaluations", eval_summary.get("total_evaluations", 0))

    avg_overall = eval_summary.get("avg_overall_score")
    col2.metric(
        "Avg Overall Score",
        f"{avg_overall:.2f}" if avg_overall else "N/A"
    )

    avg_faith = eval_summary.get("avg_faithfulness")
    col3.metric(
        "Avg Faithfulness",
        f"{avg_faith:.2f}" if avg_faith else "N/A"
    )

    avg_rel = eval_summary.get("avg_relevancy")
    col4.metric(
        "Avg Relevancy",
        f"{avg_rel:.2f}" if avg_rel else "N/A"
    )

    # Score history chart
    if eval_history:
        st.subheader("Score History")

        # Prepare data for chart
        chart_data = {
            "Query #": list(range(1, len(eval_history) + 1)),
            "Overall Score": [e.get("overall_score", 0) for e in eval_history],
            "Faithfulness": [e.get("faithfulness", 0) for e in eval_history],
            "Relevancy": [e.get("relevancy", 0) for e in eval_history]
        }

        import pandas as pd
        df = pd.DataFrame(chart_data)
        df = df.set_index("Query #")

        st.line_chart(df)

        # Recent evaluations table
        st.subheader("Recent Evaluations")
        recent = eval_history[-10:][::-1]  # Last 10, reversed

        for i, eval_item in enumerate(recent):
            with st.expander(f"Query: {eval_item.get('query', 'N/A')[:50]}..."):
                st.write(f"**Overall Score:** {eval_item.get('overall_score', 'N/A')}")
                st.write(f"**Faithfulness:** {eval_item.get('faithfulness', 'N/A')}")
                st.write(f"**Relevancy:** {eval_item.get('relevancy', 'N/A')}")
    else:
        st.info("No evaluations yet. Enable evaluation and ask some questions!")


def display_metrics_dashboard():
    """Display performance metrics dashboard"""
    st.header("Performance Metrics")

    chatbot = st.session_state.chatbot
    stats = chatbot.get_stats()

    # Session info
    st.subheader("Session Information")
    col1, col2, col3 = st.columns(3)
    col1.metric("Session ID", stats.get("session_id", "N/A"))
    col2.metric("Total Queries", stats.get("query_count", 0))
    col3.metric("Status", "Ready" if stats.get("is_ready") else "Not Ready")

    # Cache stats
    st.subheader("Cache Statistics")
    cache_stats = stats.get("llm_stats", {}).get("cache_stats", {})

    if cache_stats:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Cache Size", cache_stats.get("cache_size", 0))
        col2.metric("Cache Hits", cache_stats.get("cache_hits", 0))
        col3.metric("Cache Misses", cache_stats.get("cache_misses", 0))
        col4.metric(
            "Hit Rate",
            f"{cache_stats.get('hit_rate', 0):.1%}"
        )

    # Debug profile
    debug_profile = stats.get("debug_profile")
    if debug_profile:
        st.subheader("Average Response Times")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total", f"{debug_profile.total_time_ms:.1f}ms")
        col2.metric("Retrieval", f"{debug_profile.retrieval_time_ms:.1f}ms")
        col3.metric("LLM", f"{debug_profile.llm_time_ms:.1f}ms")


def display_debug_panel():
    """Display debug information panel"""
    st.header("Debug Information")

    chatbot = st.session_state.chatbot
    debug_info = chatbot.get_debug_info()

    st.json(debug_info)

    # Configuration info
    st.subheader("Configuration")
    config_info = {
        "LLM Provider": config.LLM_PROVIDER,
        "Ollama Model": config.OLLAMA_MODEL,
        "Ollama URL": config.OLLAMA_BASE_URL,
        "Langfuse Enabled": config.LANGFUSE_ENABLED,
        "Cache Enabled": config.CACHE_ENABLED,
        "Evaluation Enabled": config.EVALUATION_ENABLED
    }
    st.json(config_info)


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

    # Create tabs for main app and dashboards
    tab_chat, tab_eval, tab_metrics = st.tabs([
        "Chat",
        "Evaluation Dashboard",
        "Metrics"
    ])

    # Chat Tab
    with tab_chat:
        # Application header
        st.title(config.PAGE_TITLE)
        st.markdown(config.WELCOME_MESSAGE)

        # Sidebar for PDF upload and settings
        with st.sidebar:
            st.header("Document Upload")

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
                        st.success(f"{file.name} ({utils.get_file_size_mb(file):.2f} MB)")
                    else:
                        st.error(f"{file.name} is not a valid PDF")

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

            # Settings section
            st.divider()
            st.header("Settings")

            # Evaluation toggle
            eval_enabled = st.checkbox(
                "Enable Response Evaluation",
                value=st.session_state[config.SESSION_EVALUATION_ENABLED],
                help="Evaluate responses using RAGAS metrics"
            )
            st.session_state[config.SESSION_EVALUATION_ENABLED] = eval_enabled
            st.session_state.chatbot.evaluation_enabled = eval_enabled

            # Debug mode toggle
            debug_mode = st.checkbox(
                "Debug Mode",
                value=st.session_state[config.SESSION_DEBUG_MODE],
                help="Show debug information"
            )
            st.session_state[config.SESSION_DEBUG_MODE] = debug_mode

            # Chat controls
            st.divider()
            st.header("Chat Controls")

            # Clear chat history
            if st.button("Clear Chat History"):
                st.session_state.chatbot.clear_chat_history()
                st.session_state[config.SESSION_MESSAGES] = []
                st.success("Chat history cleared!")

            # Reset entire chatbot
            if st.button("Reset Chatbot", help="Clear all data and start fresh"):
                st.session_state.chatbot.reset()
                st.session_state[config.SESSION_MESSAGES] = []
                st.session_state.evaluation_results = []
                st.success("Chatbot reset successfully!")
                st.rerun()

            # Display chatbot status
            st.divider()
            if st.session_state.chatbot.is_ready:
                st.success("Chatbot is ready!")
            else:
                st.info(config.UPLOAD_PROMPT)

            # Model information
            with st.expander("Model Information"):
                model_info = st.session_state.chatbot.llm_handler.model_info
                st.write(f"**Provider:** {model_info['provider']}")
                st.write(f"**Model:** {model_info['model']}")
                st.write(f"**Endpoint:** {model_info['endpoint']}")
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
                            start_time = time.time()
                            response = st.session_state.chatbot.chat(
                                prompt,
                                evaluate=st.session_state[config.SESSION_EVALUATION_ENABLED]
                            )
                            elapsed_time = time.time() - start_time

                            # Extract answer
                            answer = response.get('result', 'Sorry, I could not generate a response.')

                            # Display with typing effect
                            utils.display_message_with_typing(answer)

                            # Show latency in debug mode
                            if st.session_state[config.SESSION_DEBUG_MODE]:
                                st.caption(f"Response time: {elapsed_time:.2f}s")
                                if response.get("cached"):
                                    st.caption("(Cached response)")

                            # Show evaluation if available
                            if response.get('evaluation'):
                                eval_result = response['evaluation']
                                evaluation_dict = {
                                    "faithfulness": eval_result.faithfulness,
                                    "answer_relevancy": eval_result.answer_relevancy,
                                    "context_precision": eval_result.context_precision,
                                    "overall_score": eval_result.overall_score
                                }
                                display_inline_evaluation(evaluation_dict)

                            # Show sources if available
                            if response.get('source_documents'):
                                with st.expander("Sources"):
                                    sources = utils.format_sources(response['source_documents'])
                                    st.markdown(sources)

                            # Add assistant response to chat history
                            message_data = {
                                "role": "assistant",
                                "content": answer
                            }

                            if response.get('evaluation'):
                                message_data["evaluation"] = evaluation_dict

                            st.session_state[config.SESSION_MESSAGES].append(message_data)

                        except Exception as e:
                            error_msg = f"Error: {str(e)}"
                            st.error(error_msg)
                            logger.error(f"Chat error: {e}")

        else:
            # Welcome screen when no PDFs are loaded
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.info(config.UPLOAD_PROMPT)

                # Instructions
                st.markdown("""
                ### How to use:
                1. Upload one or more PDF files using the sidebar
                2. Click "Process PDFs" to analyze the documents
                3. Start asking questions in the chat

                ### Features:
                - AI-powered document Q&A
                - Support for multiple PDFs
                - Conversation memory
                - Source citations
                - Response evaluation (RAGAS metrics)
                - Performance monitoring
                - Local processing (your data stays private)
                """)

    # Evaluation Dashboard Tab
    with tab_eval:
        display_evaluation_dashboard()

    # Metrics Tab
    with tab_metrics:
        display_metrics_dashboard()

        if st.session_state[config.SESSION_DEBUG_MODE]:
            st.divider()
            display_debug_panel()


if __name__ == "__main__":
    main()
