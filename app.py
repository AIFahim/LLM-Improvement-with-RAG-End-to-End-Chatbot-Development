"""
Streamlit application for the RAG Chatbot
Enhanced with evaluation UI, metrics dashboard, LLM Judge, and debug mode
"""
import streamlit as st
import logging
import time
from chatbot import RAGChatbot
from evaluator import EvaluationResult
from llm_judge import llm_judge, JudgmentCriteria
import config
import utils

# Setup logging
utils.setup_logging()
logger = logging.getLogger(__name__)


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'chatbot' not in st.session_state:
        # Default to fast evaluation for better UX (RAGAS can be slow)
        st.session_state.chatbot = RAGChatbot(use_fast_evaluation=True)
        logger.info("Initialized new chatbot instance with fast evaluation")

    if config.SESSION_MESSAGES not in st.session_state:
        st.session_state[config.SESSION_MESSAGES] = []

    if config.SESSION_EVALUATION_ENABLED not in st.session_state:
        st.session_state[config.SESSION_EVALUATION_ENABLED] = config.EVALUATION_ENABLED

    if config.SESSION_DEBUG_MODE not in st.session_state:
        st.session_state[config.SESSION_DEBUG_MODE] = False

    if 'fast_evaluation' not in st.session_state:
        st.session_state.fast_evaluation = True  # Default to fast evaluation for better UX

    if 'llm_judge_enabled' not in st.session_state:
        st.session_state.llm_judge_enabled = False  # LLM Judge off by default (slow)

    if 'judge_criteria' not in st.session_state:
        st.session_state.judge_criteria = "correctness"

    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = []

    if 'judge_history' not in st.session_state:
        st.session_state.judge_history = []


def display_chat_messages():
    """Display chat messages from history"""
    for message in st.session_state[config.SESSION_MESSAGES]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display evaluation if available
            if message.get("evaluation") and st.session_state[config.SESSION_EVALUATION_ENABLED]:
                display_inline_evaluation(message["evaluation"])

            # Display LLM Judge result if available
            if message.get("judge_result"):
                display_llm_judge_result(message["judge_result"])


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


def display_llm_judge_result(judge_result: dict):
    """Display LLM Judge result inline with the message"""
    if not judge_result:
        return

    with st.expander("LLM Judge Result", expanded=False):
        score = judge_result.get("score")
        reason = judge_result.get("reason", "No reason provided")
        criteria = judge_result.get("criteria", "unknown")

        # Display score with color
        if score is not None:
            col1, col2 = st.columns([1, 3])

            with col1:
                if score >= 0.7:
                    st.success(f"Score: {score:.2f}")
                elif score >= 0.4:
                    st.warning(f"Score: {score:.2f}")
                else:
                    st.error(f"Score: {score:.2f}")

            with col2:
                st.caption(f"Criteria: {criteria.title()}")

        # Display reasoning
        st.markdown("**Reasoning:**")
        st.markdown(f"_{reason}_")


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


def display_llm_judge_dashboard():
    """Display LLM Judge dashboard with history and comparison tool"""
    st.header("LLM-as-a-Judge Dashboard")

    # Judge history summary
    judge_history = st.session_state.judge_history

    if judge_history:
        st.subheader("Judge History")

        # Summary metrics
        col1, col2, col3 = st.columns(3)

        total_judgments = len(judge_history)
        scores = [j.get("score") for j in judge_history if j.get("score") is not None]
        avg_score = sum(scores) / len(scores) if scores else None

        col1.metric("Total Judgments", total_judgments)
        col2.metric("Avg Score", f"{avg_score:.2f}" if avg_score else "N/A")
        col3.metric("Latest Criteria", judge_history[-1].get("criteria", "N/A").title() if judge_history else "N/A")

        # Score chart
        if len(scores) > 1:
            import pandas as pd
            chart_data = pd.DataFrame({
                "Judgment #": list(range(1, len(scores) + 1)),
                "Score": scores
            })
            chart_data = chart_data.set_index("Judgment #")
            st.line_chart(chart_data)

        # Recent judgments
        st.subheader("Recent Judgments")
        for i, judgment in enumerate(reversed(judge_history[-5:])):
            with st.expander(f"Query: {judgment.get('query', 'N/A')[:40]}..."):
                st.write(f"**Score:** {judgment.get('score', 'N/A')}")
                st.write(f"**Criteria:** {judgment.get('criteria', 'N/A').title()}")
    else:
        st.info("No LLM Judge results yet. Enable LLM Judge in the sidebar and ask questions!")

    # Response Comparison Tool
    st.divider()
    st.subheader("Compare Responses (A/B Testing)")
    st.caption("Compare two different responses to the same question")

    with st.form("compare_responses"):
        question = st.text_input(
            "Question",
            placeholder="Enter the question to compare responses for..."
        )

        col1, col2 = st.columns(2)

        with col1:
            response_a = st.text_area(
                "Response A",
                placeholder="Enter first response...",
                height=150
            )

        with col2:
            response_b = st.text_area(
                "Response B",
                placeholder="Enter second response...",
                height=150
            )

        submit = st.form_submit_button("Compare Responses", type="primary")

        if submit and question and response_a and response_b:
            with st.spinner("Comparing responses with LLM Judge..."):
                try:
                    result = st.session_state.chatbot.compare_responses(
                        question, response_a, response_b
                    )

                    st.success("Comparison complete!")

                    # Display results
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Response A Score", f"{result.get('score_a', 'N/A')}")

                    with col2:
                        winner = result.get("winner", "tie")
                        if winner == "A":
                            st.success("Winner: Response A")
                        elif winner == "B":
                            st.success("Winner: Response B")
                        else:
                            st.info("Result: Tie")

                    with col3:
                        st.metric("Response B Score", f"{result.get('score_b', 'N/A')}")

                    # Reasoning
                    st.markdown("**Reasoning:**")
                    st.markdown(f"_{result.get('reason', 'No reason provided')}_")

                except Exception as e:
                    st.error(f"Comparison failed: {e}")
                    logger.error(f"Response comparison error: {e}")


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
    tab_chat, tab_eval, tab_judge, tab_metrics = st.tabs([
        "Chat",
        "Evaluation Dashboard",
        "LLM Judge",
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
                help="Evaluate responses using evaluation metrics"
            )
            st.session_state[config.SESSION_EVALUATION_ENABLED] = eval_enabled
            st.session_state.chatbot.evaluation_enabled = eval_enabled

            # Fast evaluation toggle (only show when evaluation is enabled)
            if eval_enabled:
                fast_eval = st.checkbox(
                    "Use Fast Evaluation",
                    value=st.session_state.fast_evaluation,
                    help="Fast heuristic-based evaluation (instant) vs RAGAS (slower but more accurate)"
                )
                st.session_state.fast_evaluation = fast_eval
                st.session_state.chatbot.fast_evaluation = fast_eval

                if fast_eval:
                    st.caption("Using fast heuristic evaluation")
                else:
                    st.caption("Using RAGAS evaluation (may be slow)")

            # LLM Judge toggle
            st.divider()
            st.subheader("LLM-as-a-Judge")

            llm_judge_enabled = st.checkbox(
                "Enable LLM Judge",
                value=st.session_state.llm_judge_enabled,
                help="Use LLM to judge response quality with Chain-of-Thought reasoning"
            )
            st.session_state.llm_judge_enabled = llm_judge_enabled

            if llm_judge_enabled:
                # Criteria selection
                criteria_options = {
                    "correctness": "Correctness - Is the answer factually correct?",
                    "relevance": "Relevance - Does it answer the question?",
                    "coherence": "Coherence - Is it well-structured?",
                    "helpfulness": "Helpfulness - Is it useful to the user?",
                    "completeness": "Completeness - Does it fully address the query?"
                }

                selected_criteria = st.selectbox(
                    "Judge Criteria",
                    options=list(criteria_options.keys()),
                    format_func=lambda x: criteria_options[x],
                    index=list(criteria_options.keys()).index(st.session_state.judge_criteria)
                )
                st.session_state.judge_criteria = selected_criteria
                st.caption("LLM Judge uses Ollama for evaluation (may be slow)")

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

                            # Run LLM Judge if enabled
                            judge_result = None
                            if st.session_state.llm_judge_enabled:
                                with st.spinner("Running LLM Judge..."):
                                    try:
                                        # Get contexts
                                        contexts = [
                                            doc.page_content
                                            for doc in response.get("source_documents", [])
                                        ]

                                        # Map criteria string to enum
                                        criteria_map = {
                                            "correctness": JudgmentCriteria.CORRECTNESS,
                                            "relevance": JudgmentCriteria.RELEVANCE,
                                            "coherence": JudgmentCriteria.COHERENCE,
                                            "helpfulness": JudgmentCriteria.HELPFULNESS,
                                            "completeness": JudgmentCriteria.COMPLETENESS
                                        }
                                        criteria = criteria_map.get(
                                            st.session_state.judge_criteria,
                                            JudgmentCriteria.CORRECTNESS
                                        )

                                        # Run judge
                                        judge_result = st.session_state.chatbot.judge_response(
                                            prompt, answer, contexts, criteria
                                        )

                                        # Store in history
                                        st.session_state.judge_history.append({
                                            "query": prompt[:100],
                                            "score": judge_result.get("score"),
                                            "criteria": judge_result.get("criteria"),
                                            "timestamp": time.time()
                                        })

                                    except Exception as e:
                                        logger.error(f"LLM Judge error: {e}")
                                        judge_result = {"score": None, "reason": str(e)}

                                display_llm_judge_result(judge_result)

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

                            if judge_result:
                                message_data["judge_result"] = judge_result

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

    # LLM Judge Tab
    with tab_judge:
        display_llm_judge_dashboard()

    # Metrics Tab
    with tab_metrics:
        display_metrics_dashboard()

        if st.session_state[config.SESSION_DEBUG_MODE]:
            st.divider()
            display_debug_panel()


if __name__ == "__main__":
    main()
