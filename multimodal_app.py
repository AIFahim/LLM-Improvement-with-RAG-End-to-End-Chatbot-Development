"""
Multimodal Streamlit Application - Class 11: Multimodal Agents

4-tab Streamlit UI for the Multimodal Voice Assistant Agent:
- Tab 1: Multimodal Chat (text + image + voice)
- Tab 2: Vision Studio (image analysis playground)
- Tab 3: Voice Lab (STT + TTS testing)
- Tab 4: Agent Info (tools, config, memory)
"""

import streamlit as st
import logging
import time
import json
from pathlib import Path

import config
import utils
from multimodal_agent import MultimodalAgent, MultimodalInput, MultimodalResponse, AgentMode
from vision_handler import VisionHandler
from voice_handler import VoiceHandler
from chatbot import RAGChatbot
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager

# Setup logging
utils.setup_logging()
logger = logging.getLogger(__name__)


# =============================================================================
# Session State Initialization
# =============================================================================

def initialize_session_state():
    """Initialize all session state variables."""
    if "multimodal_agent" not in st.session_state:
        st.session_state.multimodal_agent = MultimodalAgent()
        logger.info("Initialized MultimodalAgent")

    if "mm_messages" not in st.session_state:
        st.session_state.mm_messages = []

    if "mm_mode" not in st.session_state:
        st.session_state.mm_mode = config.MULTIMODAL_MODE

    if "mm_auto_tts" not in st.session_state:
        st.session_state.mm_auto_tts = config.MULTIMODAL_AUTO_TTS

    # Pending attachments (persist across Streamlit reruns)
    if "pending_image" not in st.session_state:
        st.session_state.pending_image = None
    if "pending_image_bytes" not in st.session_state:
        st.session_state.pending_image_bytes = None
    if "pending_audio" not in st.session_state:
        st.session_state.pending_audio = None
    # Track which file_uploader file IDs have been sent, to avoid re-attaching
    if "sent_image_id" not in st.session_state:
        st.session_state.sent_image_id = None
    if "sent_audio_id" not in st.session_state:
        st.session_state.sent_audio_id = None

    if "vector_store_manager" not in st.session_state:
        st.session_state.vector_store_manager = None

    if "vision_handler" not in st.session_state:
        st.session_state.vision_handler = VisionHandler()

    if "voice_handler" not in st.session_state:
        st.session_state.voice_handler = VoiceHandler()


# =============================================================================
# Tab 1: Multimodal Chat
# =============================================================================

def display_multimodal_chat():
    """ChatGPT-style multimodal chat interface with inline attachments."""

    # Display chat history
    for msg in st.session_state.mm_messages:
        with st.chat_message(msg["role"], avatar=msg.get("avatar")):
            # Show attached image inline (user messages)
            if msg.get("image_bytes"):
                st.image(msg["image_bytes"], width=250)

            # Show transcribed voice as quote (user messages)
            if msg.get("voice_text"):
                st.caption(f"*{msg['voice_text']}*")

            # Main text content
            if msg["content"]:
                st.markdown(msg["content"])

            # Audio playback (assistant TTS response)
            if msg.get("audio_bytes"):
                st.audio(msg["audio_bytes"], format="audio/wav")

            # Expandable details (assistant messages only)
            if msg["role"] == "assistant":
                details = []
                if msg.get("transcription"):
                    tr = msg["transcription"]
                    details.append(
                        f"**Transcription:** {tr.get('text', '')} "
                        f"(lang: {tr.get('language', '')}, conf: {tr.get('confidence', 0):.0%})"
                    )
                if msg.get("image_analysis"):
                    ia = msg["image_analysis"]
                    details.append(f"**Vision:** {ia.get('description', '')[:200]}...")
                if msg.get("tool_calls"):
                    details.append(f"**Tools used:** {', '.join(tc.get('name', '') for tc in msg['tool_calls'])}")
                if msg.get("processing_time_ms"):
                    details.append(f"**Time:** {msg['processing_time_ms']:.0f}ms")

                if details:
                    with st.expander("Details"):
                        for d in details:
                            st.markdown(d)
                        if msg.get("tool_calls"):
                            for tc in msg["tool_calls"]:
                                st.code(json.dumps(tc, indent=2, default=str), language="json")

    # Show pending attachment indicator above chat input
    has_pending_image = st.session_state.pending_image_bytes is not None
    has_pending_audio = st.session_state.pending_audio is not None
    send_attachments = False

    if has_pending_image or has_pending_audio:
        previews = []
        if has_pending_image:
            previews.append("image")
        if has_pending_audio:
            previews.append("voice")
        attach_col, send_col = st.columns([4, 1])
        with attach_col:
            st.info(f"Attached: {', '.join(previews)} — type a message or click Send")
        with send_col:
            send_attachments = st.button("Send", type="primary", key="send_attachments_btn")

    # Chat input pinned to bottom by Streamlit
    user_text = st.chat_input("Message...")

    # --- Determine if we should process ---
    # Process when: user typed text, OR user clicked Send with attachments
    should_process = user_text is not None or send_attachments
    has_image = has_pending_image
    has_audio = has_pending_audio

    if should_process and (user_text or has_image or has_audio):
        mm_input = MultimodalInput()

        import io as _io

        if user_text:
            mm_input.text = user_text

        image_bytes_for_display = None
        if has_image:
            image_bytes_for_display = st.session_state.pending_image_bytes
            mm_input.image = _io.BytesIO(image_bytes_for_display)
            if user_text:
                # Pass user's question directly to the vision model
                mm_input.image_prompt = user_text
            else:
                # Default prompt when image sent without text
                mm_input.text = "What's in this image? Describe it."
                mm_input.image_prompt = "Describe this image in detail."

        if has_audio:
            mm_input.audio = _io.BytesIO(st.session_state.pending_audio)

        # Clear pending attachments and mark as sent
        if has_image and st.session_state.pending_image is not None:
            img = st.session_state.pending_image
            st.session_state.sent_image_id = f"{img.name}_{img.size}"
        if has_audio:
            st.session_state.sent_audio_id = f"audio_{len(st.session_state.pending_audio)}"
        st.session_state.pending_image = None
        st.session_state.pending_image_bytes = None
        st.session_state.pending_audio = None

        # Build display text
        display_text = user_text or ""
        if not display_text and has_image:
            display_text = "What's in this image?"
        if not display_text and has_audio:
            display_text = ""

        # Save & show user message
        user_msg = {"role": "user", "content": display_text}
        if image_bytes_for_display:
            user_msg["image_bytes"] = image_bytes_for_display

        with st.chat_message("user"):
            if image_bytes_for_display:
                st.image(image_bytes_for_display, width=250)
            if display_text:
                st.markdown(display_text)

        # Process with agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                agent = st.session_state.multimodal_agent
                response = agent.process(mm_input)

            # Show transcribed voice text if no text was typed
            if response.transcription and response.transcription.success:
                if not user_text:
                    user_msg["voice_text"] = response.transcription.text
                    user_msg["content"] = response.transcription.text

            st.session_state.mm_messages.append(user_msg)

            # Main response
            st.markdown(response.text)

            # Audio playback
            if response.audio_bytes:
                st.audio(response.audio_bytes, format="audio/wav")

            # Build assistant message
            assistant_msg = {
                "role": "assistant",
                "content": response.text,
                "processing_time_ms": response.processing_time_ms,
            }

            if response.transcription and response.transcription.success:
                assistant_msg["transcription"] = {
                    "text": response.transcription.text,
                    "language": response.transcription.language,
                    "confidence": response.transcription.confidence,
                }

            if response.image_analysis and response.image_analysis.success:
                assistant_msg["image_analysis"] = {
                    "description": response.image_analysis.description,
                    "model": response.image_analysis.model,
                    "processing_time_ms": response.image_analysis.processing_time_ms,
                }

            if response.tool_calls:
                assistant_msg["tool_calls"] = [
                    {"name": tc.get("name", ""), "args": tc.get("args", {})}
                    for tc in response.tool_calls
                ]

            if response.audio_bytes:
                assistant_msg["audio_bytes"] = response.audio_bytes

            # Show details expander
            details = []
            if assistant_msg.get("transcription"):
                tr = assistant_msg["transcription"]
                details.append(f"**Transcription:** {tr['text']} (conf: {tr['confidence']:.0%})")
            if assistant_msg.get("image_analysis"):
                ia = assistant_msg["image_analysis"]
                details.append(f"**Vision:** {ia['description'][:200]}...")
            if assistant_msg.get("tool_calls"):
                details.append(f"**Tools:** {', '.join(tc['name'] for tc in assistant_msg['tool_calls'])}")
            details.append(f"**Time:** {response.processing_time_ms:.0f}ms")

            with st.expander("Details"):
                for d in details:
                    st.markdown(d)

            st.session_state.mm_messages.append(assistant_msg)


# =============================================================================
# Tab 2: Vision Studio
# =============================================================================

def display_vision_studio():
    """Image analysis playground."""
    st.header("Vision Studio")
    st.caption("Upload images and analyze them with the LLaVA vision model")

    uploaded = st.file_uploader(
        "Upload an image",
        type=list(config.VISION_SUPPORTED_FORMATS),
        key="vision_upload",
    )

    question = st.text_area(
        "What would you like to know about this image?",
        value="Describe this image in detail.",
        height=100,
        key="vision_question",
    )

    if uploaded and st.button("Analyze Image", type="primary", key="vision_analyze_btn"):
        col_img, col_result = st.columns(2)

        with col_img:
            st.image(uploaded, caption="Uploaded Image", use_container_width=True)

        with col_result:
            with st.spinner("Analyzing image with LLaVA..."):
                uploaded.seek(0)
                vh = st.session_state.vision_handler
                result = vh.analyze_image(uploaded, prompt=question)

            if result.success:
                st.success("Analysis Complete")
                st.write(result.description)
                st.divider()
                st.caption(f"Model: {result.model}")
                st.caption(f"Processing time: {result.processing_time_ms:.0f}ms")
                st.caption(f"Image size: {result.image_size}")
            else:
                st.error(f"Analysis failed: {result.error}")

    elif uploaded:
        st.image(uploaded, caption="Uploaded Image", width=400)


# =============================================================================
# Tab 3: Voice Lab
# =============================================================================

def display_voice_lab():
    """Speech-to-Text and Text-to-Speech playground."""
    st.header("Voice Lab")

    col_stt, col_tts = st.columns(2)

    # --- STT Section ---
    with col_stt:
        st.subheader("Speech-to-Text")
        st.caption("Record or upload audio to transcribe")

        audio_input = st.audio_input("Record audio", key="voice_lab_record")
        audio_file = st.file_uploader(
            "Or upload an audio file",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            key="voice_lab_upload",
        )

        audio_source = audio_input or audio_file

        if audio_source and st.button("Transcribe", type="primary", key="stt_btn"):
            with st.spinner("Transcribing..."):
                audio_source.seek(0)
                vh = st.session_state.voice_handler
                result = vh.transcribe(audio_source)

            if result.success:
                st.success("Transcription Complete")
                st.text_area("Transcribed Text", value=result.text, height=150, key="stt_result")
                st.caption(
                    f"Language: {result.language} | "
                    f"Confidence: {result.confidence:.2f} | "
                    f"Engine: {result.engine} | "
                    f"Time: {result.duration_seconds:.1f}s"
                )
            else:
                st.error(f"Transcription failed: {result.error}")

    # --- TTS Section ---
    with col_tts:
        st.subheader("Text-to-Speech")
        st.caption("Enter text and generate speech audio")

        # TTS engine selector
        tts_engines = {"gTTS (Google)": "gtts", "pyttsx3 (Offline)": "pyttsx3", "Orpheus (Ollama)": "orpheus"}
        selected_engine = st.selectbox(
            "TTS Engine",
            options=list(tts_engines.keys()),
            key="tts_engine_select",
        )
        tts_engine = tts_engines[selected_engine]

        # Orpheus voice selector
        if tts_engine == "orpheus":
            voices = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
            voice_descriptions = {
                "tara": "Female, conversational",
                "leah": "Female, warm",
                "jess": "Female, energetic",
                "leo": "Male, authoritative",
                "dan": "Male, friendly",
                "mia": "Female, professional",
                "zac": "Male, enthusiastic",
                "zoe": "Female, calm",
            }
            selected_voice = st.selectbox(
                "Voice",
                options=voices,
                format_func=lambda v: f"{v} ({voice_descriptions[v]})",
                key="orpheus_voice_select",
            )

        tts_text = st.text_area(
            "Enter text to speak",
            value="Hello! I am a multimodal AI assistant.",
            height=150,
            key="tts_input",
        )

        if tts_text and st.button("Generate Speech", type="primary", key="tts_btn"):
            with st.spinner("Generating speech..."):
                vh = st.session_state.voice_handler
                # Temporarily set engine for this request
                original_engine = vh.tts_engine
                vh.tts_engine = tts_engine
                if tts_engine == "orpheus":
                    import config as cfg
                    cfg.ORPHEUS_VOICE = selected_voice
                result = vh.synthesize(tts_text)
                vh.tts_engine = original_engine

            if result.success:
                st.success("Speech Generated")
                st.audio(result.audio_bytes, format=f"audio/{result.format}")
                st.caption(
                    f"Engine: {result.engine} | "
                    f"Size: {len(result.audio_bytes):,} bytes | "
                    f"Time: {result.duration_seconds:.1f}s"
                )
            else:
                st.error(f"TTS failed: {result.error}")


# =============================================================================
# Tab 4: Agent Info
# =============================================================================

def display_agent_info():
    """Display agent configuration and status."""
    st.header("Agent Information")

    agent = st.session_state.multimodal_agent
    info = agent.get_info()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Configuration")
        st.json({
            "mode": info["mode"],
            "auto_tts": info["auto_tts"],
            "llm_provider": info["llm_provider"],
            "llm_model": info["llm_model"],
            "vision_model": info["vision_model"],
            "vision_enabled": info["vision_enabled"],
            "voice_enabled": info["voice_enabled"],
            "stt_engine": info["stt_engine"],
            "tts_engine": info["tts_engine"],
            "whisper_model": info["whisper_model"],
        })

    with col2:
        st.subheader("Tools")
        st.metric("Total Tools", info["tool_count"])
        for tool_name in info["tools"]:
            st.write(f"- `{tool_name}`")

    st.subheader("Memory")
    st.json(info["memory"])

    # Model availability check
    st.subheader("System Status")
    col_v, col_a = st.columns(2)

    with col_v:
        if st.button("Check Vision Model", key="check_vision"):
            vh = st.session_state.vision_handler
            available = vh.check_model_available()
            if available:
                st.success(f"Vision model '{config.VISION_MODEL}' is available")
            else:
                st.error(
                    f"Vision model '{config.VISION_MODEL}' not found. "
                    f"Run: `ollama pull {config.VISION_MODEL}`"
                )

    with col_a:
        try:
            import whisper
            st.success(f"Whisper is installed (model: {config.WHISPER_MODEL_SIZE})")
        except ImportError:
            st.warning("Whisper not installed. Run: `pip install openai-whisper`")


# =============================================================================
# Sidebar
# =============================================================================

def display_sidebar():
    """Sidebar with settings and controls."""
    with st.sidebar:
        st.title("Multimodal Agent")

        # Mode selector
        mode_options = {"Full": "full", "Text Only": "text", "Vision": "vision", "Voice": "voice"}
        selected_mode = st.selectbox(
            "Agent Mode",
            options=list(mode_options.keys()),
            index=list(mode_options.values()).index(st.session_state.mm_mode),
            key="mode_selector",
        )
        new_mode = mode_options[selected_mode]
        if new_mode != st.session_state.mm_mode:
            st.session_state.mm_mode = new_mode
            st.session_state.multimodal_agent.update_mode(new_mode)
            st.rerun()

        # Auto TTS toggle
        auto_tts = st.toggle(
            "Auto Text-to-Speech",
            value=st.session_state.mm_auto_tts,
            key="auto_tts_toggle",
        )
        if auto_tts != st.session_state.mm_auto_tts:
            st.session_state.mm_auto_tts = auto_tts
            st.session_state.multimodal_agent.auto_tts = auto_tts

        st.divider()

        # --- Attachments for Multimodal Chat ---
        st.subheader("Attachments")

        uploaded_image = st.file_uploader(
            "Attach Image",
            type=list(config.VISION_SUPPORTED_FORMATS),
            key="chat_image_upload",
        )
        if uploaded_image is not None:
            img_key = f"{uploaded_image.name}_{uploaded_image.size}"
            if img_key != st.session_state.sent_image_id:
                uploaded_image.seek(0)
                st.session_state.pending_image_bytes = uploaded_image.getvalue()
                uploaded_image.seek(0)
                st.session_state.pending_image = uploaded_image
                st.image(st.session_state.pending_image_bytes, width=150, caption="Pending")

        audio_input = st.audio_input("Record Voice", key="chat_audio_input")
        if audio_input is not None:
            audio_key = f"audio_{audio_input.size}"
            if audio_key != st.session_state.sent_audio_id:
                audio_input.seek(0)
                st.session_state.pending_audio = audio_input.read()
                audio_input.seek(0)

        st.divider()

        # Model info
        with st.expander("Vision Model Info"):
            st.write(f"**Model:** {config.VISION_MODEL}")
            st.write(f"**URL:** {config.VISION_BASE_URL}")
            st.write(f"**Max Tokens:** {config.VISION_MAX_TOKENS}")
            st.write(f"**Enabled:** {config.VISION_ENABLED}")

        with st.expander("Voice Model Info"):
            st.write(f"**STT Engine:** {config.STT_ENGINE}")
            st.write(f"**Whisper Model:** {config.WHISPER_MODEL_SIZE}")
            st.write(f"**TTS Engine:** {config.TTS_ENGINE}")
            st.write(f"**Language:** {config.TTS_LANGUAGE}")
            st.write(f"**Enabled:** {config.VOICE_ENABLED}")
            st.divider()
            st.write(f"**Orpheus Model:** {config.ORPHEUS_MODEL}")
            st.write(f"**Orpheus Voice:** {config.ORPHEUS_VOICE}")
            st.write(f"**Orpheus URL:** {config.ORPHEUS_BASE_URL}")

        st.divider()

        # PDF Upload for RAG
        st.subheader("Document Upload")
        uploaded_pdfs = st.file_uploader(
            "Upload PDFs for RAG",
            type=["pdf"],
            accept_multiple_files=True,
            key="sidebar_pdf_upload",
        )

        if uploaded_pdfs and st.button("Process PDFs", key="process_pdfs_btn"):
            with st.spinner("Processing PDFs..."):
                try:
                    processor = DocumentProcessor()
                    all_chunks = []
                    for pdf in uploaded_pdfs:
                        # Save uploaded PDF
                        pdf_path = config.PDF_DIR / pdf.name
                        pdf_path.write_bytes(pdf.getvalue())
                        chunks = processor.process_pdf(str(pdf_path))
                        all_chunks.extend(chunks)

                    if all_chunks:
                        vs_manager = VectorStoreManager()
                        vs_manager.add_documents(all_chunks)
                        st.session_state.vector_store_manager = vs_manager
                        st.session_state.multimodal_agent.update_vector_store(
                            vs_manager.vector_store
                        )
                        st.success(f"Processed {len(uploaded_pdfs)} PDFs ({len(all_chunks)} chunks)")
                    else:
                        st.warning("No text chunks extracted from PDFs")
                except Exception as e:
                    st.error(f"Error processing PDFs: {e}")

        st.divider()

        # Available tools
        with st.expander("Available Tools"):
            info = st.session_state.multimodal_agent.get_info()
            for tool_name in info["tools"]:
                st.write(f"- {tool_name}")

        # Clear history
        if st.button("Clear Chat History", key="clear_history_btn"):
            st.session_state.mm_messages = []
            st.session_state.multimodal_agent.clear_memory()
            st.rerun()


# =============================================================================
# Main Application
# =============================================================================

def main():
    st.set_page_config(
        page_title="Multimodal Agent",
        page_icon="🎭",
        layout="wide",
    )

    initialize_session_state()
    display_sidebar()

    # Main tabs
    tab_chat, tab_vision, tab_voice, tab_info = st.tabs([
        "Multimodal Chat",
        "Vision Studio",
        "Voice Lab",
        "Agent Info",
    ])

    with tab_chat:
        display_multimodal_chat()

    with tab_vision:
        display_vision_studio()

    with tab_voice:
        display_voice_lab()

    with tab_info:
        display_agent_info()


if __name__ == "__main__":
    main()
