"""
Multimodal Agent Module - Class 11: Multimodal Agents

Central orchestrator combining text, vision, and voice capabilities
using LangGraph ReAct agent with multimodal tools.
"""

import logging
import time
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

import config
from vision_handler import VisionHandler, VisionResult
from voice_handler import VoiceHandler, TranscriptionResult, TTSResult
from multimodal_tools import MultimodalToolFactory
from tools import ToolFactory
from memory_manager import MemoryManager, MemoryType

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

class AgentMode(Enum):
    """Operating modes for the multimodal agent."""
    TEXT = "text"
    VISION = "vision"
    VOICE = "voice"
    FULL = "full"


@dataclass
class MultimodalInput:
    """Input container for multimodal agent processing."""
    text: Optional[str] = None
    image: Optional[Any] = None          # PIL Image, path, bytes, UploadedFile
    audio: Optional[Any] = None          # Audio bytes, path, or UploadedFile
    image_prompt: Optional[str] = None   # Specific prompt for image analysis


@dataclass
class MultimodalResponse:
    """Response container from multimodal agent processing."""
    text: str = ""
    audio_bytes: Optional[bytes] = None
    image_analysis: Optional[VisionResult] = None
    transcription: Optional[TranscriptionResult] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    processing_time_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


# =============================================================================
# System Prompt
# =============================================================================

MULTIMODAL_SYSTEM_PROMPT = """You are a helpful multimodal AI assistant with access to various tools including vision and voice capabilities.

Available capabilities:
- Text conversation with memory
- Image analysis and visual question-answering (via LLaVA vision model)
- Speech-to-text transcription (via Whisper)
- Text-to-speech synthesis
- Document search (RAG) over uploaded PDFs
- Calculator, datetime, web search, and Python code execution

When given image analysis context, incorporate the visual information into your response.
When given transcribed audio, respond to the spoken content naturally.
Use tools when needed to provide accurate answers. Think step by step.
Be concise, helpful, and natural in your responses."""


# =============================================================================
# Multimodal Agent
# =============================================================================

class MultimodalAgent:
    """
    Central orchestrator for multimodal interactions.

    Combines standard tools (ToolFactory) + multimodal tools (MultimodalToolFactory)
    with LangGraph ReAct agent and conversation memory.

    Supports 4 modes: TEXT, VISION, VOICE, FULL
    """

    def __init__(
        self,
        mode: Union[AgentMode, str] = None,
        vector_store: Any = None,
        auto_tts: bool = None,
        verbose: bool = None,
    ):
        # Configuration
        if isinstance(mode, str):
            mode = AgentMode(mode.lower())
        self.mode = mode or AgentMode(config.MULTIMODAL_MODE)
        self.auto_tts = auto_tts if auto_tts is not None else config.MULTIMODAL_AUTO_TTS
        self.verbose = verbose if verbose is not None else config.MULTIMODAL_AGENT_VERBOSE

        # Handlers (lazy init for non-required modes)
        self._vision_handler = None
        self._voice_handler = None

        # Tools
        self._tool_factory = ToolFactory(vector_store=vector_store)
        self._mm_tool_factory = MultimodalToolFactory(
            vision_handler=self.vision_handler if self.mode in (AgentMode.VISION, AgentMode.FULL) else None,
            voice_handler=self.voice_handler if self.mode in (AgentMode.VOICE, AgentMode.FULL) else None,
            vector_store=vector_store,
        )

        # Memory
        self._memory_manager = MemoryManager(
            memory_type=MemoryType.BUFFER,
            session_id="multimodal_session",
        )
        self._checkpointer = MemorySaver()
        self._thread_id = "multimodal_thread_1"

        # LLM and Agent (lazy)
        self._llm = None
        self._agent = None

        logger.info(
            f"MultimodalAgent initialized: mode={self.mode.value}, "
            f"auto_tts={self.auto_tts}"
        )

    # =========================================================================
    # Properties (Lazy Initialization)
    # =========================================================================

    @property
    def vision_handler(self) -> VisionHandler:
        if self._vision_handler is None:
            self._vision_handler = VisionHandler()
        return self._vision_handler

    @property
    def voice_handler(self) -> VoiceHandler:
        if self._voice_handler is None:
            self._voice_handler = VoiceHandler()
        return self._voice_handler

    @property
    def llm(self):
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    @property
    def agent(self):
        if self._agent is None:
            self._agent = self._create_agent()
        return self._agent

    # =========================================================================
    # Initialization Helpers
    # =========================================================================

    def _create_llm(self):
        """Create the LLM instance."""
        if config.LLM_PROVIDER == "azure":
            return AzureChatOpenAI(
                azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                api_key=config.AZURE_OPENAI_API_KEY,
                api_version=config.AZURE_OPENAI_API_VERSION,
                deployment_name=config.AZURE_OPENAI_DEPLOYMENT,
                temperature=config.LLM_TEMPERATURE,
            )
        return ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.LLM_TEMPERATURE,
        )

    def _get_tools(self) -> List[BaseTool]:
        """Get combined tools based on current mode."""
        tools = []

        # Standard tools (always available)
        tools.extend(self._tool_factory.get_all_tools())

        # Multimodal tools (based on mode)
        if self.mode in (AgentMode.VISION, AgentMode.FULL):
            mm_tools = self._mm_tool_factory.get_all_multimodal_tools()
            # Filter voice tools if mode is VISION only
            if self.mode == AgentMode.VISION:
                mm_tools = [t for t in mm_tools if "voice" not in t.name and "speech" not in t.name]
            tools.extend(mm_tools)

        elif self.mode == AgentMode.VOICE:
            mm_tools = self._mm_tool_factory.get_all_multimodal_tools()
            # Filter vision tools if mode is VOICE only
            mm_tools = [t for t in mm_tools if "image" not in t.name]
            tools.extend(mm_tools)

        return tools

    def _create_agent(self):
        """Create the LangGraph ReAct agent."""
        tools = self._get_tools()
        return create_react_agent(
            model=self.llm,
            tools=tools,
            checkpointer=self._checkpointer,
        )

    # =========================================================================
    # Main Processing
    # =========================================================================

    def process(self, input: MultimodalInput) -> MultimodalResponse:
        """
        Process a multimodal input through the agent pipeline.

        Pipeline:
        1. If audio provided -> transcribe via VoiceHandler
        2. If image provided -> analyze via VisionHandler, add context
        3. Run LangGraph ReAct agent with combined text query
        4. If auto_tts enabled -> synthesize response audio
        5. Return MultimodalResponse

        Always returns a response, never raises.
        """
        start_time = time.time()
        response = MultimodalResponse()

        try:
            query_parts = []
            transcription = None
            image_analysis = None

            # --- Step 1: Transcribe audio ---
            if input.audio is not None and self.mode in (AgentMode.VOICE, AgentMode.FULL):
                try:
                    transcription = self.voice_handler.transcribe(input.audio)
                    response.transcription = transcription
                    if transcription.success:
                        query_parts.append(transcription.text)
                        logger.info(f"Transcribed audio: {transcription.text[:80]}...")
                    else:
                        logger.warning(f"Transcription failed: {transcription.error}")
                except Exception as e:
                    logger.error(f"Audio transcription error: {e}")

            # --- Step 2: Analyze image ---
            if input.image is not None and self.mode in (AgentMode.VISION, AgentMode.FULL):
                try:
                    image_prompt = input.image_prompt or "Describe this image in detail."
                    image_analysis = self.vision_handler.analyze_image(
                        input.image, prompt=image_prompt
                    )
                    response.image_analysis = image_analysis
                    if image_analysis.success:
                        query_parts.append(
                            f"[Image Analysis: {image_analysis.description}]"
                        )
                        logger.info(f"Image analyzed: {image_analysis.description[:80]}...")
                    else:
                        logger.warning(f"Image analysis failed: {image_analysis.error}")
                except Exception as e:
                    logger.error(f"Image analysis error: {e}")

            # --- Step 3: Build text query ---
            if input.text:
                query_parts.append(input.text)

            if not query_parts:
                response.text = "No input provided. Please send text, an image, or audio."
                response.success = False
                response.error = "empty_input"
                return response

            # --- Step 4: Generate response ---
            # When an image was analyzed, always use the vision model's
            # description as the primary response. The agent LLM (e.g.
            # llama3.2:1b) is too small to properly incorporate vision context
            # and tends to output raw JSON instead of useful text.
            if image_analysis is not None and image_analysis.success and input.image is not None:
                # Direct vision response — the vision model already answered
                user_question = input.text or "Describe this image."
                # If user asked a specific question, re-analyze with that prompt
                if input.image_prompt and input.image_prompt != "Describe this image in detail.":
                    final_response = image_analysis.description
                else:
                    final_response = image_analysis.description
                response.text = final_response
                self._memory_manager.add_user_message(user_question)
                self._memory_manager.add_ai_message(final_response)
                logger.info("Using direct vision response (image input)")
            else:
                # Text-only or audio-only: use the agent LLM
                combined_query = "\n".join(query_parts)
                self._memory_manager.add_user_message(combined_query)

                messages = [
                    SystemMessage(content=MULTIMODAL_SYSTEM_PROMPT),
                    HumanMessage(content=combined_query),
                ]

                config_dict = {"configurable": {"thread_id": self._thread_id}}
                result = self.agent.invoke({"messages": messages}, config=config_dict)

                # Extract response
                output_messages = result.get("messages", [])
                final_response = ""
                tool_calls = []

                for msg in output_messages:
                    if isinstance(msg, AIMessage):
                        if msg.content:
                            final_response = msg.content
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            tool_calls.extend(msg.tool_calls)

                # Clean up raw JSON tool-call artifacts from small LLMs
                # that output function calls as text instead of proper API calls
                if final_response:
                    import re
                    # Remove lines that look like raw JSON function calls
                    cleaned = re.sub(
                        r'\{"type"\s*:\s*"function".*?\}',
                        '', final_response, flags=re.DOTALL
                    ).strip()
                    if cleaned:
                        final_response = cleaned
                    elif tool_calls:
                        # If entire response was JSON artifacts, summarize tool calls
                        tool_names = [tc.get("name", "") for tc in tool_calls]
                        final_response = f"I used the following tools: {', '.join(tool_names)}"

                response.text = final_response
                response.tool_calls = tool_calls

                if final_response:
                    self._memory_manager.add_ai_message(final_response)

            # --- Step 5: Auto TTS ---
            if self.auto_tts and final_response and self.mode in (AgentMode.VOICE, AgentMode.FULL):
                try:
                    tts_result = self.voice_handler.synthesize(final_response)
                    if tts_result.success:
                        response.audio_bytes = tts_result.audio_bytes
                except Exception as e:
                    logger.warning(f"Auto-TTS failed: {e}")

        except Exception as e:
            logger.error(f"MultimodalAgent processing error: {e}")
            response.text = f"I encountered an error processing your request: {str(e)}"
            response.success = False
            response.error = str(e)

        response.processing_time_ms = (time.time() - start_time) * 1000
        return response

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def chat(self, message: str) -> str:
        """Simple text chat returning just the response string."""
        result = self.process(MultimodalInput(text=message))
        return result.text

    def analyze_image(self, image: Any, question: str = None) -> VisionResult:
        """Direct image analysis without going through the full agent."""
        prompt = question or "Describe this image in detail."
        return self.vision_handler.analyze_image(image, prompt=prompt)

    def voice_chat(self, audio: Any) -> MultimodalResponse:
        """Process voice input and return full multimodal response."""
        return self.process(MultimodalInput(audio=audio))

    # =========================================================================
    # Management
    # =========================================================================

    def update_mode(self, mode: Union[AgentMode, str]) -> None:
        """Update the agent operating mode."""
        if isinstance(mode, str):
            mode = AgentMode(mode.lower())
        self.mode = mode
        self._agent = None  # Force recreation
        logger.info(f"Agent mode updated to: {mode.value}")

    def update_vector_store(self, vector_store: Any) -> None:
        """Update the vector store for RAG tools."""
        self._tool_factory.update_vector_store(vector_store)
        self._mm_tool_factory.update_vector_store(vector_store)
        self._agent = None  # Force recreation
        logger.info("Vector store updated")

    def clear_memory(self) -> None:
        """Clear conversation memory and reset agent."""
        self._memory_manager.clear()
        self._checkpointer = MemorySaver()
        self._agent = None
        logger.info("Agent memory cleared")

    def get_info(self) -> Dict[str, Any]:
        """Get agent information for display."""
        tools = self._get_tools()
        return {
            "mode": self.mode.value,
            "auto_tts": self.auto_tts,
            "llm_provider": config.LLM_PROVIDER,
            "llm_model": config.OLLAMA_MODEL if config.LLM_PROVIDER == "ollama" else config.AZURE_OPENAI_DEPLOYMENT,
            "vision_model": config.VISION_MODEL,
            "vision_enabled": config.VISION_ENABLED,
            "voice_enabled": config.VOICE_ENABLED,
            "stt_engine": config.STT_ENGINE,
            "tts_engine": config.TTS_ENGINE,
            "whisper_model": config.WHISPER_MODEL_SIZE,
            "tools": [t.name for t in tools],
            "tool_count": len(tools),
            "memory": self._memory_manager.get_memory_info(),
        }
