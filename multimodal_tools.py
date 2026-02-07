"""
Multimodal Tools Module - Class 11: Multimodal Agents

Custom LangChain tools for vision and voice capabilities.
Extends the BaseTool pattern from tools.py with Pydantic input schemas.
"""

import logging
from typing import Optional, List, Any, Type

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

import config
from vision_handler import VisionHandler
from voice_handler import VoiceHandler

logger = logging.getLogger(__name__)


# =============================================================================
# INPUT SCHEMAS
# =============================================================================

class ImageAnalysisInput(BaseModel):
    """Input schema for image analysis tool."""
    image_path: str = Field(description="Path to the image file to analyze")
    prompt: str = Field(
        default="Describe this image in detail.",
        description="Prompt/instruction for the vision model",
    )


class ImageQuestionInput(BaseModel):
    """Input schema for image question tool."""
    image_path: str = Field(description="Path to the image file")
    question: str = Field(description="Question to ask about the image")


class VoiceTranscriptionInput(BaseModel):
    """Input schema for voice transcription tool."""
    audio_path: str = Field(description="Path to the audio file to transcribe")
    language: str = Field(
        default="en",
        description="Language code for transcription (e.g., 'en', 'es', 'fr')",
    )


class TextToSpeechInput(BaseModel):
    """Input schema for text-to-speech tool."""
    text: str = Field(description="Text to convert to speech")


class ImageToRAGInput(BaseModel):
    """Input schema for image-to-RAG tool."""
    image_path: str = Field(description="Path to the image file")
    collection_name: str = Field(
        default="image_descriptions",
        description="Vector store collection name for the image description",
    )


# =============================================================================
# TOOLS
# =============================================================================

class ImageAnalysisTool(BaseTool):
    """
    Analyze an image using a vision model (LLaVA via Ollama).

    Loads the image, sends it to the vision model with a prompt,
    and returns a detailed description.
    """

    name: str = "image_analysis"
    description: str = (
        "Analyze an image using a vision model. Provide the image file path "
        "and an optional prompt describing what to look for. Returns a detailed "
        "description of the image content."
    )
    args_schema: Type[BaseModel] = ImageAnalysisInput

    _vision_handler: Optional[VisionHandler] = None

    def __init__(self, vision_handler: VisionHandler = None, **kwargs):
        super().__init__(**kwargs)
        self._vision_handler = vision_handler or VisionHandler()

    def _run(
        self,
        image_path: str,
        prompt: str = "Describe this image in detail.",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            result = self._vision_handler.analyze_image(image_path, prompt=prompt)
            if result.success:
                return (
                    f"Image Analysis ({result.processing_time_ms:.0f}ms):\n"
                    f"{result.description}"
                )
            return f"Error analyzing image: {result.error}"
        except Exception as e:
            logger.error(f"ImageAnalysisTool error: {e}")
            return f"Error: {str(e)}"


class ImageQuestionTool(BaseTool):
    """
    Ask a question about an image using visual question-answering.
    """

    name: str = "image_question"
    description: str = (
        "Ask a specific question about an image. Provide the image file path "
        "and your question. The vision model will analyze the image and answer."
    )
    args_schema: Type[BaseModel] = ImageQuestionInput

    _vision_handler: Optional[VisionHandler] = None

    def __init__(self, vision_handler: VisionHandler = None, **kwargs):
        super().__init__(**kwargs)
        self._vision_handler = vision_handler or VisionHandler()

    def _run(
        self,
        image_path: str,
        question: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            result = self._vision_handler.ask_about_image(image_path, question)
            if result.success:
                return f"Answer: {result.description}"
            return f"Error: {result.error}"
        except Exception as e:
            logger.error(f"ImageQuestionTool error: {e}")
            return f"Error: {str(e)}"


class VoiceTranscriptionTool(BaseTool):
    """
    Transcribe audio to text using Whisper or Google Speech Recognition.
    """

    name: str = "voice_transcription"
    description: str = (
        "Transcribe audio to text. Provide the path to an audio file. "
        "Supports WAV, MP3, FLAC, and other common formats."
    )
    args_schema: Type[BaseModel] = VoiceTranscriptionInput

    _voice_handler: Optional[VoiceHandler] = None

    def __init__(self, voice_handler: VoiceHandler = None, **kwargs):
        super().__init__(**kwargs)
        self._voice_handler = voice_handler or VoiceHandler()

    def _run(
        self,
        audio_path: str,
        language: str = "en",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            result = self._voice_handler.transcribe(audio_path, language=language)
            if result.success:
                return (
                    f"Transcription (lang={result.language}, "
                    f"confidence={result.confidence:.2f}):\n{result.text}"
                )
            return f"Error transcribing audio: {result.error}"
        except Exception as e:
            logger.error(f"VoiceTranscriptionTool error: {e}")
            return f"Error: {str(e)}"


class TextToSpeechTool(BaseTool):
    """
    Convert text to speech audio using gTTS or pyttsx3.
    """

    name: str = "text_to_speech"
    description: str = (
        "Convert text to speech audio. Provide the text to convert. "
        "Returns a confirmation and the path to the generated audio file."
    )
    args_schema: Type[BaseModel] = TextToSpeechInput

    _voice_handler: Optional[VoiceHandler] = None

    def __init__(self, voice_handler: VoiceHandler = None, **kwargs):
        super().__init__(**kwargs)
        self._voice_handler = voice_handler or VoiceHandler()

    def _run(
        self,
        text: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            result = self._voice_handler.synthesize(text)
            if result.success:
                # Save to file
                filepath = self._voice_handler.save_audio_bytes(result.audio_bytes)
                return (
                    f"Speech generated successfully ({result.engine}, "
                    f"{len(result.audio_bytes)} bytes). Saved to: {filepath}"
                )
            return f"Error generating speech: {result.error}"
        except Exception as e:
            logger.error(f"TextToSpeechTool error: {e}")
            return f"Error: {str(e)}"


class ImageToRAGTool(BaseTool):
    """
    Analyze an image and store the description in the vector database
    for later retrieval via RAG.
    """

    name: str = "image_to_rag"
    description: str = (
        "Analyze an image using a vision model and store the description "
        "in the vector database for later retrieval. Useful for indexing "
        "image content for search."
    )
    args_schema: Type[BaseModel] = ImageToRAGInput

    _vision_handler: Optional[VisionHandler] = None
    _vector_store: Any = None

    def __init__(
        self,
        vision_handler: VisionHandler = None,
        vector_store: Any = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._vision_handler = vision_handler or VisionHandler()
        self._vector_store = vector_store

    def _run(
        self,
        image_path: str,
        collection_name: str = "image_descriptions",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            if self._vector_store is None:
                return "No vector store available. Please upload documents first."

            # Analyze the image
            result = self._vision_handler.analyze_image(
                image_path,
                prompt="Describe this image in detail for document indexing.",
            )

            if not result.success:
                return f"Error analyzing image: {result.error}"

            # Store in vector DB
            from langchain_core.documents import Document

            doc = Document(
                page_content=result.description,
                metadata={
                    "source": image_path,
                    "type": "image_description",
                    "model": result.model,
                    "image_size": str(result.image_size),
                },
            )
            self._vector_store.add_documents([doc])

            return (
                f"Image analyzed and stored in vector DB.\n"
                f"Description: {result.description[:200]}..."
            )

        except Exception as e:
            logger.error(f"ImageToRAGTool error: {e}")
            return f"Error: {str(e)}"


# =============================================================================
# TOOL FACTORY
# =============================================================================

class MultimodalToolFactory:
    """
    Factory for creating multimodal tools.

    Consistent with ToolFactory from tools.py.
    Creates and manages vision/voice tools based on configuration.
    """

    def __init__(
        self,
        vision_handler: VisionHandler = None,
        voice_handler: VoiceHandler = None,
        vector_store: Any = None,
    ):
        self._vision_handler = vision_handler
        self._voice_handler = voice_handler
        self._vector_store = vector_store

        # Lazy init handlers
        self._vision_handler_init = False
        self._voice_handler_init = False

    @property
    def vision_handler(self) -> VisionHandler:
        if self._vision_handler is None and not self._vision_handler_init:
            self._vision_handler = VisionHandler()
            self._vision_handler_init = True
        return self._vision_handler

    @property
    def voice_handler(self) -> VoiceHandler:
        if self._voice_handler is None and not self._voice_handler_init:
            self._voice_handler = VoiceHandler()
            self._voice_handler_init = True
        return self._voice_handler

    def get_image_analysis_tool(self) -> ImageAnalysisTool:
        return ImageAnalysisTool(vision_handler=self.vision_handler)

    def get_image_question_tool(self) -> ImageQuestionTool:
        return ImageQuestionTool(vision_handler=self.vision_handler)

    def get_voice_transcription_tool(self) -> VoiceTranscriptionTool:
        return VoiceTranscriptionTool(voice_handler=self.voice_handler)

    def get_text_to_speech_tool(self) -> TextToSpeechTool:
        return TextToSpeechTool(voice_handler=self.voice_handler)

    def get_image_to_rag_tool(self) -> ImageToRAGTool:
        return ImageToRAGTool(
            vision_handler=self.vision_handler,
            vector_store=self._vector_store,
        )

    def get_all_multimodal_tools(self) -> List[BaseTool]:
        """
        Get all enabled multimodal tools based on config.

        Returns:
            List of enabled multimodal tool instances
        """
        tools = []
        enabled = config.MULTIMODAL_TOOLS_ENABLED

        if enabled.get("image_analysis", True) and config.VISION_ENABLED:
            tools.append(self.get_image_analysis_tool())

        if enabled.get("image_question", True) and config.VISION_ENABLED:
            tools.append(self.get_image_question_tool())

        if enabled.get("voice_transcription", True) and config.VOICE_ENABLED:
            tools.append(self.get_voice_transcription_tool())

        if enabled.get("text_to_speech", True) and config.VOICE_ENABLED:
            tools.append(self.get_text_to_speech_tool())

        if enabled.get("image_to_rag", True) and config.VISION_ENABLED:
            tools.append(self.get_image_to_rag_tool())

        logger.info(f"Created {len(tools)} multimodal tools: {[t.name for t in tools]}")
        return tools

    def update_vector_store(self, vector_store: Any) -> None:
        """Update the vector store for ImageToRAG tool."""
        self._vector_store = vector_store
