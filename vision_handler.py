"""
Vision Handler Module - Class 11: Multimodal Agents

Image processing and analysis using LLaVA via Ollama raw HTTP API.
Provides image loading, preprocessing, and visual question-answering.
"""

import base64
import io
import time
import logging
from typing import Optional, Any, List
from dataclasses import dataclass, field
from pathlib import Path

import requests
from PIL import Image

import config
from error_handler import RetryHandler, ErrorType
from monitoring import trace_function

logger = logging.getLogger(__name__)


@dataclass
class VisionResult:
    """Result from a vision model analysis."""
    description: str
    model: str = ""
    processing_time_ms: float = 0.0
    image_size: tuple = (0, 0)
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


class VisionHandler:
    """
    Handles image processing and analysis using LLaVA via Ollama.

    Uses the Ollama /api/generate endpoint with base64 image payloads
    for reliable multimodal inference.
    """

    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        max_tokens: int = None,
    ):
        self.model = model or config.VISION_MODEL
        self.base_url = (base_url or config.VISION_BASE_URL).rstrip("/")
        self.max_tokens = max_tokens or config.VISION_MAX_TOKENS
        self._retry_handler = RetryHandler(max_retries=2, base_delay=1.0)

        logger.info(f"VisionHandler initialized: model={self.model}, url={self.base_url}")

    def load_image(self, source: Any) -> Image.Image:
        """
        Load an image from various sources.

        Args:
            source: File path (str/Path), bytes, BytesIO, or Streamlit UploadedFile

        Returns:
            PIL Image object

        Raises:
            ValueError: If source type is unsupported or image can't be loaded
        """
        try:
            if isinstance(source, (str, Path)):
                path = Path(source)
                if not path.exists():
                    raise FileNotFoundError(f"Image file not found: {path}")
                suffix = path.suffix.lower().lstrip(".")
                if suffix not in config.VISION_SUPPORTED_FORMATS:
                    raise ValueError(f"Unsupported image format: {suffix}")
                return Image.open(path)

            elif isinstance(source, bytes):
                return Image.open(io.BytesIO(source))

            elif isinstance(source, io.BytesIO):
                source.seek(0)
                return Image.open(source)

            elif hasattr(source, "read") and hasattr(source, "name"):
                # Streamlit UploadedFile
                source.seek(0)
                return Image.open(source)

            else:
                raise ValueError(f"Unsupported image source type: {type(source)}")

        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            raise ValueError(f"Failed to load image: {e}")

    def image_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """Convert a PIL Image to base64 string for the Ollama API."""
        buffer = io.BytesIO()
        # Convert to RGB if needed (handles RGBA, P, etc.)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def preprocess_image(self, image: Image.Image, max_dimension: int = 1024) -> Image.Image:
        """
        Preprocess image for vision model: resize and convert to RGB.

        Args:
            image: PIL Image to preprocess
            max_dimension: Maximum width or height

        Returns:
            Preprocessed PIL Image
        """
        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize if larger than max_dimension
        w, h = image.size
        if max(w, h) > max_dimension:
            scale = max_dimension / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)
            logger.debug(f"Image resized from {w}x{h} to {new_w}x{new_h}")

        return image

    @trace_function("vision_analyze")
    def analyze_image(
        self,
        image: Any,
        prompt: str = "Describe this image in detail.",
        temperature: float = 0.3,
    ) -> VisionResult:
        """
        Analyze an image using the vision model.

        Args:
            image: Image source (path, bytes, BytesIO, UploadedFile, or PIL Image)
            prompt: Text prompt for the analysis
            temperature: Model temperature

        Returns:
            VisionResult with description and metadata
        """
        start_time = time.time()
        try:
            # Load image if not already a PIL Image
            if not isinstance(image, Image.Image):
                image = self.load_image(image)

            original_size = image.size

            # Preprocess
            image = self.preprocess_image(image)

            # Convert to base64
            image_b64 = self.image_to_base64(image)

            # Call Ollama vision API
            response_text = self._call_ollama_vision(
                prompt=prompt,
                images_b64=[image_b64],
                temperature=temperature,
            )

            processing_time = (time.time() - start_time) * 1000

            return VisionResult(
                description=response_text,
                model=self.model,
                processing_time_ms=processing_time,
                image_size=original_size,
                metadata={
                    "prompt": prompt,
                    "preprocessed_size": image.size,
                },
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Vision analysis failed: {e}")
            return VisionResult(
                description="",
                model=self.model,
                processing_time_ms=processing_time,
                error=str(e),
            )

    def ask_about_image(self, image: Any, question: str) -> VisionResult:
        """
        Ask a specific question about an image.

        Args:
            image: Image source
            question: Question about the image

        Returns:
            VisionResult with the answer
        """
        prompt = f"Look at this image and answer the following question: {question}"
        return self.analyze_image(image, prompt=prompt)

    def compare_images(self, image1: Any, image2: Any) -> VisionResult:
        """
        Compare two images and describe the differences.

        Args:
            image1: First image source
            image2: Second image source

        Returns:
            VisionResult with comparison description
        """
        start_time = time.time()
        try:
            # Load and preprocess both images
            if not isinstance(image1, Image.Image):
                image1 = self.load_image(image1)
            if not isinstance(image2, Image.Image):
                image2 = self.load_image(image2)

            image1 = self.preprocess_image(image1)
            image2 = self.preprocess_image(image2)

            images_b64 = [
                self.image_to_base64(image1),
                self.image_to_base64(image2),
            ]

            prompt = (
                "Compare these two images. Describe the key similarities "
                "and differences between them."
            )

            response_text = self._call_ollama_vision(
                prompt=prompt,
                images_b64=images_b64,
                temperature=0.3,
            )

            processing_time = (time.time() - start_time) * 1000

            return VisionResult(
                description=response_text,
                model=self.model,
                processing_time_ms=processing_time,
                metadata={"comparison": True},
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Image comparison failed: {e}")
            return VisionResult(
                description="",
                model=self.model,
                processing_time_ms=processing_time,
                error=str(e),
            )

    def check_model_available(self) -> bool:
        """Check if the vision model is available in Ollama."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                available = any(self.model in name for name in model_names)
                if not available:
                    logger.warning(
                        f"Vision model '{self.model}' not found. "
                        f"Available: {model_names}. Run: ollama pull {self.model}"
                    )
                return available
            return False
        except Exception as e:
            logger.error(f"Failed to check Ollama models: {e}")
            return False

    def _call_ollama_vision(
        self,
        prompt: str,
        images_b64: List[str],
        temperature: float = 0.3,
    ) -> str:
        """
        Internal method to call the Ollama vision API with retry.

        Args:
            prompt: Text prompt
            images_b64: List of base64-encoded images
            temperature: Model temperature

        Returns:
            Model response text

        Raises:
            RuntimeError: If all retries fail
        """
        def _do_request():
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": images_b64,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": self.max_tokens,
                },
            }

            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120,
            )

            if resp.status_code != 200:
                raise RuntimeError(
                    f"Ollama vision API error {resp.status_code}: {resp.text}"
                )

            data = resp.json()
            return data.get("response", "").strip()

        return self._retry_handler.with_retry(
            _do_request,
            retryable_exceptions=(requests.RequestException, RuntimeError),
        )
