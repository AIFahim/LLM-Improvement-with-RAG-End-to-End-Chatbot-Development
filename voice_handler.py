"""
Voice Handler Module - Class 11: Multimodal Agents

Speech-to-Text (STT) using OpenAI Whisper and
Text-to-Speech (TTS) using gTTS / pyttsx3 / Orpheus (via Ollama).
"""

import io
import os
import time
import logging
import tempfile
from typing import Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

import config
from error_handler import RetryHandler
from monitoring import trace_function

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result from speech-to-text transcription."""
    text: str
    language: str = ""
    confidence: float = 0.0
    duration_seconds: float = 0.0
    engine: str = ""
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.text.strip())


@dataclass
class TTSResult:
    """Result from text-to-speech synthesis."""
    audio_bytes: bytes = b""
    format: str = "mp3"
    duration_seconds: float = 0.0
    engine: str = ""
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and len(self.audio_bytes) > 0


class VoiceHandler:
    """
    Handles speech-to-text and text-to-speech operations.

    STT Backends:
        - whisper: OpenAI Whisper (local, high quality)
        - google: SpeechRecognition Google Web Speech API (fallback)

    TTS Backends:
        - gtts: Google Text-to-Speech (requires internet)
        - pyttsx3: Offline TTS (fallback)
        - orpheus: Orpheus TTS via Ollama (local, natural speech with emotions)
    """

    def __init__(
        self,
        stt_engine: str = None,
        tts_engine: str = None,
        whisper_model_size: str = None,
        language: str = None,
    ):
        self.stt_engine = stt_engine or config.STT_ENGINE
        self.tts_engine = tts_engine or config.TTS_ENGINE
        self.whisper_model_size = whisper_model_size or config.WHISPER_MODEL_SIZE
        self.language = language or config.TTS_LANGUAGE
        self._whisper_model = None
        self._retry_handler = RetryHandler(max_retries=1, base_delay=0.5)

        logger.info(
            f"VoiceHandler initialized: stt={self.stt_engine}, "
            f"tts={self.tts_engine}, whisper_size={self.whisper_model_size}"
        )

    def _load_whisper_model(self):
        """Lazy-load the Whisper model."""
        if self._whisper_model is None:
            try:
                import whisper
                logger.info(f"Loading Whisper model: {self.whisper_model_size}")
                self._whisper_model = whisper.load_model(
                    self.whisper_model_size,
                    device=config.WHISPER_DEVICE,
                )
                logger.info("Whisper model loaded successfully")
            except ImportError:
                raise ImportError(
                    "openai-whisper is required for STT. "
                    "Install with: pip install openai-whisper"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load Whisper model: {e}")
        return self._whisper_model

    @trace_function("voice_transcribe")
    def transcribe(
        self,
        audio_source: Any,
        language: str = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio_source: File path (str/Path), bytes, BytesIO, or Streamlit audio
            language: Language code (e.g., 'en'). None for auto-detection.

        Returns:
            TranscriptionResult with transcription text and metadata
        """
        start_time = time.time()
        language = language or self.language

        try:
            # Save audio to a temp file if needed
            audio_path = self._prepare_audio_file(audio_source)

            # Validate audio
            self.validate_audio(audio_path)

            # Transcribe using selected engine
            if self.stt_engine == "whisper":
                result = self._transcribe_whisper(audio_path, language)
            else:
                result = self._transcribe_google(audio_path, language)

            result.duration_seconds = time.time() - start_time
            return result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return TranscriptionResult(
                text="",
                language=language or "",
                engine=self.stt_engine,
                duration_seconds=time.time() - start_time,
                error=str(e),
            )

    def _prepare_audio_file(self, audio_source: Any) -> str:
        """Convert various audio sources to a temp file path."""
        if isinstance(audio_source, (str, Path)):
            path = str(audio_source)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Audio file not found: {path}")
            return path

        # For bytes or file-like objects, write to temp file
        audio_bytes = None
        if isinstance(audio_source, bytes):
            audio_bytes = audio_source
        elif isinstance(audio_source, io.BytesIO):
            audio_source.seek(0)
            audio_bytes = audio_source.read()
        elif hasattr(audio_source, "read"):
            # Streamlit UploadedFile or similar
            audio_source.seek(0)
            audio_bytes = audio_source.read()

        if audio_bytes is None:
            raise ValueError(f"Unsupported audio source type: {type(audio_source)}")

        # Write to temp file
        temp_path = config.TEMP_MEDIA_DIR / f"audio_input_{int(time.time())}.wav"
        temp_path.write_bytes(audio_bytes)
        return str(temp_path)

    def _transcribe_whisper(self, audio_path: str, language: str) -> TranscriptionResult:
        """Transcribe using OpenAI Whisper."""
        model = self._load_whisper_model()

        options = {"fp16": False}
        if language:
            options["language"] = language

        result = model.transcribe(audio_path, **options)

        text = result.get("text", "").strip()
        detected_lang = result.get("language", language or "")

        # Estimate confidence from segments
        segments = result.get("segments", [])
        confidence = 0.0
        if segments:
            avg_logprob = sum(s.get("avg_logprob", -1.0) for s in segments) / len(segments)
            # Convert log probability to approximate confidence (0-1)
            import math
            confidence = min(1.0, max(0.0, math.exp(avg_logprob)))

        return TranscriptionResult(
            text=text,
            language=detected_lang,
            confidence=confidence,
            engine="whisper",
        )

    def _transcribe_google(self, audio_path: str, language: str) -> TranscriptionResult:
        """Transcribe using Google Web Speech API via SpeechRecognition."""
        try:
            import speech_recognition as sr
        except ImportError:
            raise ImportError(
                "SpeechRecognition is required for Google STT. "
                "Install with: pip install SpeechRecognition"
            )

        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)

        lang_code = language if language else "en-US"
        if len(lang_code) == 2:
            lang_code = f"{lang_code}-{lang_code.upper()}"

        text = recognizer.recognize_google(audio_data, language=lang_code)

        return TranscriptionResult(
            text=text,
            language=language or "en",
            confidence=0.8,  # Google doesn't provide confidence
            engine="google",
        )

    @trace_function("voice_synthesize")
    def synthesize(
        self,
        text: str,
        output_format: str = "mp3",
    ) -> TTSResult:
        """
        Convert text to speech audio.

        Args:
            text: Text to convert to speech
            output_format: Output audio format (mp3, wav)

        Returns:
            TTSResult with audio bytes
        """
        start_time = time.time()

        if not text or not text.strip():
            return TTSResult(
                engine=self.tts_engine,
                error="Empty text provided",
            )

        try:
            if self.tts_engine == "orpheus":
                result = self._synthesize_orpheus(text)
            elif self.tts_engine == "gtts":
                result = self._synthesize_gtts(text)
            else:
                result = self._synthesize_pyttsx3(text)

            result.duration_seconds = time.time() - start_time
            result.format = output_format
            return result

        except Exception as e:
            logger.error(f"TTS synthesis failed with {self.tts_engine}: {e}")
            # Fallback chain: orpheus -> gtts -> pyttsx3
            fallbacks = ["gtts", "pyttsx3"]
            if self.tts_engine == "gtts":
                fallbacks = ["pyttsx3"]
            elif self.tts_engine == "pyttsx3":
                fallbacks = []

            for fallback in fallbacks:
                try:
                    logger.info(f"Falling back to {fallback} for TTS")
                    if fallback == "gtts":
                        result = self._synthesize_gtts(text)
                    else:
                        result = self._synthesize_pyttsx3(text)
                    result.duration_seconds = time.time() - start_time
                    return result
                except Exception:
                    continue

            return TTSResult(
                engine=self.tts_engine,
                duration_seconds=time.time() - start_time,
                error=str(e),
            )

    def _synthesize_gtts(self, text: str) -> TTSResult:
        """Synthesize speech using gTTS."""
        try:
            from gtts import gTTS
        except ImportError:
            raise ImportError(
                "gTTS is required for Google TTS. Install with: pip install gTTS"
            )

        tts = gTTS(text=text, lang=self.language)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)

        return TTSResult(
            audio_bytes=buffer.read(),
            format="mp3",
            engine="gtts",
        )

    def _synthesize_pyttsx3(self, text: str) -> TTSResult:
        """Synthesize speech using pyttsx3 (offline)."""
        try:
            import pyttsx3
        except ImportError:
            raise ImportError(
                "pyttsx3 is required for offline TTS. "
                "Install with: pip install pyttsx3"
            )

        engine = pyttsx3.init()

        # Save to temp file then read bytes
        temp_path = str(config.TEMP_MEDIA_DIR / f"tts_output_{int(time.time())}.wav")
        engine.save_to_file(text, temp_path)
        engine.runAndWait()

        audio_bytes = Path(temp_path).read_bytes()

        # Clean up
        try:
            os.remove(temp_path)
        except OSError:
            pass

        return TTSResult(
            audio_bytes=audio_bytes,
            format="wav",
            engine="pyttsx3",
        )

    def _synthesize_orpheus(self, text: str) -> TTSResult:
        """
        Synthesize speech using Orpheus TTS via Ollama.

        Orpheus outputs audio token IDs which are decoded to audio
        using the SNAC neural audio codec.

        Requires: pip install snac torch
        Model: ollama pull legraphista/Orpheus
        """
        try:
            import torch
            from snac import SNAC
        except ImportError:
            raise ImportError(
                "snac and torch are required for Orpheus TTS. "
                "Install with: pip install snac torch"
            )

        import requests
        import struct
        import numpy as np

        voice = config.ORPHEUS_VOICE
        model = config.ORPHEUS_MODEL
        base_url = config.ORPHEUS_BASE_URL.rstrip("/")

        # Orpheus prompt format: special tokens + voice tag + text
        prompt = f"<|audio|>{voice}: {text}<|eot_id|>"

        logger.info(f"Orpheus TTS: voice={voice}, model={model}, text={text[:60]}...")

        # Call Ollama API to generate audio tokens
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "num_predict": 4096,
                    "num_ctx": 2048,
                },
            },
            timeout=180,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"Ollama Orpheus API error {resp.status_code}: {resp.text}")

        response_text = resp.json().get("response", "")

        # Parse audio token IDs from the response
        token_ids = self._parse_orpheus_tokens(response_text)

        if not token_ids:
            raise RuntimeError("Orpheus generated no audio tokens")

        logger.info(f"Orpheus generated {len(token_ids)} audio tokens")

        # Decode tokens to audio using SNAC
        audio_bytes = self._decode_snac_tokens(token_ids, torch, SNAC)

        return TTSResult(
            audio_bytes=audio_bytes,
            format="wav",
            engine="orpheus",
        )

    def _parse_orpheus_tokens(self, response_text: str) -> list:
        """
        Parse Orpheus response into raw audio token IDs.

        Orpheus outputs custom audio tokens as '<custom_token_N>'.
        The first few low-value tokens (< 10) are control/start tokens
        and are skipped.
        """
        import re

        token_ids = []
        matches = re.findall(r"custom_token_(\d+)", response_text)
        for m in matches:
            token_ids.append(int(m))

        # Skip leading control tokens (typically IDs < 10 at the start)
        start_idx = 0
        for i, tid in enumerate(token_ids):
            if tid >= 10:
                start_idx = i
                break

        return token_ids[start_idx:]

    def _decode_snac_tokens(self, token_ids: list, torch, SNAC) -> bytes:
        """
        Decode Orpheus audio token IDs into WAV audio bytes.

        Orpheus uses SNAC with 3 codebook layers (4096 entries each).
        Tokens are flattened in groups of 7 per frame:
          [L0, L1, L2, L2, L1, L2, L2]

        The raw token IDs from Ollama map to SNAC layers as:
          Layer 0 (coarse): token_id % 4096  (IDs in range 0-4095)
          Layer 1 (mid):    (token_id - 4096) % 4096  (IDs ~4096-8191)
          Layer 2 (fine):   (token_id - 8192) % 4096  (IDs ~8192+)

        But since Orpheus interleaves all layers into a flat sequence
        with a known pattern, we redistribute by position.

        Returns WAV file bytes at 24kHz mono 16-bit.
        """
        import struct
        import numpy as np

        device = config.WHISPER_DEVICE

        # Load SNAC model (24kHz)
        snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        if device == "cuda" and torch.cuda.is_available():
            snac_model = snac_model.cuda()

        # Redistribute flat token list into 3 SNAC codebook layers
        # Pattern per frame: [L0, L1, L2, L2, L1, L2, L2] (7 tokens)
        # Each layer has 4096 codebook entries
        codes_0, codes_1, codes_2 = [], [], []

        i = 0
        while i + 6 < len(token_ids):
            # Extract and remap to codebook range (mod 4096)
            codes_0.append(token_ids[i] % 4096)
            codes_1.append(token_ids[i + 1] % 4096)
            codes_2.append(token_ids[i + 2] % 4096)
            codes_2.append(token_ids[i + 3] % 4096)
            codes_1.append(token_ids[i + 4] % 4096)
            codes_2.append(token_ids[i + 5] % 4096)
            codes_2.append(token_ids[i + 6] % 4096)
            i += 7

        if not codes_0:
            raise RuntimeError("Not enough tokens for SNAC decoding (need at least 7)")

        logger.info(
            f"SNAC decode: {len(codes_0)} frames, "
            f"L0={len(codes_0)}, L1={len(codes_1)}, L2={len(codes_2)} codes"
        )

        # Convert to tensors
        with torch.no_grad():
            c0 = torch.tensor([codes_0], dtype=torch.long)
            c1 = torch.tensor([codes_1], dtype=torch.long)
            c2 = torch.tensor([codes_2], dtype=torch.long)

            if device == "cuda" and torch.cuda.is_available():
                c0, c1, c2 = c0.cuda(), c1.cuda(), c2.cuda()

            # Decode with SNAC
            audio = snac_model.decode([c0, c1, c2])

            # Convert to numpy
            audio_np = audio.squeeze().cpu().numpy()

        # Normalize to int16 range
        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_int16 = (audio_np * 32767).astype(np.int16)

        # Write WAV bytes
        sample_rate = config.ORPHEUS_SAMPLE_RATE
        wav_buffer = io.BytesIO()
        num_samples = len(audio_int16)
        data_size = num_samples * 2  # 16-bit = 2 bytes per sample
        wav_buffer.write(b"RIFF")
        wav_buffer.write(struct.pack("<I", 36 + data_size))
        wav_buffer.write(b"WAVE")
        wav_buffer.write(b"fmt ")
        wav_buffer.write(struct.pack("<I", 16))       # chunk size
        wav_buffer.write(struct.pack("<H", 1))         # PCM format
        wav_buffer.write(struct.pack("<H", 1))         # mono
        wav_buffer.write(struct.pack("<I", sample_rate))
        wav_buffer.write(struct.pack("<I", sample_rate * 2))  # byte rate
        wav_buffer.write(struct.pack("<H", 2))         # block align
        wav_buffer.write(struct.pack("<H", 16))        # bits per sample
        wav_buffer.write(b"data")
        wav_buffer.write(struct.pack("<I", data_size))
        wav_buffer.write(audio_int16.tobytes())

        wav_buffer.seek(0)
        return wav_buffer.read()

    def save_audio_bytes(self, audio_bytes: bytes, filename: str = None) -> str:
        """
        Save audio bytes to the temp_media directory.

        Args:
            audio_bytes: Raw audio data
            filename: Optional filename (auto-generated if None)

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"audio_{int(time.time())}.mp3"

        filepath = config.TEMP_MEDIA_DIR / filename
        filepath.write_bytes(audio_bytes)
        return str(filepath)

    def validate_audio(self, audio_source: Any) -> bool:
        """
        Validate an audio file for format and size.

        Args:
            audio_source: File path string

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        if isinstance(audio_source, str):
            path = Path(audio_source)
            if not path.exists():
                raise ValueError(f"Audio file not found: {path}")

            # Check file size
            size_mb = path.stat().st_size / (1024 * 1024)
            max_size = config.VISION_MAX_IMAGE_SIZE_MB  # reuse for audio
            if size_mb > max_size * 5:  # Allow up to 50MB for audio
                raise ValueError(f"Audio file too large: {size_mb:.1f}MB (max {max_size * 5}MB)")

        return True

    def cleanup_temp_files(self) -> int:
        """
        Clean up temporary media files.

        Returns:
            Number of files cleaned up
        """
        count = 0
        if config.TEMP_MEDIA_DIR.exists():
            for f in config.TEMP_MEDIA_DIR.iterdir():
                if f.is_file():
                    try:
                        f.unlink()
                        count += 1
                    except OSError:
                        pass
        logger.info(f"Cleaned up {count} temp media files")
        return count
