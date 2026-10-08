"""Voice Assistant: Speech -> Text -> RAG Search -> Response -> Speech.

Speech-to-text (first available):
  1. Groq Whisper (``GROQ_API_KEY``, free) / OpenAI transcription API (``OPENAI_API_KEY``)
  2. Gemini audio understanding (``GEMINI_API_KEY``)
  3. ``SpeechRecognition`` package (free Google Web Speech endpoint)

Text-to-speech (first available):
  1. ``gTTS`` (free, needs internet) -> MP3 bytes
  2. OpenAI TTS API -> MP3 bytes
  3. Browser speech synthesis (the Streamlit UI falls back to the Web Speech API, no server needed)
"""
from __future__ import annotations

import base64
import io
import re
from typing import Optional, Tuple

import requests

from .config import Settings


class VoiceError(RuntimeError):
    pass


def _guess_mime(audio: bytes) -> str:
    if audio[:4] == b"RIFF":
        return "audio/wav"
    if audio[:3] == b"ID3" or audio[:2] == b"\xff\xfb":
        return "audio/mp3"
    if audio[:4] == b"OggS":
        return "audio/ogg"
    if audio[4:8] == b"ftyp":
        return "audio/mp4"
    return "audio/webm"


def speech_to_text(audio: bytes, settings: Settings) -> Tuple[str, str]:
    """Return (transcript, engine_used)."""
    if not audio:
        raise VoiceError("Empty audio")
    errors = []
    if settings.groq_api_key:
        try:
            mime = _guess_mime(audio)
            r = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                files={"file": (f"speech.{mime.split('/')[1]}", audio, mime)},
                data={"model": "whisper-large-v3-turbo"},
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["text"].strip(), "groq-whisper"
        except Exception as e:  # noqa: BLE001
            errors.append(f"groq: {e}")
    if settings.openai_api_key:
        try:
            mime = _guess_mime(audio)
            ext = mime.split("/")[1].replace("mp3", "mp3")
            r = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                files={"file": (f"speech.{ext}", audio, mime)},
                data={"model": "whisper-1"},
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["text"].strip(), "openai-whisper"
        except Exception as e:  # noqa: BLE001
            errors.append(f"openai: {e}")
    if settings.gemini_api_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.gemini_model}:generateContent"
            body = {"contents": [{"parts": [
                {"text": "Transcribe this speech exactly. Return only the transcript text."},
                {"inline_data": {"mime_type": _guess_mime(audio), "data": base64.b64encode(audio).decode()}},
            ]}]}
            r = requests.post(url, headers={"x-goog-api-key": settings.gemini_api_key}, json=body, timeout=120)
            r.raise_for_status()
            text = "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])
            return text.strip(), "gemini"
        except Exception as e:  # noqa: BLE001
            errors.append(f"gemini: {e}")
    try:
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio)) as src:  # needs WAV/AIFF/FLAC
            data = recognizer.record(src)
        return recognizer.recognize_google(data), "speech_recognition"
    except ImportError:
        errors.append("SpeechRecognition not installed")
    except Exception as e:  # noqa: BLE001
        errors.append(f"speech_recognition: {e}")
    raise VoiceError("Speech-to-text unavailable. Set GEMINI_API_KEY or OPENAI_API_KEY, or `pip install "
                     "SpeechRecognition`. Details: " + "; ".join(errors))


def clean_for_speech(text: str, max_chars: int = 2500) -> str:
    text = re.sub(r"<details>.*?</details>", "", text, flags=re.S)
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)  # citations
    text = re.sub(r"[#*_`>|]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def text_to_speech(text: str, settings: Settings, lang: str = "en") -> Optional[bytes]:
    """Return MP3 bytes, or None when only browser-side synthesis is available."""
    text = clean_for_speech(text)
    if not text:
        return None
    try:
        from gtts import gTTS

        buf = io.BytesIO()
        gTTS(text=text, lang=lang).write_to_fp(buf)
        return buf.getvalue()
    except Exception:  # noqa: BLE001 - not installed or offline
        pass
    if settings.openai_api_key:
        try:
            r = requests.post(
                "https://api.openai.com/v1/audio/speech",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                json={"model": "tts-1", "voice": "alloy", "input": text},
                timeout=120,
            )
            r.raise_for_status()
            return r.content
        except Exception:  # noqa: BLE001
            pass
    return None


def browser_tts_html(text: str) -> str:
    """HTML snippet that reads ``text`` aloud with the browser's built-in speech synthesis."""
    import json

    safe = json.dumps(clean_for_speech(text))
    return f"""
<button id="speak" style="padding:6px 14px;border-radius:8px;border:1px solid #888;cursor:pointer;background:transparent;color:inherit">
🔊 Read aloud</button>
<button id="stop" style="padding:6px 14px;border-radius:8px;border:1px solid #888;cursor:pointer;background:transparent;color:inherit">
⏹ Stop</button>
<script>
const t = {safe};
document.getElementById('speak').onclick = () => {{ speechSynthesis.cancel(); speechSynthesis.speak(new SpeechSynthesisUtterance(t)); }};
document.getElementById('stop').onclick = () => speechSynthesis.cancel();
</script>"""
