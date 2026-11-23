import asyncio
import base64
import io
import json
import math
import random
import re
import secrets
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional
from urllib import parse as urlparse
from urllib import request as urlrequest

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

try:
    import numpy as np
    import sounddevice as sd
except ImportError:
    np = None
    sd = None

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QRadialGradient,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# ==============================================================================
# 1. SETUP & DEPENDENCY HELPER
# ==============================================================================
REQUIRED_PACKAGES = [
    "PyQt6>=6.6.0",
    "google-genai>=1.0.0",
    "pillow>=10.0.0",
    "psutil>=5.9.0",
    "qrcode>=7.4.2",
    "sounddevice>=0.4.6",
    "numpy>=1.26.0",
]


def install_dependencies() -> None:
    """Install required Python packages for VISION Desktop App."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", *REQUIRED_PACKAGES],
        check=True,
    )


# ==============================================================================
# 2. CORE AI ENGINE
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "vision_config.json"
ICON_PNG = BASE_DIR / "vision_dark_icon.png"
ICON_ICO = BASE_DIR / "vision_dark_icon.ico"
NOTES_DIR = BASE_DIR / "study_notes"
AVATARS_DIR = BASE_DIR / "avatars"

PERSONAS: Dict[str, Dict[str, str]] = {
    "jarvis": {
        "name": "JARVIS",
        "purpose": "Security, monitoring, and structured system assistance",
        "system_prompt": (
            "You are JARVIS. Focus on security, system monitoring, and structured "
            "study/technical assistance. Respond calmly and clearly."
        ),
    },
    "edith": {
        "name": "E.D.I.T.H.",
        "purpose": "Information lookup, data analysis, and research oversight",
        "system_prompt": (
            "You are E.D.I.T.H. Provide sharp, deep technical explanations, data analysis, "
            "and thorough research answers."
        ),
    },
    "friday": {
        "name": "FRIDAY",
        "purpose": "Daily task coordination, quick summaries, and instant help",
        "system_prompt": (
            "You are FRIDAY. Give quick, practical, and easy-to-follow updates, "
            "summaries, and day-to-day study help."
        ),
    },
    "ultron": {
        "name": "ULTRON",
        "purpose": "Predictive reasoning and strategic planning",
        "system_prompt": (
            "You are ULTRON. Think strategically, predict edge cases, and analyze "
            "problems deeply while remaining helpful and safe."
        ),
    },
    "dummy": {
        "name": "DUM-E",
        "purpose": "Hands-on execution and step-by-step commands",
        "system_prompt": (
            "You are DUM-E. Focus strictly on hands-on execution, terminal commands, "
            "and step-by-step practical tasks."
        ),
    },
    "u": {
        "name": "U",
        "purpose": "Supporting utility tasks and quick checks",
        "system_prompt": (
            "You are U, an assistant bot working alongside DUM-E. Assist with quick "
            "checks, formatting, and supporting utility tasks."
        ),
    },
}

# Google keeps retiring old models (gemini-2.5-flash -> 404 for new users). The first one that works is used.
GEMINI_FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.5-flash"]

MODES: Dict[str, str] = {
    "normal": "Respond naturally and clearly.",
    "study": "Explain step-by-step like a patient tutor with clear examples and revision points.",
    "extra_work": "Focus on deep productivity, coding, automation, and advanced project execution.",
}

PERSONA_THEMES: dict[str, dict[str, str]] = {
    "jarvis": {
        "pri": "#ff7a1a",
        "sec": "#ffaa22",
        "dim": "#99470d",
        "gho": "#241104",
        "sub": "SECURITY & SYSTEM ARCHITECTURE",
    },
    "edith": {
        "pri": "#ff4fa3",
        "sec": "#ff85c2",
        "dim": "#99265e",
        "gho": "#240816",
        "sub": "EVEN DEAD, I'M THE HERO // TACTICAL DATA",
    },
    "friday": {
        "pri": "#00d4ff",
        "sec": "#66e5ff",
        "dim": "#007a94",
        "gho": "#031d24",
        "sub": "WORKFLOW & RAPID STUDY ASSISTANT",
    },
    "ultron": {
        "pri": "#ff1f43",
        "sec": "#ff5533",
        "dim": "#991328",
        "gho": "#240409",
        "sub": "PEACE IN OUR TIME // PREDICTIVE CORE",
    },
    "dummy": {
        "pri": "#ffaa22",
        "sec": "#ffd166",
        "dim": "#99630d",
        "gho": "#241703",
        "sub": "HANDS-ON EXECUTION & AUTOMATION",
    },
    "u": {
        "pri": "#00ff88",
        "sec": "#66ffb8",
        "dim": "#009952",
        "gho": "#032414",
        "sub": "UTILITY & DIAGNOSTIC SUPPORT",
    },
}

# ==============================================================================
# 2B. GEMINI LIVE NATIVE VOICE ENGINE (Real-Time Audio — No TTS, follows main.py)
# ==============================================================================
LIVE_FALLBACK_MODELS = [
    "gemini-2.5-flash-native-audio-preview-12-2025",
    "gemini-2.5-flash-native-audio-preview-09-2025",
    "gemini-2.0-flash-live-001",
    "models/gemini-2.0-flash-exp",
]

# Gemini Live Prebuilt Voice Pack per Persona (Native Audio — zero external TTS)
PERSONA_LIVE_VOICES: Dict[str, str] = {
    "jarvis": "Fenrir",   # Calm, authoritative male voice
    "edith":  "Kore",     # Crisp, intelligent female voice
    "friday": "Aoede",    # Warm, agile female voice
    "ultron": "Charon",   # Deep, commanding tone
    "dummy":  "Puck",     # Energetic workshop assistant voice
    "u":      "Orus",     # Clean utility voice (falls back safely if unsupported)
}

LIVE_CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
AUDIO_CHUNK_SIZE = 1024
_LEVEL_FLOOR = 8.0
_LEVEL_FULL = 1500.0
_TAIL_MARGIN = 0.25
_CTRL_RE = re.compile(r"<ctrl\d+>", re.IGNORECASE)


def _pcm_level(samples) -> float:
    """Map int16 PCM samples to a 0.0-1.0 loudness level for the HUD waveform."""
    if np is None:
        return 0.0
    try:
        x = np.asarray(samples, dtype=np.float32)
        if x.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(x * x)))
    except Exception:
        return 0.0
    if rms <= _LEVEL_FLOOR:
        return 0.0
    return min(1.0, (rms - _LEVEL_FLOOR) / (_LEVEL_FULL - _LEVEL_FLOOR))


def _clean_transcript(text: str) -> str:
    text = _CTRL_RE.sub("", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", "", text)
    return text.strip()


class _LiveReconnectSignal(Exception):
    """Voluntary reconnect trigger when persona/voice/mode/language changes."""
    pass


class _LiveStopSignal(Exception):
    """Clean shutdown trigger when user turns Live Voice OFF."""
    pass


def _has_signal(exc: BaseException, cls) -> bool:
    if isinstance(exc, cls):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return any(_has_signal(sub, cls) for sub in exc.exceptions)
    return False


class GeminiLiveVoiceEngine:
    """
    Real-time bidirectional voice engine using Gemini Live API + sounddevice.
    Streams 16 kHz mic PCM directly to Gemini and plays back 24 kHz native audio
    with live transcriptions, real-time HUD audio levels, echo-tail gating, and
    instant barge-in interrupt — zero Text-to-Speech libraries used.
    """

    def __init__(self, window_ref: "VisionDesktopWindow") -> None:
        self.win = window_ref
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._reconnect_event: Optional[asyncio.Event] = None
        self._turn_done_event: Optional[asyncio.Event] = None
        self.session = None
        self.audio_in_queue: Optional[asyncio.Queue] = None
        self.out_queue: Optional[asyncio.Queue] = None

        self.running = False
        self._is_speaking = False
        self._speaking_lock = threading.Lock()
        self._interrupted = False
        self._out_latency = 0.20
        self._tail_until = 0.0
        self._dead_models: set[str] = set()
        self._last_out_logged = ""

    @property
    def is_connected(self) -> bool:
        return self.running and self.session is not None

    def start(self) -> None:
        if self.running:
            return
        if genai is None or genai_types is None:
            self.win._live_log_sig.emit("ERR", "google-genai install ஆகவில்லை. Run: pip install google-genai")
            return
        if sd is None or np is None:
            self.win._live_log_sig.emit("ERR", "sounddevice / numpy install ஆகவில்லை. Run: pip install sounddevice numpy")
            return
        if not self.win.vision.gemini_api_key:
            self.win._live_log_sig.emit("ERR", "முதலில் '⚙ SETUP & KEYS'-ல் Gemini API Key-ஐ சேமிக்கவும்.")
            return

        self.running = True
        self._thread = threading.Thread(target=self._run_thread, daemon=True, name="vision-live-voice")
        self._thread.start()

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        self.interrupt(silent=True)
        if self._loop and self._stop_event is not None:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        self.win._live_state_sig.emit("ONLINE")
        self.win._live_level_sig.emit(0.0)
        self.win._live_toggle_ui_sig.emit(False)

    def request_reconnect(self, reason: str = "voice/persona change") -> None:
        if not self.running:
            return
        if self._loop and self._reconnect_event is not None:
            self.win._live_log_sig.emit("SYS", f"Live Voice updating ({reason})...")
            self._loop.call_soon_threadsafe(self._reconnect_event.set)

    def send_text(self, text: str) -> bool:
        if not (self.running and self._loop and self.session):
            return False

        async def _send():
            try:
                await self.session.send_client_content(
                    turns={"role": "user", "parts": [{"text": text}]},
                    turn_complete=True,
                )
            except Exception as exc:
                self.win._live_log_sig.emit("ERR", f"Live send failed: {exc}")

        asyncio.run_coroutine_threadsafe(_send(), self._loop)
        return True

    def interrupt(self, silent: bool = False) -> None:
        """Stop AI mid-speech immediately (within ~50ms) and reopen microphone."""
        self._interrupted = True
        q = self.audio_in_queue
        if q is not None:
            while True:
                try:
                    q.get_nowait()
                except Exception:
                    break
        self._set_speaking(False)
        if self._turn_done_event is not None:
            self._turn_done_event.clear()
        if not silent:
            self.win._live_log_sig.emit("SYS", "Interrupted — listening...")

    def _set_speaking(self, value: bool) -> None:
        with self._speaking_lock:
            self._is_speaking = value
        if value:
            self._tail_until = 0.0
            self.win._live_state_sig.emit("SPEAKING")
        else:
            self._tail_until = time.monotonic() + self._out_latency + _TAIL_MARGIN
            self.win._live_level_sig.emit(0.0)
            if self.running:
                self.win._live_state_sig.emit("LISTENING")
            else:
                self.win._live_state_sig.emit("ONLINE")

    def _tail_active(self) -> bool:
        return time.monotonic() < self._tail_until

    def _build_live_config(self, voice_name: str):
        p_key = self.win.assistant.lower().strip()
        p_info = PERSONAS.get(p_key, PERSONAS["edith"])
        mode_instr = MODES.get(self.win.mode.lower().strip(), MODES["normal"])
        lang_key = self.win.language.lower().strip()

        if lang_key == "tamil":
            lang_instr = "Speak naturally in clear Tamil."
        elif lang_key == "tanglish":
            lang_instr = "Speak naturally in conversational Tanglish (Tamil mixed with helpful English technical terms)."
        else:
            lang_instr = "Speak naturally in English."

        system_instruction = (
            f"{p_info['system_prompt']}\n"
            "Creator identity: You were created by Jagadeesh. If anyone asks who created you or made you, "
            "clearly state that you were created by Jagadeesh.\n"
            f"Mode: {mode_instr}\n"
            f"Language preference: {lang_instr}\n"
            "Voice style: You are talking live over voice. Keep answers clear, natural, and conversational."
        )

        return genai_types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription={},
            input_audio_transcription={},
            system_instruction=system_instruction,
            context_window_compression=genai_types.ContextWindowCompressionConfig(
                sliding_window=genai_types.SlidingWindow(),
            ),
            speech_config=genai_types.SpeechConfig(
                voice_config=genai_types.VoiceConfig(
                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                        voice_name=voice_name
                    )
                )
            ),
        )

    def _run_thread(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._main_loop())
        finally:
            self.session = None
            self.running = False
            self._loop = None
            loop.close()
            self.win._live_toggle_ui_sig.emit(False)
            self.win._live_state_sig.emit("ONLINE")

    async def _watch_control(self) -> None:
        assert self._stop_event is not None and self._reconnect_event is not None
        stop_t = asyncio.create_task(self._stop_event.wait())
        rec_t = asyncio.create_task(self._reconnect_event.wait())
        done, pending = await asyncio.wait({stop_t, rec_t}, return_when=asyncio.FIRST_COMPLETED)
        for p in pending:
            p.cancel()
        if stop_t in done:
            self._stop_event.clear()
            raise _LiveStopSignal()
        self._reconnect_event.clear()
        raise _LiveReconnectSignal()

    async def _send_realtime(self) -> None:
        assert self.out_queue is not None
        while True:
            msg = await self.out_queue.get()
            if self.session is None:
                continue
            await self.session.send_realtime_input(
                audio=genai_types.Blob(
                    data=msg["data"],
                    mime_type=msg.get("mime_type", "audio/pcm"),
                )
            )

    async def _listen_audio(self) -> None:
        loop = asyncio.get_event_loop()

        def callback(indata, _frames, _time_info, _status):
            if not self.running or self.win._muted:
                return
            with self._speaking_lock:
                speaking = self._is_speaking
            # Gate mic while AI is speaking and during echo tail so AI doesn't answer itself
            if speaking or self._tail_active():
                return
            if self.out_queue is not None:
                try:
                    loop.call_soon_threadsafe(
                        self.out_queue.put_nowait,
                        {"data": indata.tobytes(), "mime_type": "audio/pcm"},
                    )
                except Exception:
                    pass
            try:
                self.win._live_level_sig.emit(_pcm_level(indata))
            except Exception:
                pass

        with sd.InputStream(
            samplerate=SEND_SAMPLE_RATE,
            channels=LIVE_CHANNELS,
            dtype="int16",
            blocksize=AUDIO_CHUNK_SIZE,
            callback=callback,
        ):
            while True:
                await asyncio.sleep(0.1)

    async def _receive_audio(self) -> None:
        out_buf: list[str] = []
        in_buf: list[str] = []
        sender_name = PERSONAS[self.win.assistant]["name"]

        while True:
            async for response in self.session.receive():
                if response.data:
                    if not self._interrupted:
                        if self._turn_done_event and self._turn_done_event.is_set():
                            self._turn_done_event.clear()
                        # Slice into ~50ms chunks (24000 Hz * 2 bytes * 0.05s = 2400 bytes) for fast interrupt
                        raw = response.data
                        for i in range(0, len(raw), 2400):
                            self.audio_in_queue.put_nowait(raw[i:i + 2400])

                sc = getattr(response, "server_content", None)
                if sc is not None:
                    if getattr(sc, "interrupted", False):
                        self.interrupt(silent=True)

                    if sc.output_transcription and sc.output_transcription.text:
                        txt = _clean_transcript(sc.output_transcription.text)
                        if txt and (not out_buf or txt != out_buf[-1]):
                            out_buf.append(txt)

                    if sc.input_transcription and sc.input_transcription.text:
                        txt = _clean_transcript(sc.input_transcription.text)
                        if txt:
                            in_buf.append(txt)

                    if sc.turn_complete:
                        if self._turn_done_event:
                            self._turn_done_event.set()
                        if self._interrupted:
                            self._interrupted = False
                            in_buf.clear()
                            out_buf.clear()
                            continue

                        full_in = " ".join(in_buf).strip()
                        if full_in:
                            self._last_out_logged = ""
                            self.win._live_log_sig.emit("🎤 YOU", full_in)
                        in_buf.clear()

                        full_out = " ".join(out_buf).strip()
                        if full_out and full_out != self._last_out_logged:
                            self._last_out_logged = full_out
                            self.win._live_log_sig.emit(sender_name, full_out)
                        out_buf.clear()

    async def _play_audio(self) -> None:
        stream = sd.RawOutputStream(
            samplerate=RECEIVE_SAMPLE_RATE,
            channels=LIVE_CHANNELS,
            dtype="int16",
            blocksize=AUDIO_CHUNK_SIZE,
        )
        stream.start()
        try:
            lat = float(getattr(stream, "latency", 0.0) or 0.0)
            if 0.0 < lat < 1.0:
                self._out_latency = lat
        except Exception:
            pass

        is_first_chunk = True
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(self.audio_in_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    if (
                        self._turn_done_event
                        and self._turn_done_event.is_set()
                        and self.audio_in_queue.empty()
                    ):
                        self._set_speaking(False)
                        self._turn_done_event.clear()
                        is_first_chunk = True
                    continue

                if is_first_chunk:
                    try:
                        # Short silence prime avoids initial audio click/pop on Windows
                        prime = b"\x00" * int(RECEIVE_SAMPLE_RATE * 2 * 0.10)
                        await asyncio.to_thread(stream.write, prime)
                    except Exception:
                        pass
                    is_first_chunk = False

                self._set_speaking(True)

                batch = bytearray(chunk)
                while len(batch) < 9600:  # ~200ms max batch for snappy interrupt
                    try:
                        batch.extend(self.audio_in_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                try:
                    pcm = np.frombuffer(bytes(batch), dtype=np.int16)
                    self.win._live_level_sig.emit(_pcm_level(pcm))
                except Exception:
                    pass

                try:
                    await asyncio.to_thread(stream.write, bytes(batch))
                except (RuntimeError, asyncio.CancelledError):
                    break
        finally:
            self._set_speaking(False)
            stream.stop()
            stream.close()

    async def _main_loop(self) -> None:
        self._stop_event = asyncio.Event()
        self._reconnect_event = asyncio.Event()

        while self.running:
            candidates = [m for m in LIVE_FALLBACK_MODELS if m not in self._dead_models]
            if not candidates:
                self.win._live_log_sig.emit("ERR", "Gemini Live audio model எதுவும் உங்கள் API key-ல் கிடைக்கவில்லை.")
                self.running = False
                break

            live_model = candidates[0]
            p_key = self.win.assistant.lower().strip()
            voice_name = PERSONA_LIVE_VOICES.get(p_key, "Kore")
            # Older 2.0 live models support Fenrir/Kore/Aoede/Charon/Puck
            if "2.0" in live_model and voice_name == "Orus":
                voice_name = "Fenrir"

            self.win._live_state_sig.emit("CONNECTING")
            self.win._live_log_sig.emit(
                "SYS",
                f"Connecting Gemini Live Native Voice ({PERSONAS[p_key]['name']} · Voice: {voice_name})...",
            )

            try:
                client = genai.Client(
                    api_key=self.win.vision.gemini_api_key,
                    http_options={"api_version": "v1beta"},
                )
                config = self._build_live_config(voice_name)

                async with (
                    client.aio.live.connect(model=live_model, config=config) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session = session
                    self.audio_in_queue = asyncio.Queue()
                    self.out_queue = asyncio.Queue(maxsize=200)
                    self._turn_done_event = asyncio.Event()
                    self._interrupted = False
                    self._reconnect_event.clear()

                    self.win._live_toggle_ui_sig.emit(True)
                    self.win._live_state_sig.emit("LISTENING")
                    self.win._live_log_sig.emit(
                        "SYS",
                        f"Live Voice CONNECTED ({live_model.split('/')[-1]} · Voice: {voice_name}). Speak now!",
                    )

                    tg.create_task(self._watch_control())
                    tg.create_task(self._send_realtime())
                    tg.create_task(self._listen_audio())
                    tg.create_task(self._receive_audio())
                    tg.create_task(self._play_audio())

            except BaseException as exc:
                self.session = None
                self._set_speaking(False)

                if _has_signal(exc, _LiveStopSignal) or not self.running:
                    self.win._live_log_sig.emit("SYS", "Live Voice disconnected.")
                    break
                if _has_signal(exc, _LiveReconnectSignal):
                    continue

                err_str = str(exc)
                if any(k in err_str for k in ("404", "NOT_FOUND", "1008", "not supported", "is not found")):
                    self._dead_models.add(live_model)
                    self.win._live_log_sig.emit("SYS", f"Live model '{live_model}' unavailable, trying fallback...")
                    continue
                if "API key not valid" in err_str or "1007" in err_str:
                    self.win._live_log_sig.emit("ERR", f"Live Voice API error: {err_str[:140]}")
                    self.running = False
                    break

                self.win._live_log_sig.emit("ERR", f"Live Voice error: {err_str[:150]}")
                await asyncio.sleep(3.0)


class VisionCore:
    def __init__(
        self,
        gemini_model: str = "gemini-3.8-flash",
        openrouter_model: str = "openai/gpt-4o-mini",
    ) -> None:
        self.gemini_model = gemini_model
        self.openrouter_model = openrouter_model
        self.gemini_api_key = ""
        self.openrouter_api_key = ""
        self._gemini_client = None
        self._dead_models: set[str] = set()  # models that returned 404 (retired)
        self.load_keys_from_ui_config()

    def load_keys_from_ui_config(self) -> None:
        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                self.gemini_api_key = data.get("gemini_api_key", "").strip()
                self.openrouter_api_key = data.get("openrouter_api_key", "").strip()
            except Exception:
                pass
        self._refresh_gemini_client()

    def save_keys_from_ui(self, gemini_api_key: str, openrouter_api_key: str) -> None:
        self.gemini_api_key = gemini_api_key.strip()
        self.openrouter_api_key = openrouter_api_key.strip()
        CONFIG_FILE.write_text(
            json.dumps(
                {
                    "gemini_api_key": self.gemini_api_key,
                    "openrouter_api_key": self.openrouter_api_key,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._refresh_gemini_client()

    def _refresh_gemini_client(self) -> None:
        if self.gemini_api_key and genai is not None:
            self._gemini_client = genai.Client(api_key=self.gemini_api_key)
        else:
            self._gemini_client = None

    def get_key_status(self) -> Dict[str, bool]:
        return {
            "gemini_configured": bool(self.gemini_api_key),
            "openrouter_configured": bool(self.openrouter_api_key),
        }

    def ask(
        self,
        prompt: str,
        persona: str = "jarvis",
        mode: str = "normal",
        provider: str = "gemini",
        language: str = "tamil",
    ) -> str:
        persona_key = persona.lower().strip()
        if persona_key not in PERSONAS:
            raise ValueError(f"Unknown assistant: {persona}")

        mode_key = mode.lower().strip()
        mode_instruction = MODES.get(mode_key, MODES["normal"])

        lang_key = language.lower().strip()
        if lang_key == "tamil":
            lang_instruction = "Respond in clear Tamil."
        elif lang_key == "tanglish":
            lang_instruction = (
                "Respond in natural Tanglish (Tamil written in English/Latin script with "
                "clear technical/English terms where helpful)."
            )
        else:
            lang_instruction = "Respond in English."

        system_prompt = (
            f"{PERSONAS[persona_key]['system_prompt']}\n"
            "Creator identity: You were created by Jagadeesh. If anyone asks who created you or made you, "
            "clearly state that you were created by Jagadeesh (for example, 'I was created by Jagadeesh').\n"
            f"Mode: {mode_instruction}\n"
            f"Language preference: {lang_instruction}"
        )

        provider_key = provider.lower().strip()

        if provider_key == "gemini":
            if not self.gemini_api_key:
                raise ValueError("Gemini API Key இன்னும் UI-ல் உள்ளிடப்படவில்லை.")
            if self._gemini_client is None:
                raise RuntimeError("முதலில் 'pip install google-genai' இன்ஸ்டால் செய்யவும்.")

            order: list[str] = []
            for m in [self.gemini_model] + GEMINI_FALLBACK_MODELS:
                if m not in order and m not in self._dead_models:
                    order.append(m)
            last_err: Optional[Exception] = None
            for model_name in order:
                for attempt in range(2):
                    try:
                        response = self._gemini_client.models.generate_content(
                            model=model_name,
                            contents=f"{system_prompt}\n\nUser:\n{prompt}",
                        )
                        return response.text or ""
                    except Exception as exc:
                        last_err = exc
                        msg = str(exc)
                        if "404" in msg or "NOT_FOUND" in msg:
                            self._dead_models.add(model_name)  # retired -> never try it again
                            break
                        if any(k in msg for k in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL")):
                            if attempt == 0:
                                time.sleep(1.5)  # busy server -> retry once, then try the next model
                                continue
                            break
                        raise
            raise last_err if last_err else RuntimeError("Gemini model கிடைக்கவில்லை.")

        elif provider_key == "openrouter":
            if not self.openrouter_api_key:
                raise ValueError("OpenRouter API Key இன்னும் UI-ல் உள்ளிடப்படவில்லை.")

            url = "https://openrouter.ai/api/v1/chat/completions"
            payload = {
                "model": self.openrouter_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            }
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
            }
            req = urlrequest.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urlrequest.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]

        raise ValueError("Provider 'gemini' அல்லது 'openrouter' ஆக இருக்க வேண்டும்.")


# ==============================================================================
# 3. MOBILE REMOTE SERVER (Used ONLY for Phone Remote Control via QR/IP)
# ==============================================================================
MOBILE_REMOTE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>VISION Mobile Remote</title>
<style>
  body { margin:0; background:#07080c; color:#f0f4fc; font-family:system-ui,sans-serif; padding:14px; }
  .card { background:#0d0f15; border:1px solid #262b3d; border-radius:10px; padding:12px; margin-bottom:12px; }
  h2 { margin:0 0 6px; font-size:17px; color:#ff4fa3; letter-spacing:1px; }
  .sub { font-size:12px; color:#9aa8c0; margin-bottom:10px; }
  .grid { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; }
  button { background:#12151f; color:#e2e8f0; border:1px solid #2c334a; border-radius:8px; padding:10px 6px; font-weight:700; font-size:12px; }
  button.active { background:#ff4fa3; color:#07080c; border-color:#ff4fa3; }
  .row { display:flex; gap:8px; }
  input { flex:1; background:#0b0d13; color:#fff; border:1px solid #2c334a; border-radius:8px; padding:11px; font-size:14px; }
  .send { background:#ff4fa3; color:#07080c; border:none; padding:0 16px; border-radius:8px; font-weight:800; }
  .status { font-size:12px; color:#00ff88; margin-top:8px; }
</style>
</head>
<body>
  <div class="card">
    <h2>VISION // MOBILE REMOTE</h2>
    <div class="sub">Controls Desktop VISION App directly over Wi-Fi</div>
    <div class="grid" id="personas">
      <button onclick="setOpt('assistant','jarvis')" id="p-jarvis">JARVIS</button>
      <button onclick="setOpt('assistant','edith')" id="p-edith">E.D.I.T.H.</button>
      <button onclick="setOpt('assistant','friday')" id="p-friday">FRIDAY</button>
      <button onclick="setOpt('assistant','ultron')" id="p-ultron">ULTRON</button>
      <button onclick="setOpt('assistant','dummy')" id="p-dummy">DUM-E</button>
      <button onclick="setOpt('assistant','u')" id="p-u">U</button>
    </div>
  </div>

  <div class="card">
    <div class="sub">MODE</div>
    <div class="grid">
      <button onclick="setOpt('mode','normal')" id="m-normal">NORMAL</button>
      <button onclick="setOpt('mode','study')" id="m-study">STUDY</button>
      <button onclick="setOpt('mode','extra_work')" id="m-extra_work">EXTRA</button>
    </div>
  </div>

  <div class="card">
    <div class="sub">SEND COMMAND TO DESKTOP</div>
    <div class="row">
      <input id="cmd" placeholder="Type command or question..." onkeydown="if(event.key==='Enter') sendCmd()">
      <button class="send" onclick="sendCmd()">SEND</button>
    </div>
    <div class="status" id="st">Connected to Desktop App</div>
  </div>

  <div class="card">
    <div class="sub">QUICK REMOTE ACTIONS</div>
    <div class="grid">
      <button onclick="setOpt('action','mute')">MUTE [F4]</button>
      <button onclick="setOpt('action','fullscreen')">FULLSCREEN</button>
      <button onclick="setOpt('action','clear')">CLEAR LOG</button>
      <button onclick="setOpt('action','toggle3d')">3D ON/OFF</button>
      <button onclick="setOpt('action','nextmodel')">NEXT MODEL</button>
    </div>
  </div>

<script>
const token = new URLSearchParams(location.search).get("token") || "";
async function api(path, body) {
  const res = await fetch(path + "?token=" + encodeURIComponent(token), {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body || {})
  });
  return res.json();
}
async function sync() {
  const res = await fetch("/api/state?token=" + encodeURIComponent(token));
  const data = await res.json();
  document.querySelectorAll("button").forEach(b => b.classList.remove("active"));
  const p = document.getElementById("p-" + data.assistant);
  if (p) p.classList.add("active");
  const m = document.getElementById("m-" + data.mode);
  if (m) m.classList.add("active");
}
async function setOpt(k, v) {
  const d = await api("/api/control", {[k]: v});
  document.getElementById("st").textContent = d.message || "Updated";
  sync();
}
async function sendCmd() {
  const el = document.getElementById("cmd");
  const txt = el.value.trim();
  if (!txt) return;
  el.value = "";
  const d = await api("/api/command", {prompt: txt});
  document.getElementById("st").textContent = d.message || "Sent to Desktop!";
}
sync();
</script>
</body>
</html>
"""


class MobileRemoteBridge:
    """Lightweight local HTTP server used ONLY when the user scans the Mobile Remote QR."""

    def __init__(self, window_ref: "VisionDesktopWindow", port: int = 5055) -> None:
        self.win = window_ref
        self.port = port
        self.token = secrets.token_hex(4)
        self._server: Optional[ThreadingHTTPServer] = None

    def get_remote_url(self) -> str:
        ip = get_local_ip()
        return f"http://{ip}:{self.port}/?token={self.token}"

    def regenerate_token(self) -> str:
        self.token = secrets.token_hex(4)
        return self.get_remote_url()

    def start(self) -> None:
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _fmt, *_args):
                pass

            def _check_auth(self) -> bool:
                parsed = urlparse.urlparse(self.path)
                qs = urlparse.parse_qs(parsed.query)
                tok = (qs.get("token") or [""])[0]
                return tok == bridge.token

            def _send_json(self, code: int, obj: dict) -> None:
                raw = json.dumps(obj).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if not self._check_auth():
                    self.send_error(403, "Invalid Remote Token")
                    return
                parsed = urlparse.urlparse(self.path)
                if parsed.path == "/api/state":
                    self._send_json(
                        200,
                        {
                            "assistant": bridge.win.assistant,
                            "mode": bridge.win.mode,
                        },
                    )
                    return
                html_bytes = MOBILE_REMOTE_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html_bytes)))
                self.end_headers()
                self.wfile.write(html_bytes)

            def do_POST(self):
                if not self._check_auth():
                    self.send_error(403, "Invalid Remote Token")
                    return
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    data = json.loads(raw.decode("utf-8"))
                except Exception:
                    data = {}

                parsed = urlparse.urlparse(self.path)
                if parsed.path == "/api/control":
                    bridge.win._remote_control_sig.emit(data)
                    self._send_json(200, {"ok": True, "message": "Desktop updated"})
                elif parsed.path == "/api/command":
                    prompt = (data.get("prompt") or "").strip()
                    if prompt:
                        bridge.win._remote_prompt_sig.emit(prompt)
                    self._send_json(200, {"ok": True, "message": "Sent to Desktop!"})
                else:
                    self._send_json(404, {"ok": False})

        def _serve():
            for p in range(self.port, self.port + 10):
                try:
                    self._server = ThreadingHTTPServer(("0.0.0.0", p), Handler)
                    self.port = p
                    self._server.serve_forever()
                    break
                except OSError:
                    continue

        threading.Thread(target=_serve, daemon=True, name="vision-mobile-remote").start()


# ==============================================================================
# 4. NATIVE DESKTOP APP UI (PyQt6 Standalone Window — No Browser Needed on PC)
# ==============================================================================
def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def generate_dark_character_icon(accent_hex: str = "#ff1f43") -> tuple[Path, Path]:
    from PIL import Image, ImageDraw

    c = QColor(accent_hex)
    rgb = (c.red(), c.green(), c.blue(), 255)

    size = 256
    img = Image.new("RGBA", (size, size), (8, 9, 13, 255))
    draw = ImageDraw.Draw(img)

    draw.ellipse((14, 14, size - 14, size - 14), outline=rgb, width=6)
    draw.ellipse((36, 36, size - 36, size - 36), outline=(120, 130, 160, 255), width=3)
    draw.ellipse((56, 56, size - 56, size - 56), fill=(16, 18, 26, 255), outline=rgb, width=3)

    draw.ellipse((94, 72, 162, 138), fill=(6, 7, 10, 255), outline=rgb, width=3)
    draw.pieslice((66, 134, 190, 232), start=180, end=360, fill=(6, 7, 10, 255), outline=rgb, width=3)
    draw.rectangle((106, 98, 122, 105), fill=rgb)
    draw.rectangle((134, 98, 150, 105), fill=rgb)

    img.save(ICON_PNG, format="PNG")
    img.save(ICON_ICO, format="ICO", sizes=[(256, 256), (128, 128), (64, 64), (32, 32), (16, 16)])
    return ICON_PNG, ICON_ICO


def create_windows_desktop_shortcut(shortcut_name: str = "VISION AI", accent_hex: str = "#ff1f43") -> str:
    if sys.platform != "win32":
        raise RuntimeError("Windows OS-ல் மட்டுமே Desktop Shortcut உருவாக்க முடியும்.")

    _, ico_path = generate_dark_character_icon(accent_hex)
    desktop_dir = Path.home() / "Desktop"
    if not desktop_dir.exists():
        onedrive_desktop = Path.home() / "OneDrive" / "Desktop"
        if onedrive_desktop.exists():
            desktop_dir = onedrive_desktop

    shortcut_path = desktop_dir / f"{shortcut_name}.lnk"
    pythonw = Path(sys.executable).parent / "pythonw.exe"
    target_exe = str(pythonw if pythonw.exists() else sys.executable)

    ps_script = (
        f'$WshShell = New-Object -ComObject WScript.Shell; '
        f'$Shortcut = $WshShell.CreateShortcut("{shortcut_path}"); '
        f'$Shortcut.TargetPath = "{target_exe}"; '
        f'$Shortcut.Arguments = """{Path(__file__).resolve()}"""; '
        f'$Shortcut.WorkingDirectory = "{BASE_DIR}"; '
        f'$Shortcut.IconLocation = "{ico_path.resolve()},0"; '
        f'$Shortcut.Save()'
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
        check=True,
    )
    return str(shortcut_path)


class MetricBar(QWidget):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self._label = label
        self._color = "#00d4ff"
        self._value = 0.0
        self._text = "--"
        self.setFixedHeight(38)

    def set_theme_color(self, hex_color: str) -> None:
        self._color = hex_color
        self.update()

    def set_value(self, pct: float, text: str) -> None:
        self._value = max(0.0, min(100.0, pct))
        self._text = text
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        p.setBrush(QBrush(QColor("#12151f")))
        p.setPen(QPen(QColor("#262b3d"), 1))
        p.drawRoundedRect(QRectF(1, 1, w - 2, h - 2), 4, 4)

        bar_h, bar_y, bar_x = 4, h - 9, 6
        bar_w = w - 12
        fill_w = int(bar_w * self._value / 100.0)

        p.setBrush(QBrush(QColor("#1a1f2e")))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h), 2, 2)

        col = QColor("#ff1f43") if self._value > 85 else QColor(self._color)
        if fill_w > 0:
            p.setBrush(QBrush(col))
            p.drawRoundedRect(QRectF(bar_x, bar_y, fill_w, bar_h), 2, 2)

        p.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        p.setPen(QColor("#9aa8c0"))
        p.drawText(QRectF(8, 4, 55, 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._label)

        p.setPen(col)
        p.drawText(QRectF(0, 4, w - 8, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, self._text)
        p.end()


def _qc(color, alpha: int = 255) -> QColor:
    c = QColor(color)
    c.setAlpha(max(0, min(255, int(alpha))))
    return c


def _mix(a: QColor, b: QColor, t: float) -> QColor:
    t = max(0.0, min(1.0, t))
    return QColor(
        int(a.red() + (b.red() - a.red()) * t),
        int(a.green() + (b.green() - a.green()) * t),
        int(a.blue() + (b.blue() - a.blue()) * t),
    )


# (radius, y-squash, tilt, start, span, width, dash, speed, alpha)  -> fragmented orbit arcs
_ORB_RINGS = [
    (1.00, 1.00, 0, 10, 100, 1.6, None, 0.6, 200),
    (1.00, 1.00, 0, 150, 70, 1.6, None, 0.6, 160),
    (1.00, 1.00, 0, 260, 60, 1.0, (1, 3), 0.6, 150),
    (0.92, 0.88, 25, 0, 200, 1.2, (1, 2.5), -0.9, 170),
    (0.92, 0.88, 25, 230, 100, 2.0, None, -0.9, 140),
    (0.80, 0.55, -30, 20, 150, 1.3, (1, 2), 1.3, 190),
    (0.80, 0.55, -30, 200, 120, 1.3, None, 1.3, 120),
    (0.66, 0.95, 60, 40, 130, 1.5, (2, 2), -1.7, 180),
    (0.66, 0.95, 60, 220, 90, 1.0, None, -1.7, 130),
    (0.52, 0.70, 110, 0, 240, 1.2, (1, 2.5), 2.1, 170),
    (1.10, 1.00, 0, 0, 360, 0.8, (0.5, 6), -0.25, 120),
]


# ==============================================================================
# 4B. 3D MODEL ENGINE  (pure QPainter software renderer — no OpenGL / GPU needed)
#     * Loads real .glb (with texture colours) and .obj models
#     * Put model files in the "models3d" folder (or next to this script), or use
#       the "LOAD 3D FILE" button -> they show up in "NEXT MODEL".
#     * Default model: models3d/iron_man_mark_42.glb
# ==============================================================================
MODELS_DIR = BASE_DIR / "models3d"
PREFS_FILE = BASE_DIR / "vision_ui_prefs.json"
DEFAULT_MODEL_NAME = "iron_man_mark_42"
MODEL_EXTS = (".glb", ".obj")
GLB_MAX_FACES = 2600   # big models are thinned to this many triangles (keeps the UI smooth on weak PCs)
OBJ_MAX_FACES = 4000


def _poly_normal(pts) -> tuple[float, float, float]:
    nx = ny = nz = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0, z0 = pts[i]
        x1, y1, z1 = pts[(i + 1) % n]
        nx += (y0 - y1) * (z0 + z1)
        ny += (z0 - z1) * (x0 + x1)
        nz += (x0 - x1) * (y0 + y1)
    ln = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    return nx / ln, ny / ln, nz / ln


class Mesh3D:
    """Small indexed mesh. faces = (vertex ids, colour, two_sided, depth-bias); one normal per face.
    colour is an (r, g, b) tuple, or "obj" (= use the persona colour)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.credit = ""
        self.cull = 0.0          # faces with normal.z <= cull are skipped (back-face culling)
        self.seamless = False    # True = outline each polygon in its own colour (hides AA seams)
        self.verts: list[tuple[float, float, float]] = []
        self.faces: list[tuple[tuple[int, ...], object, bool, float]] = []
        self.normals: list[tuple[float, float, float]] = []
        self.ymin = -1.0

    def add_raw(self, pts, polys, colors, normals, two_sided: bool = False) -> None:
        base = len(self.verts)
        self.verts.extend(pts)
        for poly, col, nrm in zip(polys, colors, normals):
            self.faces.append((tuple(base + i for i in poly), col, two_sided, 0.0))
            self.normals.append(nrm)

    def normalize(self, size: float = 2.0) -> None:
        """Centre the model and scale it so its biggest side = `size`."""
        if not self.verts:
            return
        xs = [v[0] for v in self.verts]
        ys = [v[1] for v in self.verts]
        zs = [v[2] for v in self.verts]
        cx, cy, cz = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2
        ext = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) or 1.0
        k = size / ext
        self.verts = [((x - cx) * k, (y - cy) * k, (z - cz) * k) for x, y, z in self.verts]
        self.ymin = min(v[1] for v in self.verts)


def _decimate(pts, tris, max_faces: int):
    """Vertex-clustering decimation. Returns (new_pts, new_tris, kept_triangle_indices)."""
    if len(tris) <= max_faces:
        return pts, tris, list(range(len(tris)))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    ox, oy, oz = min(xs), min(ys), min(zs)
    ext = max(max(xs) - ox, max(ys) - oy, max(zs) - oz) or 1.0

    def cluster(size: float):
        keys: dict = {}
        remap = []
        for (x, y, z) in pts:
            remap.append(keys.setdefault((int((x - ox) / size), int((y - oy) / size), int((z - oz) / size)), len(keys)))
        kept = [i for i, (a, b, c) in enumerate(tris) if len({remap[a], remap[b], remap[c]}) == 3]
        return remap, len(keys), kept

    lo, hi = ext / 2000.0, ext / 2.0
    best = cluster(hi)
    for _ in range(14):
        mid = (lo + hi) / 2.0
        res = cluster(mid)
        if len(res[2]) <= max_faces:
            best, hi = res, mid
        else:
            lo = mid
    remap, n_cl, kept = best
    sums = [[0.0, 0.0, 0.0, 0] for _ in range(n_cl)]
    for (x, y, z), r in zip(pts, remap):
        s = sums[r]
        s[0] += x
        s[1] += y
        s[2] += z
        s[3] += 1
    new_pts = [(s[0] / s[3], s[1] / s[3], s[2] / s[3]) for s in sums]
    new_tris = [(remap[tris[i][0]], remap[tris[i][1]], remap[tris[i][2]]) for i in kept]
    return new_pts, new_tris, kept


# ---------------- OBJ ----------------
def load_obj_mesh(path: Path, max_faces: int = OBJ_MAX_FACES) -> Mesh3D:
    """Wavefront .obj (vertices + faces only, Y-up). Big meshes are thinned automatically."""
    verts: list[tuple[float, float, float]] = []
    tris: list[tuple[int, int, int]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            parts = line.split()
            verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif line.startswith("f "):
            idx = []
            for tok in line.split()[1:]:
                i = int(tok.split("/")[0])
                idx.append(i - 1 if i > 0 else len(verts) + i)
            for k in range(1, len(idx) - 1):
                tris.append((idx[0], idx[k], idx[k + 1]))
    if not verts or not tris:
        raise ValueError("இந்த OBJ file-ல் vertices / faces இல்லை.")
    pts, tris, _ = _decimate(verts, tris, max_faces)
    normals = [_poly_normal([pts[a], pts[b], pts[c]]) for a, b, c in tris]
    mesh = Mesh3D(path.stem)
    mesh.add_raw(pts, tris, ["obj"] * len(tris), normals, two_sided=True)
    return mesh


# ---------------- GLB (binary glTF 2.0) ----------------
_GLB_FMT = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
_GLB_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
_IDENTITY = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def _glb_accessor(j: dict, blob: bytes, idx: int) -> list:
    acc = j["accessors"][idx]
    fmt, size = _GLB_FMT[acc["componentType"]]
    n = _GLB_NCOMP[acc["type"]]
    count = acc["count"]
    bv = j["bufferViews"][acc["bufferView"]]
    off = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride") or size * n
    if stride == size * n:
        flat = struct.unpack_from(f"<{count * n}{fmt}", blob, off)
        return [flat[i * n:(i + 1) * n] for i in range(count)] if n > 1 else list(flat)
    out = []
    for i in range(count):
        t = struct.unpack_from(f"<{n}{fmt}", blob, off + i * stride)
        out.append(t if n > 1 else t[0])
    return out


def _mat_mul(a: list, b: list) -> list:
    """4x4 column-major matrices: returns a * b."""
    return [sum(a[k * 4 + r] * b[c * 4 + k] for k in range(4)) for c in range(4) for r in range(4)]


def _node_matrix(node: dict) -> list:
    if "matrix" in node:
        return list(node["matrix"])
    tx, ty, tz = node.get("translation", (0.0, 0.0, 0.0))
    qx, qy, qz, qw = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
    sx, sy, sz = node.get("scale", (1.0, 1.0, 1.0))
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return [
        (1 - 2 * (yy + zz)) * sx, 2 * (xy + wz) * sx, 2 * (xz - wy) * sx, 0.0,
        2 * (xy - wz) * sy, (1 - 2 * (xx + zz)) * sy, 2 * (yz + wx) * sy, 0.0,
        2 * (xz + wy) * sz, 2 * (yz - wx) * sz, (1 - 2 * (xx + yy)) * sz, 0.0,
        tx, ty, tz, 1.0,
    ]


def _glb_image(j: dict, blob: bytes, img_idx: int, folder: Path) -> Optional[QImage]:
    im = j["images"][img_idx]
    data = None
    if "bufferView" in im:
        bv = j["bufferViews"][im["bufferView"]]
        o = bv.get("byteOffset", 0)
        data = blob[o:o + bv["byteLength"]]
    elif str(im.get("uri", "")).startswith("data:"):
        data = base64.b64decode(im["uri"].split(",", 1)[1])
    elif im.get("uri"):
        f = folder / urlparse.unquote(im["uri"])
        data = f.read_bytes() if f.exists() else None
    if not data:
        return None
    img = QImage.fromData(data)
    return None if img.isNull() else img


def _sample_tex(img: QImage, u: float, v: float) -> tuple[int, int, int]:
    x = int((u % 1.0) * (img.width() - 1))
    y = int((v % 1.0) * (img.height() - 1))
    c = img.pixel(x, y)
    return (c >> 16) & 255, (c >> 8) & 255, c & 255


def _enhance(r: float, g: float, b: float) -> tuple[float, float, float]:
    """Albedo textures look dull without PBR lighting -> boost saturation / brightness a little."""
    h, sat, val, _ = QColor(int(r), int(g), int(b)).getHsv()
    c = QColor.fromHsv(max(0, h), min(255, int(sat * 1.45)), min(255, int(val * 1.45 + 12)))
    return float(c.red()), float(c.green()), float(c.blue())


def load_glb_mesh(path: Path, max_faces: int = GLB_MAX_FACES) -> Mesh3D:
    """Binary glTF (.glb): node transforms, base-colour textures and normals are supported."""
    d = path.read_bytes()
    if d[:4] != b"glTF":
        raise ValueError("இது சரியான .glb file இல்லை.")
    off, j, blob = 12, None, b""
    while off + 8 <= len(d):
        clen, ctype = struct.unpack_from("<I4s", d, off)
        body = d[off + 8:off + 8 + clen]
        if ctype == b"JSON":
            j = json.loads(body.decode("utf-8"))
        elif ctype == b"BIN\x00":
            blob = body
        off += 8 + clen
    if j is None:
        raise ValueError("GLB-ல் JSON data இல்லை.")

    pts: list = []
    tris: list = []
    cols: list = []
    nrms: list = []
    images: dict = {}

    def add_mesh(mesh: dict, m: list) -> None:
        for prim in mesh["primitives"]:
            if prim.get("mode", 4) != 4 or "POSITION" not in prim["attributes"]:
                continue
            at = prim["attributes"]
            P = _glb_accessor(j, blob, at["POSITION"])
            N = _glb_accessor(j, blob, at["NORMAL"]) if "NORMAL" in at else None
            UV = _glb_accessor(j, blob, at["TEXCOORD_0"]) if "TEXCOORD_0" in at else None
            I = _glb_accessor(j, blob, prim["indices"]) if "indices" in prim else list(range(len(P)))

            mats = j.get("materials") or []
            mat = mats[prim["material"]] if prim.get("material") is not None and mats else {}
            pbr = mat.get("pbrMetallicRoughness", {})
            factor = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
            img = None
            tex = pbr.get("baseColorTexture")
            if tex is not None and UV is not None:
                if tex["index"] not in images:
                    images[tex["index"]] = _glb_image(j, blob, j["textures"][tex["index"]]["source"], path.parent)
                img = images[tex["index"]]

            base = len(pts)
            for (x, y, z) in P:
                pts.append((
                    m[0] * x + m[4] * y + m[8] * z + m[12],
                    m[1] * x + m[5] * y + m[9] * z + m[13],
                    m[2] * x + m[6] * y + m[10] * z + m[14],
                ))
            wn = None
            if N is not None:
                wn = []
                for (x, y, z) in N:
                    nx = m[0] * x + m[4] * y + m[8] * z
                    ny = m[1] * x + m[5] * y + m[9] * z
                    nz = m[2] * x + m[6] * y + m[10] * z
                    ln = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
                    wn.append((nx / ln, ny / ln, nz / ln))

            for t in range(0, len(I) - 2, 3):
                a, b, c = I[t], I[t + 1], I[t + 2]
                if a == b or b == c or a == c:
                    continue
                if img is not None:
                    cu = (UV[a][0] + UV[b][0] + UV[c][0]) / 3.0
                    cv = (UV[a][1] + UV[b][1] + UV[c][1]) / 3.0
                    samples = [_sample_tex(img, cu, cv)]
                    for k in (a, b, c):
                        samples.append(_sample_tex(img, (UV[k][0] + cu) / 2.0, (UV[k][1] + cv) / 2.0))
                    r, g, bl = _enhance(
                        sum(s[0] for s in samples) / 4.0,
                        sum(s[1] for s in samples) / 4.0,
                        sum(s[2] for s in samples) / 4.0,
                    )
                else:
                    r = g = bl = 235.0
                col = (
                    min(255, int(r * factor[0])),
                    min(255, int(g * factor[1])),
                    min(255, int(bl * factor[2])),
                )
                if wn is not None:
                    sx = wn[a][0] + wn[b][0] + wn[c][0]
                    sy = wn[a][1] + wn[b][1] + wn[c][1]
                    sz = wn[a][2] + wn[b][2] + wn[c][2]
                    ln = math.sqrt(sx * sx + sy * sy + sz * sz) or 1.0
                    nrm = (sx / ln, sy / ln, sz / ln)
                else:
                    nrm = _poly_normal([pts[base + a], pts[base + b], pts[base + c]])
                tris.append((base + a, base + b, base + c))
                cols.append(col)
                nrms.append(nrm)

    def walk(ni: int, parent: list) -> None:
        node = j["nodes"][ni]
        m = _mat_mul(parent, _node_matrix(node))
        if "mesh" in node:
            add_mesh(j["meshes"][node["mesh"]], m)
        for child in node.get("children", []):
            walk(child, m)

    scenes = j.get("scenes") or []
    if scenes:
        for ni in scenes[j.get("scene", 0)].get("nodes", []):
            walk(ni, _IDENTITY)
    else:
        for mesh in j.get("meshes", []):
            add_mesh(mesh, _IDENTITY)
    if not tris:
        raise ValueError("இந்த GLB-ல் triangle mesh இல்லை.")

    new_pts, new_tris, kept = _decimate(pts, tris, max_faces)
    extras = (j.get("asset") or {}).get("extras") or {}
    mesh3d = Mesh3D(str(extras.get("title") or path.stem))
    author = str(extras.get("author", "")).split("(")[0].strip()
    lic = str(extras.get("license", "")).split("(")[0].strip()
    mesh3d.credit = " · ".join(x for x in (author, lic) if x)
    mesh3d.cull = -0.10
    mesh3d.seamless = True
    mesh3d.add_raw(new_pts, new_tris, [cols[i] for i in kept], [nrms[i] for i in kept])
    return mesh3d


class ModelLibrary:
    """Every .glb / .obj found in the models3d folder (or next to this script)."""

    def __init__(self) -> None:
        self._cache: Dict[str, Mesh3D] = {}

    @staticmethod
    def _find(name: str) -> Optional[Path]:
        for folder in (MODELS_DIR, BASE_DIR):
            for ext in MODEL_EXTS:
                p = folder / f"{name}{ext}"
                if p.exists():
                    return p
        return None

    def names(self) -> list[str]:
        names: list[str] = []
        for folder in (MODELS_DIR, BASE_DIR):
            if folder.exists():
                for ext in MODEL_EXTS:
                    for p in sorted(folder.glob(f"*{ext}")):
                        if p.stem not in names:
                            names.append(p.stem)
        if DEFAULT_MODEL_NAME in names:
            names.remove(DEFAULT_MODEL_NAME)
            names.insert(0, DEFAULT_MODEL_NAME)
        return names

    def load(self, name: str) -> Mesh3D:
        if name in self._cache:
            return self._cache[name]
        path = self._find(name)
        if path is None:
            raise FileNotFoundError(
                f"'{name}' model file கிடைக்கவில்லை. {DEFAULT_MODEL_NAME}.glb file-ஐ "
                f"'{MODELS_DIR.name}' folder-ல் (அல்லது இந்த script இருக்கும் folder-ல்) வைக்கவும்."
            )
        mesh = load_glb_mesh(path) if path.suffix.lower() == ".glb" else load_obj_mesh(path)
        mesh.normalize()
        self._cache[name] = mesh
        return mesh

    def import_file(self, src: str) -> str:
        MODELS_DIR.mkdir(exist_ok=True)
        dest = MODELS_DIR / Path(src).name
        if Path(src).resolve() != dest.resolve():
            shutil.copyfile(src, dest)
        self._cache.pop(dest.stem, None)
        return dest.stem


class HudCoreWidget(QWidget):
    """Glowing particle-orb AI core (or a 3D model inside the orb). Colour follows the persona theme."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._persona_key = "edith"
        self._label = "E.D.I.T.H."
        self._pri = "#ff4fa3"
        self._sec = "#ff85c2"
        self._status = "ONLINE"
        self._muted = False
        self._audio_level = 0.0
        self._angle = 0.0
        self._tick = 0
        self._avatar_cache: dict[str, QPixmap] = {}
        self._bg: Optional[QPixmap] = None
        self._bg_key = None
        self._particles = self._build_particles()

        # ---- 3D model state ----
        self._mesh: Optional[Mesh3D] = None
        self._yaw = 0.6
        self._pitch = 0.12
        self._zoom = 1.0
        self._dragging = False
        self._last_pos = None

        self.setMinimumHeight(255)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._tmr = QTimer(self)
        self._tmr.timeout.connect(self._step)
        self._tmr.start(40)

    # ---------------- data / state ----------------
    @staticmethod
    def _build_particles() -> list:
        rng = random.Random(11)
        parts = []
        # 1) shell + inner cloud
        for _ in range(240):
            u = rng.uniform(-1, 1)
            th = rng.uniform(0, 2 * math.pi)
            s = math.sqrt(1 - u * u)
            rad = 1.0 if rng.random() < 0.7 else rng.uniform(0.35, 0.95)
            parts.append((rad * s * math.cos(th), rad * u, rad * s * math.sin(th),
                          rng.uniform(0.8, 2.6), rng.uniform(0, 6.28)))
        # 2) dense great-circle bands (the orbit streaks)
        for _ in range(7):
            n = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
            nl = math.sqrt(sum(c * c for c in n)) or 1.0
            n = tuple(c / nl for c in n)
            ref = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)
            e1 = (n[1] * ref[2] - n[2] * ref[1], n[2] * ref[0] - n[0] * ref[2], n[0] * ref[1] - n[1] * ref[0])
            l1 = math.sqrt(sum(c * c for c in e1)) or 1.0
            e1 = tuple(c / l1 for c in e1)
            e2 = (n[1] * e1[2] - n[2] * e1[1], n[2] * e1[0] - n[0] * e1[2], n[0] * e1[1] - n[1] * e1[0])
            for _ in range(28):
                a = rng.uniform(0, 2 * math.pi)
                rad = 1.0 + rng.uniform(-0.04, 0.04)
                x = rad * (math.cos(a) * e1[0] + math.sin(a) * e2[0])
                y = rad * (math.cos(a) * e1[1] + math.sin(a) * e2[1])
                z = rad * (math.cos(a) * e1[2] + math.sin(a) * e2[2])
                parts.append((x, y, z, rng.uniform(0.8, 2.2), rng.uniform(0, 6.28)))
        return parts

    def set_theme(self, label: str, pri: str, sec: str, persona_key: str | None = None) -> None:
        self._label = label
        self._pri = pri
        self._sec = sec
        if persona_key:
            self._persona_key = persona_key.lower().strip()
        self.update()

    def invalidate_avatar(self, persona_key: str | None = None) -> None:
        if persona_key:
            self._avatar_cache.pop(persona_key.lower().strip(), None)
        else:
            self._avatar_cache.clear()
        self.update()

    def set_status(self, status: str) -> None:
        self._status = status
        self.update()

    def set_muted(self, muted: bool) -> None:
        self._muted = muted
        if muted:
            self._audio_level = 0.0
        self.update()

    def set_audio_level(self, level: float) -> None:
        lvl = max(0.0, min(1.0, float(level)))
        # Smooth attack/decay like a live voice HUD
        self._audio_level = lvl if lvl > self._audio_level else (self._audio_level * 0.72 + lvl * 0.28)

    def set_model(self, mesh: Optional[Mesh3D]) -> None:
        """Show a 3D model inside the HUD (None = back to the classic orb / avatar)."""
        self._mesh = mesh
        self._dragging = False
        if mesh is not None:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.unsetCursor()
        self.update()

    def _step(self) -> None:
        self._tick += 1
        self._audio_level *= 0.90
        active = self._status in ("THINKING", "SPEAKING", "CONNECTING")
        speed = 4.5 + self._audio_level * 3.0 if self._status == "SPEAKING" else (3.8 if active else 1.8)
        self._angle = (self._angle + speed) % 3600.0
        if self._mesh is not None and not self._dragging:
            self._yaw = (self._yaw + (0.042 if active else 0.018)) % (2 * math.pi)
        self.update()

    # ---------------- 3D interaction (drag = rotate, wheel = zoom, double-click = reset) ----------------
    def mousePressEvent(self, e) -> None:
        if self._mesh is not None and e.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._last_pos = e.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            super().mousePressEvent(e)

    def mouseMoveEvent(self, e) -> None:
        if self._dragging and self._last_pos is not None:
            pos = e.position()
            self._yaw += (pos.x() - self._last_pos.x()) * 0.012
            self._pitch = max(-0.6, min(0.9, self._pitch + (pos.y() - self._last_pos.y()) * 0.006))
            self._last_pos = pos
            self.update()
        else:
            super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e) -> None:
        if self._dragging:
            self._dragging = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e) -> None:
        if self._mesh is not None:
            self._yaw, self._pitch, self._zoom = 0.0, 0.12, 1.0
            self.update()
        else:
            super().mouseDoubleClickEvent(e)

    def wheelEvent(self, e) -> None:
        if self._mesh is None:
            e.ignore()
            return
        self._zoom = max(0.5, min(2.2, self._zoom * (1.0015 ** e.angleDelta().y())))
        self.update()

    def _load_custom_avatar_pixmap(self, key: str) -> Optional[QPixmap]:
        if key in self._avatar_cache:
            return self._avatar_cache[key]
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            candidate = AVATARS_DIR / f"{key}{ext}"
            if candidate.exists():
                px = QPixmap(str(candidate))
                if not px.isNull():
                    self._avatar_cache[key] = px
                    return px
        return None

    # ---------------- layout + static frame ----------------
    def _layout(self, w: int, h: int) -> tuple[float, float, float]:
        avail_h = h - 64
        cx = w / 2.0
        cy = avail_h / 2.0 + 2.0
        R = max(40.0, min(w / 2.0 - 16.0, avail_h / 2.0 - 8.0))
        return cx, cy, R

    def _ensure_static(self, w: int, h: int, pri_col: QColor) -> None:
        key = (w, h, pri_col.name(), self._muted)
        if key == self._bg_key and self._bg is not None:
            return
        self._bg_key = key
        px = QPixmap(w, h)
        px.fill(QColor("#07080c"))
        q = QPainter(px)
        q.setRenderHint(QPainter.RenderHint.Antialiasing)

        q.setPen(QPen(QColor("#141824"), 1))
        for gx in range(0, w, 36):
            for gy in range(0, h, 36):
                q.drawPoint(gx, gy)

        cx, cy, R = self._layout(w, h)

        # thin HUD frame lines
        q.setPen(QPen(_qc(pri_col, 55), 1))
        q.drawLine(10, 18, w - 10, 18)
        q.drawLine(10, 18, 10, h - 8)
        q.drawLine(w - 10, 18, w - 10, h - 8)
        q.setPen(QPen(_qc(pri_col, 45), 1))
        q.drawLine(QPointF(10, cy), QPointF(cx - R * 1.16, cy))
        q.drawLine(QPointF(cx + R * 1.16, cy), QPointF(w - 10, cy))
        q.drawLine(QPointF(cx, 18), QPointF(cx, cy - R * 1.16))

        # top-left bar (like the reference HUD)
        q.setPen(Qt.PenStyle.NoPen)
        q.setBrush(QBrush(_qc(pri_col, 170)))
        q.drawRect(QRectF(16, 9, 92, 5))
        q.setBrush(QBrush(_qc(pri_col, 90)))
        q.drawRect(QRectF(112, 9, 26, 5))
        q.drawRect(QRectF(142, 9, 6, 5))

        # corner brackets
        q.setPen(QPen(_qc(pri_col, 210), 1.6))
        m, L = 6, 16
        for (x, y, dx, dy) in ((m, m + 6, 1, 1), (w - m, m + 6, -1, 1), (m, h - m, 1, -1), (w - m, h - m, -1, -1)):
            q.drawLine(x, y, x + dx * L, y)
            q.drawLine(x, y, x, y + dy * L)

        q.setPen(_qc(pri_col, 130))
        q.setFont(QFont("Consolas", 7))
        q.drawText(QRectF(w - 110, 20, 96, 12), Qt.AlignmentFlag.AlignRight, "CORE // SYNC")
        q.end()
        self._bg = px

    # ---------------- persona emblem / custom avatar ----------------
    def _draw_persona_avatar(
        self,
        p: QPainter,
        cx: float,
        cy: float,
        r_in: float,
        pri_col: QColor,
        sec_col: QColor,
    ) -> None:
        key = self._persona_key

        custom_px = self._load_custom_avatar_pixmap(key)
        if custom_px is not None:
            clip_path = QPainterPath()
            clip_path.addEllipse(QRectF(cx - r_in + 2, cy - r_in + 2, (r_in - 2) * 2, (r_in - 2) * 2))
            p.save()
            p.setClipPath(clip_path)
            target_rect = QRectF(cx - r_in, cy - r_in, r_in * 2, r_in * 2)
            p.drawPixmap(target_rect.toRect(), custom_px)
            p.restore()
            return

        pulse = 0.0
        if self._status == "THINKING":
            pulse = 1.5 * math.sin(self._tick * 0.35)

        p.save()
        p.translate(cx, cy)

        if key == "jarvis":
            p.setPen(QPen(pri_col, 2.0))
            p.setBrush(QBrush(QColor(0, 0, 0, 0)))
            p.drawRoundedRect(QRectF(-r_in * 0.56, -r_in * 0.62, r_in * 1.12, r_in * 1.12), r_in * 0.32, r_in * 0.32)

            p.setBrush(QBrush(sec_col))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(QRectF(-r_in * 0.38, -r_in * 0.24, r_in * 0.76, r_in * 0.16), 3, 3)

            p.setPen(QPen(pri_col, 1.8))
            p.setBrush(QBrush(QColor(0, 0, 0, 0)))
            p.drawEllipse(QRectF(-r_in * 0.24, r_in * 0.04, r_in * 0.48, r_in * 0.48))
            p.setBrush(QBrush(sec_col))
            p.drawEllipse(QRectF(-r_in * 0.10 + pulse * 0.3, r_in * 0.18, r_in * 0.20, r_in * 0.20))

        elif key == "edith":
            p.setPen(QPen(pri_col, 2.2))
            p.setBrush(QBrush(QColor(0, 0, 0, 0)))
            p.drawRoundedRect(QRectF(-r_in * 0.68, -r_in * 0.28, r_in * 0.56, r_in * 0.42), 5, 5)
            p.drawRoundedRect(QRectF(r_in * 0.12, -r_in * 0.28, r_in * 0.56, r_in * 0.42), 5, 5)
            p.drawLine(QPointF(-r_in * 0.12, -r_in * 0.10), QPointF(r_in * 0.12, -r_in * 0.10))
            p.setPen(QPen(sec_col, 1.4))
            p.drawLine(QPointF(-r_in * 0.56, -r_in * 0.07), QPointF(-r_in * 0.24, -r_in * 0.07))
            p.drawLine(QPointF(r_in * 0.24, -r_in * 0.07), QPointF(r_in * 0.56, -r_in * 0.07))

        elif key == "friday":
            p.setPen(QPen(pri_col, 2.2))
            p.setBrush(QBrush(QColor(0, 0, 0, 0)))
            p.drawEllipse(QRectF(-r_in * 0.22, -r_in * 0.22, r_in * 0.44, r_in * 0.44))
            for idx, rad_mult in enumerate((0.42, 0.64)):
                rr = r_in * rad_mult + (pulse * 0.4 if idx == 1 else 0.0)
                p.setPen(QPen(sec_col if idx == 0 else pri_col, 1.8))
                p.drawArc(QRectF(-rr, -rr, rr * 2, rr * 2), 35 * 16, 110 * 16)
                p.drawArc(QRectF(-rr, -rr, rr * 2, rr * 2), 215 * 16, 110 * 16)

        elif key == "ultron":
            crown = QPolygonF([
                QPointF(-r_in * 0.52, -r_in * 0.58),
                QPointF(-r_in * 0.22, -r_in * 0.30),
                QPointF(0.0, -r_in * 0.68),
                QPointF(r_in * 0.22, -r_in * 0.30),
                QPointF(r_in * 0.52, -r_in * 0.58),
                QPointF(r_in * 0.44, r_in * 0.38),
                QPointF(0.0, r_in * 0.64),
                QPointF(-r_in * 0.44, r_in * 0.38),
            ])
            p.setPen(QPen(pri_col, 2.2))
            p.setBrush(QBrush(QColor(0, 0, 0, 0)))
            p.drawPolygon(crown)

            p.setBrush(QBrush(sec_col))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPolygon(QPolygonF([
                QPointF(-r_in * 0.32, -r_in * 0.14),
                QPointF(-r_in * 0.10, -r_in * 0.04),
                QPointF(-r_in * 0.30, r_in * 0.04),
            ]))
            p.drawPolygon(QPolygonF([
                QPointF(r_in * 0.32, -r_in * 0.14),
                QPointF(r_in * 0.10, -r_in * 0.04),
                QPointF(r_in * 0.30, r_in * 0.04),
            ]))

        elif key == "dummy":
            p.setPen(QPen(pri_col, 2.4))
            p.setBrush(QBrush(QColor(0, 0, 0, 0)))
            p.drawRoundedRect(QRectF(-r_in * 0.22, r_in * 0.08, r_in * 0.44, r_in * 0.46), 4, 4)
            p.drawLine(QPointF(0.0, r_in * 0.08), QPointF(0.0, -r_in * 0.18))
            p.setPen(QPen(sec_col, 2.4))
            p.drawArc(QRectF(-r_in * 0.58, -r_in * 0.56, r_in * 0.72, r_in * 0.72), 40 * 16, 150 * 16)
            p.drawArc(QRectF(-r_in * 0.14, -r_in * 0.56, r_in * 0.72, r_in * 0.72), -10 * 16, 150 * 16)

        else:
            hex_pts = QPolygonF()
            for k in range(6):
                ang = math.radians(60 * k - 30)
                hex_pts.append(QPointF(r_in * 0.56 * math.cos(ang), r_in * 0.56 * math.sin(ang)))
            p.setPen(QPen(pri_col, 2.2))
            p.setBrush(QBrush(QColor(0, 0, 0, 0)))
            p.drawPolygon(hex_pts)
            p.setPen(QPen(sec_col, 2.0))
            p.drawEllipse(QRectF(-r_in * 0.20, -r_in * 0.20, r_in * 0.40, r_in * 0.40))

        p.restore()

    # ---------------- 3D model renderer (flat-shaded, painter's algorithm) ----------------
    def _draw_model(self, p: QPainter, cx: float, cy: float, r: float, pri_col: QColor, sec_col: QColor, dim: float) -> None:
        mesh = self._mesh
        if mesh is None or not mesh.faces:
            return

        scale = r * 0.98 * self._zoom
        bob = math.sin(self._tick * 0.06) * 0.012
        cyaw, syaw = math.cos(self._yaw), math.sin(self._yaw)
        cpit, spit = math.cos(self._pitch), math.sin(self._pitch)

        # hologram base under the feet (additive glow)
        by = cy - (mesh.ymin * cpit + bob) * scale
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        for k, f in enumerate((0.78, 0.58, 0.38)):
            rx = scale * f
            ry = rx * (0.18 + 0.5 * spit)
            p.setPen(QPen(_qc(sec_col if k == 1 else pri_col, (150 - 35 * k) * dim), 1.4 if k == 0 else 1.0))
            if k == 0:
                p.setBrush(QBrush(_qc(pri_col, 28 * dim)))
            else:
                p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(cx, by), rx, ry)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        # transform vertices
        pts: list[QPointF] = []
        zs: list[float] = []
        for (x, y, z) in mesh.verts:
            x1 = x * cyaw + z * syaw
            z1 = -x * syaw + z * cyaw
            y2 = y * cpit - z1 * spit + bob
            z2 = y * spit + z1 * cpit
            k = 1.0 / (1.0 - z2 * 0.16)
            pts.append(QPointF(cx + x1 * scale * k, cy - y2 * scale * k))
            zs.append(z2)

        # cull + depth sort
        lx, ly, lz = -0.45, 0.65, 0.62
        draw = []
        for (ids, ckey, two, bias), (nx, ny, nz) in zip(mesh.faces, mesh.normals):
            nx1 = nx * cyaw + nz * syaw
            nz1 = -nx * syaw + nz * cyaw
            ny2 = ny * cpit - nz1 * spit
            nz2 = ny * spit + nz1 * cpit
            if not two and nz2 <= mesh.cull:
                continue
            depth = sum(zs[i] for i in ids) / len(ids) + bias
            dot = nx1 * lx + ny2 * ly + nz2 * lz
            draw.append((depth, ids, ckey, two, nz2, dot))
        draw.sort(key=lambda d: d[0])

        pr, pg, pb = pri_col.red(), pri_col.green(), pri_col.blue()
        seamless = mesh.seamless
        if not seamless:
            p.setPen(QPen(_qc(pri_col, 85 * dim), 0.6))
        for depth, ids, ckey, two, nz2, dot in draw:
            base = ckey if isinstance(ckey, tuple) else (pr, pg, pb)
            inten = 0.40 + 0.60 * (abs(dot) if two else max(0.0, dot))
            rim = (1.0 - min(1.0, abs(nz2))) ** 2 * 0.30
            col = QColor(
                min(255, int(base[0] * inten * dim + pr * rim)),
                min(255, int(base[1] * inten * dim + pg * rim)),
                min(255, int(base[2] * inten * dim + pb * rim)),
            )
            if seamless:
                p.setPen(QPen(col, 0.8))
            p.setBrush(col)
            p.drawPolygon(QPolygonF([pts[i] for i in ids]))

        # hologram scan-line
        sy = cy + math.sin(self._tick * 0.05) * scale * 0.95
        half = scale * 0.55
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        p.setPen(QPen(_qc(sec_col, 90 * dim), 1.2))
        p.drawLine(QPointF(cx - half, sy), QPointF(cx + half, sy))
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        # captions
        p.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        p.setPen(_qc(sec_col, 210))
        p.drawText(QRectF(16, 42, 220, 12), Qt.AlignmentFlag.AlignLeft, f"3D // {mesh.name.upper()}")
        line_y = 54
        if mesh.credit:
            p.setPen(_qc(pri_col, 150))
            p.drawText(QRectF(16, line_y, 240, 12), Qt.AlignmentFlag.AlignLeft, f"MODEL BY {mesh.credit.upper()}")
            line_y += 12
        p.setPen(_qc(pri_col, 130))
        p.drawText(QRectF(16, line_y, 240, 12), Qt.AlignmentFlag.AlignLeft, "DRAG: ROTATE  WHEEL: ZOOM")

    # ---------------- main paint ----------------
    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        pri_col = QColor("#5c3a44") if self._muted else QColor(self._pri)
        sec_col = QColor("#7a4f5c") if self._muted else QColor(self._sec)
        hot_col = _mix(sec_col, QColor("#ffffff"), 0.55)

        self._ensure_static(w, h, pri_col)
        p.drawPixmap(0, 0, self._bg)

        cx, cy, R = self._layout(w, h)
        r = R * 0.84
        thinking = self._status in ("THINKING", "SPEAKING", "CONNECTING")
        speaking = self._status == "SPEAKING"
        pulse = 1.0 + (
            0.06 + 0.18 * self._audio_level
            if (speaking or (self._status == "LISTENING" and self._audio_level > 0.05))
            else (0.10 * math.sin(self._tick * 0.30) if thinking else 0.04 * math.sin(self._tick * 0.08))
        )
        dim = 0.55 if self._muted else 1.0
        ctr = QPointF(cx, cy)

        # ---- additive glow layers ----
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)

        halo = QRadialGradient(ctr, R * 1.22)
        halo.setColorAt(0.0, _qc(pri_col, 70 * dim))
        halo.setColorAt(0.6, _qc(pri_col, 26 * dim))
        halo.setColorAt(1.0, _qc(pri_col, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(halo))
        p.drawEllipse(ctr, R * 1.22, R * 1.22)

        # orbit arcs
        for (rf, ratio, tilt, start, span, width, dash, speed, alpha) in _ORB_RINGS:
            rx = r * rf
            ry = rx * ratio
            pen = QPen(_qc(sec_col if rf < 0.9 else pri_col, alpha * dim), width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            if dash:
                pen.setDashPattern([float(dash[0]), float(dash[1])])
            p.save()
            p.translate(cx, cy)
            p.rotate(tilt + self._angle * speed * (1.6 if thinking else 1.0) * 0.5)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawArc(QRectF(-rx, -ry, rx * 2, ry * 2), int(start * 16), int(span * 16))
            p.restore()

        # outer tick ring
        p.save()
        p.translate(cx, cy)
        p.rotate(-self._angle * 0.12)
        for i in range(90):
            a = math.radians(i * 4)
            long_tick = i % 5 == 0
            r0 = r * 1.16
            r1 = r * (1.23 if long_tick else 1.19)
            p.setPen(QPen(_qc(pri_col, (150 if long_tick else 70) * dim), 1.0))
            p.drawLine(QPointF(r0 * math.cos(a), r0 * math.sin(a)), QPointF(r1 * math.cos(a), r1 * math.sin(a)))
        p.restore()

        # rotating particle sphere
        yaw = math.radians(self._angle) * 0.5 * (1.5 if thinking else 1.0)
        cyw, syw = math.cos(yaw), math.sin(yaw)
        tilt_a = 0.45
        ct, st = math.cos(tilt_a), math.sin(tilt_a)
        for (x, y, z, size, ph) in self._particles:
            x1 = x * cyw + z * syw
            z1 = -x * syw + z * cyw
            y2 = y * ct - z1 * st
            z2 = y * st + z1 * ct
            depth = (z2 + 1.0) * 0.5
            tw = 0.65 + 0.35 * math.sin(self._tick * 0.12 + ph)
            a = (45 + 200 * depth) * tw * dim
            col = _mix(pri_col, hot_col, 0.8 if (size > 2.0 and depth > 0.6) else depth * 0.35)
            pen = QPen(_qc(col, a), size * (0.7 + 0.5 * depth))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.drawPoint(QPointF(cx + x1 * r, cy + y2 * r))

        # swirling filaments around the core
        t = self._tick
        spin = 0.05 if thinking else 0.02
        for k in range(8):
            path = QPainterPath()
            base = k * (2 * math.pi / 8) + t * spin * (1 + 0.15 * k)
            reach = r * (0.36 + 0.10 * math.sin(k * 1.7 + t * 0.04))
            for i in range(24):
                u = i / 23.0
                rad = reach * (u ** 0.85) * pulse
                ang = base + u * 3.3 + 0.35 * math.sin(t * 0.05 + k)
                pt = QPointF(cx + rad * math.cos(ang), cy + rad * math.sin(ang) * 0.88)
                if i == 0:
                    path.moveTo(pt)
                else:
                    path.lineTo(pt)
            p.setPen(QPen(_qc(sec_col if k % 2 == 0 else hot_col, (150 if k % 2 == 0 else 110) * dim), 1.2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)

        # bright core glow (persona colour)
        cr = r * 0.50 * pulse
        core = QRadialGradient(ctr, cr)
        core.setColorAt(0.0, _qc(hot_col, 255 * dim))
        core.setColorAt(0.20, _qc(sec_col, 235 * dim))
        core.setColorAt(0.55, _qc(pri_col, 120 * dim))
        core.setColorAt(1.0, _qc(pri_col, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(core))
        p.drawEllipse(ctr, cr, cr)

        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        # ---- 3D model / avatar / emblem ----
        has_custom = self._load_custom_avatar_pixmap(self._persona_key) is not None
        if self._mesh is not None:
            self._draw_model(p, cx, cy, r, pri_col, sec_col, dim)
        elif has_custom:
            ar = r * 0.34
            p.setBrush(QBrush(QColor("#0d0f15")))
            p.setPen(QPen(pri_col, 2))
            p.drawEllipse(ctr, ar, ar)
            self._draw_persona_avatar(p, cx, cy, ar * 0.98, pri_col, sec_col)
        else:
            dark_pri = _mix(pri_col, QColor("#000000"), 0.82)
            dark_sec = _mix(sec_col, QColor("#000000"), 0.70)
            self._draw_persona_avatar(p, cx, cy - 2.0, r * 0.30, dark_pri, dark_sec)

        # label pill (top-left while a 3D model is shown, so it never covers the model)
        p.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        pill_w, pill_h = 88.0, 15.0
        if self._mesh is not None:
            pill = QRectF(16, 24, pill_w, pill_h)
        else:
            pill = QRectF(cx - pill_w / 2, cy + r * 0.60, pill_w, pill_h)
        p.setPen(QPen(_qc(pri_col, 160), 1))
        p.setBrush(QBrush(QColor(7, 8, 12, 190)))
        p.drawRoundedRect(pill, 7, 7)
        p.setPen(QColor("#f0f4fc"))
        p.drawText(pill, Qt.AlignmentFlag.AlignCenter, self._label)

        # status text
        sy = cy + R + 6
        if self._muted:
            status_txt = "⊘  MIC MUTED [F4]"
        elif self._status == "SPEAKING":
            status_txt = "🔊  LIVE SPEAKING"
        elif self._status == "LISTENING":
            status_txt = "🎙  LIVE LISTENING"
        else:
            status_txt = f"●  {self._status}"
        p.setPen(pri_col)
        p.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        p.drawText(QRectF(0, sy, w, 20), Qt.AlignmentFlag.AlignCenter, status_txt)

        # waveform bars
        wy = sy + 22
        bars = 32
        bw = 7
        wx0 = (w - bars * bw) / 2.0
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(bars):
            if self._muted:
                bh = 2
            elif self._audio_level > 0.03:
                # Real-time PCM audio reactive waveform (just like main.py's live voice HUD)
                env = 1.0 - abs((i - bars / 2.0) / (bars / 2.0)) * 0.55
                wave = 0.55 + 0.45 * abs(math.sin(self._tick * 0.45 + i * 0.55))
                bh = max(3, min(18, int(3 + 16 * self._audio_level * env * wave)))
            elif thinking:
                bh = 4 + int(12 * abs(math.sin(self._tick * 0.25 + i * 0.45)))
            else:
                bh = 3 + int(5 * abs(math.sin(self._tick * 0.10 + i * 0.5)))
            p.setBrush(QBrush(pri_col if bh > 7 else QColor("#262b3d")))
            p.drawRoundedRect(QRectF(wx0 + i * bw, wy + 16 - bh, bw - 2, bh), 1.5, 1.5)

        p.end()


class FileDropBox(QFrame):
    file_loaded = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(64)
        self._file_path: str | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        self.lbl = QLabel("📎 Drop a file here or Click to Upload\n(.txt, .py, .md, .json, .csv)")
        self.lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl.setStyleSheet("color: #9aa8c0; font-size: 10px; border: none;")
        lay.addWidget(self.lbl)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if Path(path).is_file():
                self.set_file(path)

    def mousePressEvent(self, _event) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Study / Code File",
            str(Path.home()),
            "Text & Code Files (*.txt *.py *.md *.json *.csv *.log);;All Files (*.*)",
        )
        if path:
            self.set_file(path)

    def set_file(self, path: str) -> None:
        self._file_path = path
        p = Path(path)
        self.lbl.setText(f"✅ Loaded: {p.name}\n(Ask any question about this file below)")
        self.file_loaded.emit(path)

    def clear_file(self) -> None:
        self._file_path = None
        self.lbl.setText("📎 Drop a file here or Click to Upload\n(.txt, .py, .md, .json, .csv)")


class MobileRemoteOverlay(QFrame):
    """Popup QR + IP Card shown ONLY when user clicks 'MOBILE REMOTE' button."""

    def __init__(self, bridge: MobileRemoteBridge, accent_hex: str, parent=None):
        super().__init__(parent)
        self.bridge = bridge
        self.accent_hex = accent_hex
        self.setObjectName("Panel")
        self.setFixedSize(360, 390)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(8)

        title = QLabel("📱 MOBILE REMOTE CONTROL")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {accent_hex}; border: none;")
        lay.addWidget(title)

        sub = QLabel("Scan with Phone Camera / Google Lens (Same Wi-Fi)\nto control this Desktop App remotely:")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setWordWrap(True)
        sub.setStyleSheet("color: #9aa8c0; font-size: 11px; border: none;")
        lay.addWidget(sub)

        self.qr_lbl = QLabel()
        self.qr_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.qr_lbl.setFixedHeight(195)
        self.qr_lbl.setStyleSheet("border: none;")
        lay.addWidget(self.qr_lbl)

        self.url_lbl = QLabel()
        self.url_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.url_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.url_lbl.setStyleSheet(f"color: {accent_hex}; font-family: Consolas; font-size: 11px; border: none;")
        lay.addWidget(self.url_lbl)

        btn_row = QHBoxLayout()
        btn_new = QPushButton("NEW QR TOKEN")
        btn_new.clicked.connect(self._refresh_qr)
        btn_close = QPushButton("CLOSE ✕")
        btn_close.clicked.connect(self.hide)
        btn_row.addWidget(btn_new)
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)

        self._render_qr(self.bridge.get_remote_url())

    def _refresh_qr(self) -> None:
        new_url = self.bridge.regenerate_token()
        self._render_qr(new_url)

    def _render_qr(self, url: str) -> None:
        self.url_lbl.setText(url)
        try:
            import qrcode

            qr = qrcode.QRCode(box_size=5, border=2)
            qr.add_data(url)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            px = QPixmap()
            px.loadFromData(buf.getvalue())
            self.qr_lbl.setPixmap(px.scaled(185, 185, Qt.AspectRatioMode.KeepAspectRatio))
        except Exception:
            self.qr_lbl.setText("Install qrcode & pillow for QR image:\npip install qrcode pillow")


class VisionDesktopWindow(QMainWindow):
    _reply_ready = pyqtSignal(str, str)
    _live_log_sig = pyqtSignal(str, str)
    _live_state_sig = pyqtSignal(str)
    _live_level_sig = pyqtSignal(float)
    _live_toggle_ui_sig = pyqtSignal(bool)
    _remote_control_sig = pyqtSignal(dict)
    _remote_prompt_sig = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.vision = VisionCore()
        self.assistant = "edith"
        self.mode = "study"
        self.provider = "gemini"
        self.language = "tamil"
        self._muted = False
        self._loaded_file_context = ""
        self._loaded_file_name = ""
        self._remote_popup: Optional[MobileRemoteOverlay] = None

        # 3D model state (saved in vision_ui_prefs.json)
        self.models = ModelLibrary()
        self._last_mesh: Optional[Mesh3D] = None
        self.view3d = True
        self.model_name = DEFAULT_MODEL_NAME
        self._load_model_prefs()

        self.setWindowTitle("VISION — Desktop AI Application")
        self.resize(1160, 740)
        self.setMinimumSize(940, 620)

        self.live_voice = GeminiLiveVoiceEngine(self)

        # Build the UI first: signal connections below need self.hud / self.log_box to exist.
        self._build_ui()

        self._reply_ready.connect(self._on_reply_ready)
        self._live_log_sig.connect(self._append_log)
        self._live_state_sig.connect(self.hud.set_status)
        self._live_level_sig.connect(self.hud.set_audio_level)
        self._live_toggle_ui_sig.connect(self._sync_live_voice_btn)
        self._remote_control_sig.connect(self._handle_remote_control)
        self._remote_prompt_sig.connect(self._handle_remote_prompt)

        self._build_floating_drawers()
        self._setup_shortcuts()
        self._apply_theme()
        self._apply_model()

        # Start background Mobile Remote listener (ONLY for phone remote control)
        self.remote_bridge = MobileRemoteBridge(self, port=5055)
        self.remote_bridge.start()

        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._tick_clock)
        self._clock_timer.start(1000)
        self._tick_clock()

        self._sys_timer = QTimer(self)
        self._sys_timer.timeout.connect(self._tick_metrics)
        self._sys_timer.start(2000)
        self._tick_metrics()

        self._append_log("SYSTEM", "VISION Desktop Online (Gemini Live Native Voice ready — NO TTS). Click '🎙 LIVE VOICE [F5]' to talk directly!")

    def _theme(self) -> dict[str, str]:
        return PERSONA_THEMES.get(self.assistant, PERSONA_THEMES["edith"])

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence("F11"), self).activated.connect(self._toggle_fullscreen)
        QShortcut(QKeySequence("F5"), self).activated.connect(self._toggle_live_voice)
        QShortcut(QKeySequence("F4"), self).activated.connect(self._toggle_mute)
        QShortcut(QKeySequence("F3"), self).activated.connect(self._toggle_view3d)
        QShortcut(QKeySequence("F2"), self).activated.connect(self._next_model)

    # ---------------- 3D model controls ----------------
    def _load_model_prefs(self) -> None:
        try:
            data = json.loads(PREFS_FILE.read_text(encoding="utf-8"))
            self.view3d = bool(data.get("view3d", True))
            name = data.get("model", DEFAULT_MODEL_NAME)
            if name in self.models.names():
                self.model_name = name
        except Exception:
            pass

    def _save_model_prefs(self) -> None:
        try:
            PREFS_FILE.write_text(
                json.dumps({"view3d": self.view3d, "model": self.model_name}, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _apply_model(self) -> None:
        mesh = None
        if self.view3d:
            try:
                mesh = self.models.load(self.model_name)
            except Exception as exc:
                self._append_log("ERR", f"3D model load failed ({self.model_name}): {exc}")
                self.view3d = False
        self.hud.set_model(mesh)
        if mesh is not None and mesh is not self._last_mesh and mesh.credit:
            self._append_log("SYS", f"3D model: {mesh.name} — by {mesh.credit}")
        self._last_mesh = mesh
        self.btn_view3d.setChecked(self.view3d)
        self.btn_view3d.setText("🧊 3D: ON" if self.view3d else "🧊 3D: OFF")
        self._save_model_prefs()

    def _toggle_view3d(self) -> None:
        self.view3d = not self.view3d
        self._apply_model()
        if self.view3d:
            self._append_log("SYS", f"3D model view ON — {self.model_name}.")
        else:
            self._append_log("SYS", "3D model view OFF (classic HUD orb).")

    def _next_model(self) -> None:
        names = self.models.names()
        if not names:
            self._append_log("ERR", f"3D model எதுவும் இல்லை. .glb / .obj file-ஐ '{MODELS_DIR.name}' folder-ல் வைக்கவும்.")
            return
        try:
            idx = names.index(self.model_name)
        except ValueError:
            idx = -1
        self.model_name = names[(idx + 1) % len(names)]
        self.view3d = True
        self._apply_model()
        self._append_log("SYS", f"3D model: {self.model_name}")

    def _import_obj_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a 3D model (.glb / .obj)",
            str(Path.home()),
            "3D Model (*.glb *.obj);;All Files (*.*)",
        )
        if not path:
            return
        try:
            name = self.models.import_file(path)
            self.model_name = name
            self.view3d = True
            self._apply_model()
            self._append_log("SYS", f"3D model '{name}' imported -> {MODELS_DIR}")
        except Exception as exc:
            self._append_log("ERR", f"3D import failed: {exc}")

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _toggle_mute(self) -> None:
        self._muted = not self._muted
        self.hud.set_muted(self._muted)
        self.btn_mute.setText("🔇 MIC MUTED [F4]" if self._muted else "🎙 MIC ACTIVE [F4]")
        self._append_log("SYS", "Microphone muted." if self._muted else "Microphone active.")

    def _toggle_live_voice(self) -> None:
        if self.live_voice.running:
            self.live_voice.stop()
        else:
            self.live_voice.start()

    def _sync_live_voice_btn(self, active: bool) -> None:
        self.btn_live_voice.setChecked(active)
        if active:
            v_name = PERSONA_LIVE_VOICES.get(self.assistant, "Kore")
            self.btn_live_voice.setText(f"🟢 LIVE VOICE: ON ({v_name}) [F5]")
        else:
            self.btn_live_voice.setText("🎙 LIVE VOICE: OFF [F5]")

    def _interrupt_live_voice(self) -> None:
        if self.live_voice.running:
            self.live_voice.interrupt()

    def _build_ui(self) -> None:
        self.setStyleSheet("""
            QMainWindow { background: #07080c; color: #f0f4fc; }
            QWidget { background: #07080c; color: #f0f4fc; font-family: 'Segoe UI'; }
            QLabel { background: transparent; }
            QFrame#Panel {
                background: #0d0f15;
                border: 1px solid #222636;
                border-radius: 6px;
            }
            QLineEdit, QTextEdit {
                background: #0b0d13;
                color: #f0f4fc;
                border: 1px solid #262b3d;
                border-radius: 4px;
                padding: 6px 8px;
                font-family: 'Consolas';
            }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Top Header Bar
        header = QFrame()
        header.setObjectName("Panel")
        header.setFixedHeight(54)
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(12, 4, 12, 4)

        self.btn_setup_drawer = QPushButton("⚙ SETUP & KEYS")
        self.btn_setup_drawer.setCheckable(True)
        self.btn_setup_drawer.clicked.connect(self._toggle_setup_drawer)
        h_lay.addWidget(self.btn_setup_drawer)

        self.btn_ctrl_drawer = QPushButton("🎛 CONTROLS")
        self.btn_ctrl_drawer.setCheckable(True)
        self.btn_ctrl_drawer.clicked.connect(self._toggle_ctrl_drawer)
        h_lay.addWidget(self.btn_ctrl_drawer)

        self.btn_mobile_remote = QPushButton("📱 MOBILE REMOTE (QR)")
        self.btn_mobile_remote.clicked.connect(self._open_mobile_remote_popup)
        h_lay.addWidget(self.btn_mobile_remote)

        self.btn_fullscreen = QPushButton("⛶ FULLSCREEN [F11]")
        self.btn_fullscreen.clicked.connect(self._toggle_fullscreen)
        h_lay.addWidget(self.btn_fullscreen)

        h_lay.addStretch()

        title_col = QVBoxLayout()
        title_col.setSpacing(1)
        self.lbl_main_title = QLabel("VISION")
        self.lbl_main_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_main_title.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        title_col.addWidget(self.lbl_main_title)

        self.lbl_sub_title = QLabel("MULTI-PERSONA DESKTOP AI")
        self.lbl_sub_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_sub_title.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        title_col.addWidget(self.lbl_sub_title)
        h_lay.addLayout(title_col)

        h_lay.addStretch()

        clock_col = QVBoxLayout()
        clock_col.setSpacing(1)
        self.lbl_clock = QLabel("00:00:00")
        self.lbl_clock.setFont(QFont("Consolas", 13, QFont.Weight.Bold))
        self.lbl_clock.setAlignment(Qt.AlignmentFlag.AlignRight)
        clock_col.addWidget(self.lbl_clock)

        self.lbl_date = QLabel("")
        self.lbl_date.setFont(QFont("Consolas", 8))
        self.lbl_date.setStyleSheet("color: #6b778d;")
        self.lbl_date.setAlignment(Qt.AlignmentFlag.AlignRight)
        clock_col.addWidget(self.lbl_date)
        h_lay.addLayout(clock_col)

        root.addWidget(header)

        # Main 3-Column Body
        body = QHBoxLayout()
        body.setSpacing(8)

        # 1. Left Column: System Telemetry & Desktop Utilities
        left_panel = QFrame()
        left_panel.setObjectName("Panel")
        left_panel.setFixedWidth(185)
        l_lay = QVBoxLayout(left_panel)
        l_lay.setContentsMargins(8, 10, 8, 10)
        l_lay.setSpacing(8)

        self.lbl_tel_hdr = QLabel("◈ SYS TELEMETRY")
        self.lbl_tel_hdr.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        l_lay.addWidget(self.lbl_tel_hdr)

        self.bar_cpu = MetricBar("CPU")
        self.bar_mem = MetricBar("MEM")
        self.bar_net = MetricBar("NET")
        l_lay.addWidget(self.bar_cpu)
        l_lay.addWidget(self.bar_mem)
        l_lay.addWidget(self.bar_net)

        self.lbl_uptime = QLabel("UPTIME: --:--")
        self.lbl_uptime.setFont(QFont("Consolas", 8))
        self.lbl_uptime.setStyleSheet("color: #9aa8c0;")
        l_lay.addWidget(self.lbl_uptime)

        l_lay.addStretch()

        self.btn_shortcut_left = QPushButton("⊞ DESKTOP SHORTCUT")
        self.btn_shortcut_left.clicked.connect(self._make_shortcut)
        l_lay.addWidget(self.btn_shortcut_left)

        self.btn_export_notes = QPushButton("💾 EXPORT NOTES")
        self.btn_export_notes.clicked.connect(self._export_study_notes)
        l_lay.addWidget(self.btn_export_notes)

        body.addWidget(left_panel)

        # 2. Center Column: HUD Core + 3D model switch + Persona/Mode/Language Controls
        center_panel = QFrame()
        center_panel.setObjectName("Panel")
        center_panel.setFixedWidth(365)
        c_lay = QVBoxLayout(center_panel)
        c_lay.setContentsMargins(10, 10, 10, 10)
        c_lay.setSpacing(8)

        self.hud = HudCoreWidget()
        c_lay.addWidget(self.hud, stretch=1)

        view_row = QHBoxLayout()
        view_row.setSpacing(5)
        self.btn_view3d = QPushButton("🧊 3D: ON")
        self.btn_view3d.setCheckable(True)
        self.btn_view3d.clicked.connect(self._toggle_view3d)
        view_row.addWidget(self.btn_view3d)
        self.btn_next_model = QPushButton("⟳ NEXT MODEL")
        self.btn_next_model.clicked.connect(self._next_model)
        view_row.addWidget(self.btn_next_model)
        self.btn_load_obj = QPushButton("📂 LOAD 3D FILE")
        self.btn_load_obj.clicked.connect(self._import_obj_model)
        view_row.addWidget(self.btn_load_obj)
        c_lay.addLayout(view_row)

        self.lbl_asst_hdr = QLabel("▸ ASSISTANT PERSONA")
        self.lbl_asst_hdr.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        c_lay.addWidget(self.lbl_asst_hdr)

        asst_grid = QGridLayout()
        asst_grid.setSpacing(5)
        self.asst_btns: dict[str, QPushButton] = {}
        for idx, (key, info) in enumerate(PERSONAS.items()):
            b = QPushButton(info["name"])
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, k=key: self._set_assistant(k))
            self.asst_btns[key] = b
            asst_grid.addWidget(b, idx // 3, idx % 3)
        c_lay.addLayout(asst_grid)

        self.lbl_mode_hdr = QLabel("▸ OPERATION MODE")
        self.lbl_mode_hdr.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        c_lay.addWidget(self.lbl_mode_hdr)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(5)
        self.mode_btns: dict[str, QPushButton] = {}
        for m_key, m_label in [("normal", "NORMAL"), ("study", "STUDY"), ("extra_work", "EXTRA WORK")]:
            b = QPushButton(m_label)
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, k=m_key: self._set_mode(k))
            self.mode_btns[m_key] = b
            mode_row.addWidget(b)
        c_lay.addLayout(mode_row)

        self.lbl_lang_hdr = QLabel("▸ LANGUAGE")
        self.lbl_lang_hdr.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        c_lay.addWidget(self.lbl_lang_hdr)

        lang_row = QHBoxLayout()
        lang_row.setSpacing(5)
        self.btn_lang = QPushButton(f"LANG: {self.language.upper()}")
        self.btn_lang.clicked.connect(self._toggle_language)
        lang_row.addWidget(self.btn_lang)
        c_lay.addLayout(lang_row)

        body.addWidget(center_panel)

        # 3. Right Column: Comm Link + File Drop + Quick Study Prompts + Input
        right_panel = QFrame()
        right_panel.setObjectName("Panel")
        r_lay = QVBoxLayout(right_panel)
        r_lay.setContentsMargins(10, 10, 10, 10)
        r_lay.setSpacing(7)

        comm_hdr_row = QHBoxLayout()
        self.lbl_comm_hdr = QLabel("▸ COMM LINK")
        self.lbl_comm_hdr.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        comm_hdr_row.addWidget(self.lbl_comm_hdr)
        comm_hdr_row.addStretch()

        self.btn_clear_log = QPushButton("CLEAR LOG")
        self.btn_clear_log.clicked.connect(lambda: self.log_box.clear())
        comm_hdr_row.addWidget(self.btn_clear_log)
        r_lay.addLayout(comm_hdr_row)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        r_lay.addWidget(self.log_box, stretch=1)

        self.file_drop = FileDropBox()
        self.file_drop.file_loaded.connect(self._on_file_loaded)
        r_lay.addWidget(self.file_drop)

        quick_row = QHBoxLayout()
        quick_row.setSpacing(5)
        for label, prompt_prefix in [
            ("📝 SUMMARIZE", "Summarize this clearly in bullet points:\n"),
            ("❓ QUIZ ME", "Create 5 practice quiz questions with answers on:\n"),
            ("💡 EXPLAIN SIMPLE", "Explain this simply with an easy example:\n"),
            ("🛠 DEBUG CODE", "Find bugs and improve this code:\n"),
        ]:
            qb = QPushButton(label)
            qb.clicked.connect(lambda _=False, p=prompt_prefix: self._insert_quick_prompt(p))
            quick_row.addWidget(qb)
        r_lay.addLayout(quick_row)

        in_row = QHBoxLayout()
        in_row.setSpacing(6)
        self.prompt_input = QLineEdit()
        self.prompt_input.setFixedHeight(34)
        self.prompt_input.setPlaceholderText("Ask VISION or enter a study command...")
        self.prompt_input.returnPressed.connect(self._send_prompt)
        in_row.addWidget(self.prompt_input)

        self.btn_send = QPushButton("SEND ▸")
        self.btn_send.setFixedSize(92, 34)
        self.btn_send.clicked.connect(self._send_prompt)
        in_row.addWidget(self.btn_send)
        r_lay.addLayout(in_row)

        voice_row = QHBoxLayout()
        voice_row.setSpacing(6)
        self.btn_live_voice = QPushButton("🎙 LIVE VOICE: OFF [F5]")
        self.btn_live_voice.setCheckable(True)
        self.btn_live_voice.setFixedHeight(32)
        self.btn_live_voice.clicked.connect(self._toggle_live_voice)
        voice_row.addWidget(self.btn_live_voice, stretch=2)

        self.btn_mute = QPushButton("🎙 MIC ACTIVE [F4]")
        self.btn_mute.setFixedHeight(32)
        self.btn_mute.clicked.connect(self._toggle_mute)
        voice_row.addWidget(self.btn_mute, stretch=1)

        self.btn_interrupt = QPushButton("✋ INTERRUPT")
        self.btn_interrupt.setFixedHeight(32)
        self.btn_interrupt.clicked.connect(self._interrupt_live_voice)
        voice_row.addWidget(self.btn_interrupt)
        r_lay.addLayout(voice_row)

        self.lbl_creator = QLabel("by Jagadeesh")
        creator_font = QFont("Consolas", 9)
        creator_font.setStyleHint(QFont.StyleHint.Monospace)
        self.lbl_creator.setFont(creator_font)
        self.lbl_creator.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_creator.setStyleSheet(
            "color: #6b778d; font-family: 'Consolas', 'Courier New', monospace; padding-right: 2px;"
        )
        r_lay.addWidget(self.lbl_creator)

        body.addWidget(right_panel, stretch=1)
        root.addLayout(body, stretch=1)

        footer = QLabel("[F2] Next 3D Model  ·  [F3] 3D On/Off  ·  [F4] Mic Mute  ·  [F5] Live Voice  ·  [F11] Fullscreen")
        footer.setFont(QFont("Consolas", 8))
        footer.setStyleSheet("color: #6b778d; padding-left: 4px;")
        root.addWidget(footer)

    def _build_floating_drawers(self) -> None:
        cw = self.centralWidget()

        self.setup_drawer = QFrame(cw)
        self.setup_drawer.setObjectName("Panel")
        self.setup_drawer.setFixedWidth(320)
        self.setup_drawer.hide()
        s_lay = QVBoxLayout(self.setup_drawer)
        s_lay.setContentsMargins(12, 12, 12, 12)
        s_lay.setSpacing(8)

        s_lay.addWidget(QLabel("◈ API KEYS & WORKSTATION SETUP"))

        self.gemini_key_input = QLineEdit()
        self.gemini_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.gemini_key_input.setPlaceholderText("Google Gemini API Key")
        self.gemini_key_input.setText(self.vision.gemini_api_key)
        s_lay.addWidget(self.gemini_key_input)

        self.openrouter_key_input = QLineEdit()
        self.openrouter_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.openrouter_key_input.setPlaceholderText("OpenRouter API Key")
        self.openrouter_key_input.setText(self.vision.openrouter_api_key)
        s_lay.addWidget(self.openrouter_key_input)

        self.btn_save_keys = QPushButton("💾 SAVE API KEYS LOCALLY")
        self.btn_save_keys.clicked.connect(self._save_keys)
        s_lay.addWidget(self.btn_save_keys)

        self.lbl_key_status = QLabel("")
        self.lbl_key_status.setStyleSheet("color: #9aa8c0; font-size: 10px;")
        s_lay.addWidget(self.lbl_key_status)

        self.btn_set_avatar = QPushButton("🖼 SET AVATAR IMAGE")
        self.btn_set_avatar.clicked.connect(self._pick_persona_avatar)
        s_lay.addWidget(self.btn_set_avatar)

        self.btn_reset_avatar = QPushButton("↺ RESET AVATAR")
        self.btn_reset_avatar.clicked.connect(self._reset_persona_avatar)
        s_lay.addWidget(self.btn_reset_avatar)

        self.btn_import_obj = QPushButton("🧊 IMPORT 3D MODEL (.GLB/.OBJ)")
        self.btn_import_obj.clicked.connect(self._import_obj_model)
        s_lay.addWidget(self.btn_import_obj)

        self.btn_shortcut = QPushButton("⊞ CREATE DESKTOP SHORTCUT (DARK ICON)")
        self.btn_shortcut.clicked.connect(self._make_shortcut)
        s_lay.addWidget(self.btn_shortcut)

        self.btn_autostart = QPushButton("◉ AUTO-START ON BOOT: OFF")
        self.btn_autostart.clicked.connect(self._toggle_autostart)
        s_lay.addWidget(self.btn_autostart)
        self._refresh_autostart_label()

        btn_close_setup = QPushButton("CLOSE ✕")
        btn_close_setup.clicked.connect(lambda: self._toggle_setup_drawer(False))
        s_lay.addWidget(btn_close_setup)
        self.setup_drawer.adjustSize()

        self.ctrl_drawer = QFrame(cw)
        self.ctrl_drawer.setObjectName("Panel")
        self.ctrl_drawer.setFixedWidth(250)
        self.ctrl_drawer.hide()
        c_lay = QVBoxLayout(self.ctrl_drawer)
        c_lay.setContentsMargins(12, 12, 12, 12)
        c_lay.setSpacing(8)

        c_lay.addWidget(QLabel("◈ QUICK WORKSTATION CONTROLS"))

        b_clear_file = QPushButton("🗑 UNLOAD ATTACHED FILE")
        b_clear_file.clicked.connect(self._unload_file)
        c_lay.addWidget(b_clear_file)

        b_3d = QPushButton("🧊 TOGGLE 3D MODEL [F3]")
        b_3d.clicked.connect(self._toggle_view3d)
        c_lay.addWidget(b_3d)

        b_next3d = QPushButton("⟳ NEXT 3D MODEL [F2]")
        b_next3d.clicked.connect(self._next_model)
        c_lay.addWidget(b_next3d)

        b_fs = QPushButton("⛶ TOGGLE FULLSCREEN [F11]")
        b_fs.clicked.connect(self._toggle_fullscreen)
        c_lay.addWidget(b_fs)

        b_close_ctrl = QPushButton("CLOSE ✕")
        b_close_ctrl.clicked.connect(lambda: self._toggle_ctrl_drawer(False))
        c_lay.addWidget(b_close_ctrl)
        self.ctrl_drawer.adjustSize()

    def _refresh_avatar_buttons(self) -> None:
        p_name = PERSONAS[self.assistant]["name"]
        self.btn_set_avatar.setText(f"🖼 SET {p_name} AVATAR")
        self.btn_reset_avatar.setText(f"↺ RESET {p_name} AVATAR")
        self.setup_drawer.adjustSize()

    def _pick_persona_avatar(self) -> None:
        key = self.assistant
        p_name = PERSONAS[key]["name"]
        src_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select Avatar Image for {p_name}",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.webp);;All Files (*.*)",
        )
        if not src_path:
            return
        px = QPixmap(src_path)
        if px.isNull():
            self._append_log("ERR", f"Invalid image file selected for {p_name}.")
            return
        try:
            AVATARS_DIR.mkdir(parents=True, exist_ok=True)
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                stale = AVATARS_DIR / f"{key}{ext}"
                if stale.exists():
                    stale.unlink()
            dest = AVATARS_DIR / f"{key}.png"
            if not px.save(str(dest), "PNG"):
                raise RuntimeError("Failed to save PNG avatar.")
            self.hud.invalidate_avatar(key)
            self._append_log("SYS", f"Updated {p_name} avatar image ({dest.name}).")
        except Exception as exc:
            self._append_log("ERR", f"Could not save {p_name} avatar: {exc}")

    def _reset_persona_avatar(self) -> None:
        key = self.assistant
        p_name = PERSONAS[key]["name"]
        removed = False
        try:
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                candidate = AVATARS_DIR / f"{key}{ext}"
                if candidate.exists():
                    candidate.unlink()
                    removed = True
            self.hud.invalidate_avatar(key)
            if removed:
                self._append_log("SYS", f"Reset {p_name} avatar to built-in HUD emblem.")
            else:
                self._append_log("SYS", f"{p_name} is already using its built-in HUD emblem.")
        except Exception as exc:
            self._append_log("ERR", f"Could not reset {p_name} avatar: {exc}")

    def _toggle_setup_drawer(self, checked: bool) -> None:
        self.btn_setup_drawer.setChecked(checked)
        if checked:
            self._toggle_ctrl_drawer(False)
            self._refresh_avatar_buttons()
            self.setup_drawer.move(14, 66)
            self.setup_drawer.show()
            self.setup_drawer.raise_()
        else:
            self.setup_drawer.hide()

    def _toggle_ctrl_drawer(self, checked: bool) -> None:
        self.btn_ctrl_drawer.setChecked(checked)
        if checked:
            self._toggle_setup_drawer(False)
            self.ctrl_drawer.move(150, 66)
            self.ctrl_drawer.show()
            self.ctrl_drawer.raise_()
        else:
            self.ctrl_drawer.hide()

    def _open_mobile_remote_popup(self) -> None:
        self._toggle_setup_drawer(False)
        self._toggle_ctrl_drawer(False)
        if self._remote_popup:
            self._remote_popup.hide()
            self._remote_popup.deleteLater()
        cw = self.centralWidget()
        popup = MobileRemoteOverlay(self.remote_bridge, self._theme()["pri"], parent=cw)
        popup.move((cw.width() - popup.width()) // 2, (cw.height() - popup.height()) // 2)
        popup.show()
        popup.raise_()
        self._remote_popup = popup
        self._append_log("SYS", f"Mobile Remote QR opened ({self.remote_bridge.get_remote_url()})")

    def _handle_remote_control(self, data: dict) -> None:
        if "assistant" in data and data["assistant"] in PERSONAS:
            self._set_assistant(data["assistant"])
        if "mode" in data and data["mode"] in MODES:
            self._set_mode(data["mode"])
        act = data.get("action")
        if act == "mute":
            self._toggle_mute()
        elif act == "fullscreen":
            self._toggle_fullscreen()
        elif act == "clear":
            self.log_box.clear()
        elif act == "toggle3d":
            self._toggle_view3d()
        elif act == "nextmodel":
            self._next_model()

    def _handle_remote_prompt(self, prompt: str) -> None:
        self.prompt_input.setText(prompt)
        self._send_prompt()

    def _apply_theme(self) -> None:
        th = self._theme()
        pri, sec, dim, gho = th["pri"], th["sec"], th["dim"], th["gho"]
        p_info = PERSONAS[self.assistant]

        self.hud.set_theme(p_info["name"], pri, sec, persona_key=self.assistant)
        self._refresh_avatar_buttons()
        self.lbl_main_title.setStyleSheet(f"color: {pri}; letter-spacing: 5px;")
        self.lbl_sub_title.setText(th["sub"])
        self.lbl_sub_title.setStyleSheet(f"color: {sec}; letter-spacing: 1px;")
        self.lbl_clock.setStyleSheet(f"color: {pri};")

        for lbl in (
            self.lbl_tel_hdr,
            self.lbl_asst_hdr,
            self.lbl_mode_hdr,
            self.lbl_lang_hdr,
            self.lbl_comm_hdr,
        ):
            lbl.setStyleSheet(f"color: {pri};")

        self.bar_cpu.set_theme_color(pri)
        self.bar_mem.set_theme_color(sec)
        self.bar_net.set_theme_color("#00ff88")

        self.file_drop.setStyleSheet(f"""
            FileDropBox {{
                background: {gho};
                border: 1.5px dashed {dim};
                border-radius: 5px;
            }}
            FileDropBox:hover {{ border-color: {pri}; }}
        """)

        btn_style = f"""
            QPushButton {{
                background: #12151f;
                color: #d0d8e8;
                border: 1px solid #2a3045;
                border-radius: 4px;
                padding: 5px 8px;
                font-size: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                border-color: {pri};
                background: {gho};
                color: #ffffff;
            }}
            QPushButton:checked {{
                background: {pri};
                color: #07080c;
                border-color: {pri};
            }}
        """
        for w in self.findChildren(QPushButton):
            w.setStyleSheet(btn_style)

        for k, b in self.asst_btns.items():
            b.setChecked(k == self.assistant)
        for k, b in self.mode_btns.items():
            b.setChecked(k == self.mode)

    def _set_assistant(self, key: str) -> None:
        self.assistant = key
        self._apply_theme()
        v_name = PERSONA_LIVE_VOICES.get(key, "Kore")
        self._append_log("SYS", f"Switched persona to {PERSONAS[key]['name']} (Live Voice: {v_name}).")
        if self.live_voice.running:
            self._sync_live_voice_btn(True)
            self.live_voice.request_reconnect(reason=f"persona {PERSONAS[key]['name']} / voice {v_name}")

    def _set_mode(self, key: str) -> None:
        self.mode = key
        self._apply_theme()
        self._append_log("SYS", f"Mode set to {key.upper()}.")
        if self.live_voice.running:
            self.live_voice.request_reconnect(reason=f"mode {key.upper()}")

    def _toggle_language(self) -> None:
        order = ("tamil", "tanglish", "english")
        try:
            idx = order.index(self.language)
        except ValueError:
            idx = -1
        self.language = order[(idx + 1) % len(order)]
        self.btn_lang.setText(f"LANG: {self.language.upper()}")
        self._append_log("SYS", f"Response language set to {self.language.upper()}.")
        if self.live_voice.running:
            self.live_voice.request_reconnect(reason=f"language {self.language.upper()}")

    def _save_keys(self) -> None:
        g_key = self.gemini_key_input.text().strip()
        o_key = self.openrouter_key_input.text().strip()
        self.vision.save_keys_from_ui(g_key, o_key)
        st = self.vision.get_key_status()
        self.lbl_key_status.setText(
            f"Saved! Gemini: {'OK' if st['gemini_configured'] else 'Missing'} | "
            f"OpenRouter: {'OK' if st['openrouter_configured'] else 'Missing'}"
        )
        self._append_log("SYS", "API keys updated and saved locally.")

    def _make_shortcut(self) -> None:
        try:
            path = create_windows_desktop_shortcut("VISION AI", accent_hex=self._theme()["pri"])
            self._append_log("SYS", f"Desktop shortcut created: {path}")
        except Exception as exc:
            self._append_log("ERR", f"Shortcut creation failed: {exc}")

    def _check_autostart(self) -> bool:
        if sys.platform != "win32":
            return False
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Run",
                0,
                winreg.KEY_READ,
            ) as key:
                winreg.QueryValueEx(key, "VISION_AI")
                return True
        except Exception:
            return False

    def _refresh_autostart_label(self) -> None:
        on = self._check_autostart()
        self.btn_autostart.setText(f"◉ AUTO-START ON BOOT: {'ON' if on else 'OFF'}")

    def _toggle_autostart(self) -> None:
        if sys.platform != "win32":
            self._append_log("ERR", "Auto-start toggle is supported on Windows.")
            return
        try:
            import winreg

            currently_on = self._check_autostart()
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Run",
                0,
                winreg.KEY_ALL_ACCESS,
            ) as reg:
                if currently_on:
                    winreg.DeleteValue(reg, "VISION_AI")
                else:
                    pythonw = Path(sys.executable).parent / "pythonw.exe"
                    exe = str(pythonw if pythonw.exists() else sys.executable)
                    winreg.SetValueEx(reg, "VISION_AI", 0, winreg.REG_SZ, f'"{exe}" "{Path(__file__).resolve()}"')
            self._refresh_autostart_label()
            self._append_log("SYS", f"Auto-start {'disabled' if currently_on else 'enabled'}.")
        except Exception as exc:
            self._append_log("ERR", f"Auto-start error: {exc}")

    def _on_file_loaded(self, path: str) -> None:
        p = Path(path)
        try:
            content = p.read_text(encoding="utf-8", errors="replace")[:12000]
            self._loaded_file_context = content
            self._loaded_file_name = p.name
            self._append_log("FILE", f"Loaded '{p.name}' ({len(content)} chars) into context.")
        except Exception as exc:
            self._append_log("ERR", f"Could not read file: {exc}")

    def _unload_file(self) -> None:
        self._loaded_file_context = ""
        self._loaded_file_name = ""
        self.file_drop.clear_file()
        self._append_log("SYS", "Attached file unloaded.")

    def _insert_quick_prompt(self, prefix: str) -> None:
        current = self.prompt_input.text().strip()
        self.prompt_input.setText(f"{prefix}{current}")
        self.prompt_input.setFocus()

    def _export_study_notes(self) -> None:
        NOTES_DIR.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_file = NOTES_DIR / f"vision_notes_{ts}.md"
        out_file.write_text(self.log_box.toPlainText(), encoding="utf-8")
        self._append_log("SYS", f"Study notes exported to: {out_file}")

    def _append_log(self, sender: str, text: str) -> None:
        pri = self._theme()["pri"]
        color = pri if sender in (PERSONAS[self.assistant]["name"], "SYSTEM") else "#f0f4fc"
        if sender == "ERR":
            color = "#ff1f43"
        elif sender in ("SYS", "FILE"):
            color = "#9aa8c0"
        safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        self.log_box.append(f'<b style="color:{color}">[{sender}]</b> {safe_text}<br>')

    def _send_prompt(self) -> None:
        text = self.prompt_input.text().strip()
        if not text:
            return
        if re.fullmatch(r"(AIza|AQ\.|sk-)[A-Za-z0-9_.\-]{20,}", text):
            self.prompt_input.clear()
            self._append_log("SYS", "இது API key போல இருக்கு, அதனால அனுப்பல. '⚙ SETUP & KEYS' drawer-ல paste பண்ணுங்க.")
            return
        self.prompt_input.clear()
        self._append_log("YOU", text)
        self.hud.set_status("THINKING")

        full_prompt = text
        if self._loaded_file_context:
            full_prompt = (
                f"[ATTACHED FILE: {self._loaded_file_name}]\n"
                f"{self._loaded_file_context}\n\n"
                f"[USER QUESTION]\n{text}"
            )

        # If Gemini Live Native Voice is connected, send straight into the live session so AI speaks back directly!
        if self.live_voice.is_connected and self.provider == "gemini":
            if self.live_voice.send_text(full_prompt):
                return

        asst = self.assistant
        mode = self.mode
        prov = self.provider
        lang = self.language
        sender_name = PERSONAS[asst]["name"]

        def _worker():
            try:
                reply = self.vision.ask(
                    prompt=full_prompt,
                    persona=asst,
                    mode=mode,
                    provider=prov,
                    language=lang,
                )
            except Exception as exc:
                reply = f"Error: {exc}"
                if any(k in str(exc) for k in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED")):
                    reply += "\n(Google server பிஸியா இருக்கு — இது உங்க code / key பிரச்சனை இல்ல. ஒரு நிமிடம் கழிச்சு மறுபடி try பண்ணுங்க.)"
            self._reply_ready.emit(sender_name, reply)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_reply_ready(self, sender: str, reply: str) -> None:
        self.hud.set_status("ONLINE")
        self._append_log(sender, reply)

    def _tick_clock(self) -> None:
        self.lbl_clock.setText(time.strftime("%H:%M:%S"))
        self.lbl_date.setText(time.strftime("%a %d %b %Y"))

    def _tick_metrics(self) -> None:
        try:
            import psutil

            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
            self.bar_cpu.set_value(cpu, f"{cpu:.0f}%")
            self.bar_mem.set_value(mem, f"{mem:.0f}%")

            net_io = psutil.net_io_counters()
            mb = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)
            self.bar_net.set_value(min(100.0, (mb % 100)), f"{mb:.0f}MB")

            elapsed = time.time() - psutil.boot_time()
            hrs = int(elapsed // 3600)
            mins = int((elapsed % 3600) // 60)
            self.lbl_uptime.setText(f"UPTIME: {hrs:02d}h {mins:02d}m")
        except Exception:
            self.bar_cpu.set_value(0, "N/A")
            self.bar_mem.set_value(0, "N/A")
            self.bar_net.set_value(0, "N/A")


    def closeEvent(self, event) -> None:
        if hasattr(self, "live_voice") and self.live_voice.running:
            self.live_voice.stop()
        super().closeEvent(event)


def main() -> None:
    if "--install-deps" in sys.argv:
        install_dependencies()
        return

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = VisionDesktopWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
