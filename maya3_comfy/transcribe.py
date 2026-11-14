"""
ComfyUI Custom Node: Whisper Audio Transcription (Maya2 Native Extension)

Transcribes any AUDIO source with segment-level and word-level timestamps
using Whisper.

Two interchangeable engines:

1. faster-whisper (CTranslate2) - RECOMMENDED, and the default engine.
   Typically 3-5x faster than openai-whisper with equal accuracy, and it
   does NOT depend on Triton, so word timestamps work out of the box on
   Windows portable/embedded ComfyUI installs (where Triton's tcc compile
   of cuda_utils.c fails with 'Python.h not found' and openai-whisper
   falls back to a very slow alignment path).
2. openai-whisper - the original implementation, kept as a fallback for
   users who already have it installed.

- Languages: English (en), Hindi (hi) + automatic language detection
- task=translate turns Hindi (or any) speech into English text
- Outputs: plain text, JSON segments (with word timestamps), SRT subtitles,
  a readable inline-timestamped transcript, the detected language, and the
  original audio (passthrough, so the node can sit inline in a pipeline)
"""

import gc
import json
import os
import time

import numpy as np
import torch
import torchaudio

# Node directory for bundled Whisper checkpoints
NODE_DIR = os.path.dirname(__file__)

# Optional: ComfyUI models directory (missing when running outside ComfyUI)
try:
    import folder_paths
    _COMFYUI_MODELS_DIR = getattr(folder_paths, "models_dir", None)
except Exception:
    _COMFYUI_MODELS_DIR = None

# ---------------------------------------------------------------------------
# Engine imports (each optional; the node reports a clear error if none load)
# ---------------------------------------------------------------------------
FASTER_WHISPER_AVAILABLE = False
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FASTER_WHISPER_AVAILABLE = True
    print("[Maya2 STT] faster-whisper available - fast transcription engine ready (default)")
except Exception as faster_error:
    print(f"[Maya2 STT] faster-whisper not found - fast engine disabled: {faster_error}")
    print("[Maya2 STT] Install it with: pip install faster-whisper")

WHISPER_AVAILABLE = False
whisper = None
try:
    import whisper
    WHISPER_AVAILABLE = True
    print("[Maya2 STT] OpenAI Whisper available - fallback transcription engine ready")
except Exception as whisper_error:
    print(f"[Maya2 STT] OpenAI Whisper not found - fallback engine disabled: {whisper_error}")
    print("[Maya2 STT] Install it with: pip install openai-whisper")

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
SUPPORTED_LANGUAGES = ["auto", "en", "hi"]
WHISPER_LANGUAGE_CODES = ("en", "hi")

# The language dropdown shows readable names; these map a selection back to
# the code Whisper expects. Bare codes ("auto"/"en"/"hi") still work, so old
# saved workflows keep loading unchanged.
LANGUAGE_LABELS = {"auto": "Auto-detect", "en": "English", "hi": "Hindi"}
LANGUAGE_CHOICES = ["Auto-detect", "English", "Hindi"]
LANGUAGE_CHOICE_TO_CODE = {label: code for code, label in LANGUAGE_LABELS.items()}


def _language_code(value) -> str:
    """Normalize a widget selection ('English' or 'en') to 'auto'/'en'/'hi'."""
    text = str(value or "").strip()
    if text in SUPPORTED_LANGUAGES:
        return text
    return LANGUAGE_CHOICE_TO_CODE.get(text, "auto")

# CTranslate2 compute types: auto = fp16 on GPU / int8 on CPU
COMPUTE_TYPES = ["auto", "float16", "int8", "int8_float16", "float", "float32"]
ENGINES = ["faster-whisper", "openai-whisper"]


class Maya2NativeTranscribe:
    """
    Whisper-based audio transcription with timestamps.
    Supports English and Hindi (plus auto-detection), with a choice of
    engine: faster-whisper (fast, CTranslate2) or openai-whisper (fallback).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "whisper_model": (WHISPER_MODELS, {
                    "default": "base",
                    "tooltip": "Whisper checkpoint: tiny (fastest) → large-v3 (best). base is a good speed/quality compromise. Multilingual models - required for Hindi",
                }),
                "language": (LANGUAGE_CHOICES, {
                    "default": "Auto-detect",
                    "tooltip": "Auto-detect = let Whisper detect the language | English / Hindi force that language",
                }),
                "task": (["transcribe", "translate"], {
                    "default": "transcribe",
                    "tooltip": "transcribe: same language as the audio | translate: translate the speech into English text",
                }),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            },
            "optional": {
                "word_timestamps": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Attach per-word start/end times to each segment (used by segments_json)",
                }),
                "beam_size": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1}),
                "temperature": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0.0 = greedy/deterministic decoding (recommended)",
                }),
                "initial_prompt": ("STRING", {
                    "default": "",
                    "tooltip": "Optional hint text (names, spellings, punctuation style). Greatly improves Hindi proper-noun accuracy",
                }),
                "model_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional local Whisper checkpoint (faster-whisper folder or openai-whisper .pt file). Empty = auto-download by model name",
                }),
                "engine": (ENGINES, {
                    "default": "faster-whisper",
                    "tooltip": "faster-whisper (CTranslate2): 3-5x faster, no Triton dependency, recommended - especially on Windows | openai-whisper: original implementation fallback",
                }),
                "compute_type": (COMPUTE_TYPES, {
                    "default": "auto",
                    "tooltip": "faster-whisper only: auto = float16 on CUDA, int8 on CPU. int8 is much faster than float on CPU",
                }),
                "cpu_threads": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": "faster-whisper only: CPU threads (0 = leave at CTranslate2 default = 4)",
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep the Whisper model in memory after the run (faster next run, holds VRAM/RAM) | "
                               "off: unload it to free memory - it is reloaded on the next run",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "AUDIO")
    RETURN_NAMES = ("text", "segments_json", "srt", "timestamped_text", "detected_language", "audio")
    FUNCTION = "transcribe_audio"
    CATEGORY = "Maya2 Native"

    def __init__(self):
        self.models = {}  # engine-key -> loaded model instance

    # ------------------------------------------------------------------
    # Checkpoint resolution (shared by both engines)
    # ------------------------------------------------------------------
    def _resolve_faster_whisper_source(self, model_name: str, custom_path: str = ""):
        """
        faster-whisper accepts a model name (Systran/faster-whisper-<name>
        on HuggingFace) or a local folder path.
        Priority: explicit path → node models dir → ComfyUI models dir → HF name.
        """
        candidates = []

        if custom_path and os.path.isdir(custom_path):
            candidates.append(custom_path)

        # Local folders shipped with the node / placed in ComfyUI models
        for base in (
            os.path.join(NODE_DIR, "models", "faster-whisper"),
            os.path.join(NODE_DIR, "models", "whisper"),
            os.path.join(_COMFYUI_MODELS_DIR or "", "faster-whisper"),
            os.path.join(_COMFYUI_MODELS_DIR or "", "whisper"),
        ):
            candidates.append(os.path.join(base, model_name))

        # This pack's own cache (see _get_faster_whisper_model download_root).
        maya3_whisper = os.path.join(_COMFYUI_MODELS_DIR or "", "maya3", "whisper")
        candidates.append(os.path.join(maya3_whisper, model_name))
        snapshots = os.path.join(
            maya3_whisper, f"models--Systran--faster-whisper-{model_name}", "snapshots"
        )
        if os.path.isdir(snapshots):
            for snapshot in sorted(os.listdir(snapshots), reverse=True):
                candidates.append(os.path.join(snapshots, snapshot))

        for candidate in candidates:
            if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "model.bin")):
                print(f"[Maya2 STT] Using local faster-whisper model folder: {candidate}")
                return candidate

        hf_name = f"Systran/faster-whisper-{model_name}"
        print(f"[Maya2 STT] No local faster-whisper model for '{model_name}' - using {hf_name} (downloads on first use)")
        return hf_name

    def _resolve_openai_source(self, model_name: str, custom_path: str = ""):
        """
        openai-whisper accepts a .pt checkpoint path or a model name
        (auto-download from OpenAI on first use).
        Priority: explicit file → node/ComfyUI models dirs → model name.
        """
        candidates = []

        if custom_path:
            if os.path.isfile(custom_path):
                candidates.append(custom_path)
            elif os.path.isdir(custom_path):
                candidates.append(os.path.join(custom_path, f"{model_name}.pt"))

        candidates.append(os.path.join(NODE_DIR, "models", "whisper", f"{model_name}.pt"))
        if _COMFYUI_MODELS_DIR:
            candidates.append(os.path.join(_COMFYUI_MODELS_DIR, "whisper", f"{model_name}.pt"))

        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                print(f"[Maya2 STT] Using local Whisper checkpoint: {candidate}")
                return candidate

        print(f"[Maya2 STT] No local Whisper checkpoint for '{model_name}' - downloading on first use")
        return model_name

    # ------------------------------------------------------------------
    # Model loading / caching (per engine + settings key)
    # ------------------------------------------------------------------
    def _get_faster_whisper_model(self, model_name, device, compute_type, cpu_threads, custom_path):
        key = ("faster-whisper", model_name, device, compute_type, cpu_threads, custom_path or "")
        if key in self.models:
            return self.models[key]

        self._release_models()
        source = self._resolve_faster_whisper_source(model_name, custom_path)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        kwargs = {"cpu_threads": int(cpu_threads)} if cpu_threads and cpu_threads > 0 else {}
        if _COMFYUI_MODELS_DIR:
            kwargs["download_root"] = os.path.join(_COMFYUI_MODELS_DIR, "maya3", "whisper")

        print(f"[Maya2 STT] Loading faster-whisper '{model_name}' on {device} ({compute_type})...")
        model = FasterWhisperModel(source, device=device, compute_type=compute_type, **kwargs)
        self.models[key] = model
        print(f"[Maya2 STT] faster-whisper '{model_name}' ready")
        return model

    def _get_openai_model(self, model_name, device, custom_path):
        key = ("openai-whisper", model_name, device, custom_path or "")
        if key in self.models:
            return self.models[key]

        self._release_models()
        source = self._resolve_openai_source(model_name, custom_path)

        print(f"[Maya2 STT] Loading Whisper '{model_name}' on {device}...")
        model = whisper.load_model(source, device=device)
        self.models[key] = model
        print(f"[Maya2 STT] Whisper '{model_name}' ready")
        return model

    def _release_models(self):
        """
        Unload every cached Whisper model and free VRAM/RAM.
        Called when switching engines/models and after each completed
        transcription - the model is simply reloaded on the next run.
        """
        if self.models:
            released = ", ".join(str(key[0]) for key in self.models)
            self.models.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[Maya2 STT] Unloaded Whisper model(s) to free memory: {released}")

    def _require_engine(self, engine):
        """Raise a helpful error when the selected engine is not installed."""
        if engine == "faster-whisper" and not FASTER_WHISPER_AVAILABLE:
            raise ImportError(
                "faster-whisper is not installed. Install it with: pip install faster-whisper "
                "(or switch the 'engine' widget to openai-whisper)"
            )
        if engine == "openai-whisper" and not WHISPER_AVAILABLE:
            raise ImportError(
                "openai-whisper is not installed. Install it with: pip install openai-whisper "
                "(or switch the 'engine' widget to faster-whisper)"
            )

    # ------------------------------------------------------------------
    # Audio conversion (ComfyUI AUDIO -> 16 kHz mono float32)
    # ------------------------------------------------------------------
    def _audio_to_16k_mono(self, audio) -> np.ndarray:
        if audio is None:
            raise ValueError("Whisper transcription received no audio")

        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate", 16000)
        if waveform is None or waveform.numel() == 0:
            raise ValueError("Whisper transcription received empty audio")

        # ComfyUI AUDIO: [batch, channels, samples] → first clip → mono
        if waveform.dim() == 3:
            waveform = waveform[0]
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform[0]

        mono = waveform.detach().float().cpu()
        if sample_rate != 16000:
            mono = torchaudio.functional.resample(mono, sample_rate, 16000)

        mono = np.clip(mono.numpy(), -1.0, 1.0).astype(np.float32)
        return np.ascontiguousarray(mono)

    # ------------------------------------------------------------------
    # Timestamp helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _format_srt_timestamp(seconds) -> str:
        """00:00:01,234 style timestamp (SRT subtitle format)."""
        total_ms = max(0, int(round(float(seconds) * 1000)))
        hours, total_ms = divmod(total_ms, 3_600_000)
        minutes, total_ms = divmod(total_ms, 60_000)
        secs, ms = divmod(total_ms, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

    @staticmethod
    def _format_short_timestamp(seconds) -> str:
        """MM:SS.mmm style timestamp (HH added only when needed)."""
        total_ms = max(0, int(round(float(seconds) * 1000)))
        hours, total_ms = divmod(total_ms, 3_600_000)
        minutes, total_ms = divmod(total_ms, 60_000)
        secs, ms = divmod(total_ms, 1000)
        if hours:
            return f"{hours}:{minutes:02d}:{secs:02d}.{ms:03d}"
        return f"{minutes:02d}:{secs:02d}.{ms:03d}"

    # ------------------------------------------------------------------
    # Transcription (engine dispatch)
    # ------------------------------------------------------------------
    def transcribe_audio(self, audio, whisper_model="base", language="Auto-detect",
                         task="transcribe", device="auto",
                         word_timestamps=True, beam_size=5, temperature=0.0,
                         initial_prompt="", model_path="",
                         engine="faster-whisper", compute_type="auto", cpu_threads=0,
                         keep_model_loaded=True):
        print("[Maya2 STT] Whisper transcription starting...")

        # --- validate / normalize widget values (None-safe) ---
        if whisper_model is None or whisper_model not in WHISPER_MODELS:
            whisper_model = "base"
        language = _language_code(language)
        if task not in ("transcribe", "translate"):
            task = "transcribe"
        if device not in ("auto", "cpu", "cuda"):
            device = "auto"
        if word_timestamps is None:
            word_timestamps = True
        if beam_size is None or beam_size < 1:
            beam_size = 5
        if temperature is None or not (0.0 <= float(temperature) <= 1.0):
            temperature = 0.0
        if engine not in ENGINES:
            engine = "faster-whisper" if FASTER_WHISPER_AVAILABLE else "openai-whisper"
        if compute_type not in COMPUTE_TYPES:
            compute_type = "auto"
        if cpu_threads is None or cpu_threads < 0:
            cpu_threads = 0
        self._require_engine(engine)

        # Resolve device once so both engines print and behave consistently
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Convert/validate the audio first so bad input fails fast,
        # before a potentially heavy model load
        audio_np = self._audio_to_16k_mono(audio)
        duration_s = len(audio_np) / 16000.0

        # Only pass languages Whisper actually knows; anything else = auto-detect
        whisper_language = language if language in WHISPER_LANGUAGE_CODES else None

        t0 = time.perf_counter()
        if engine == "faster-whisper":
            segments_out, text, detected_language = self._transcribe_faster(
                audio_np, whisper_model, whisper_language, task, device,
                word_timestamps, beam_size, temperature, initial_prompt,
                model_path, compute_type, cpu_threads,
            )
        else:
            segments_out, text, detected_language = self._transcribe_openai(
                audio_np, whisper_model, whisper_language, task, device,
                word_timestamps, beam_size, temperature, initial_prompt,
                model_path,
            )
        elapsed = time.perf_counter() - t0

        segments_json = json.dumps(segments_out, ensure_ascii=False, indent=2)

        # SRT subtitles (00:00:01,234 style timestamps)
        srt_blocks = []
        for i, seg in enumerate(segments_out, 1):
            srt_blocks.append(
                f"{i}\n"
                f"{self._format_srt_timestamp(seg['start'])} --> {self._format_srt_timestamp(seg['end'])}\n"
                f"{seg['text']}\n"
            )
        srt_text = "\n".join(srt_blocks)

        # Readable transcript with inline timestamps
        timestamped_text = "\n".join(
            f"[{self._format_short_timestamp(seg['start'])} - "
            f"{self._format_short_timestamp(seg['end'])}] {seg['text']}"
            for seg in segments_out
        )

        word_count = sum(len(seg.get("words", [])) for seg in segments_out)
        speed = duration_s / elapsed if elapsed > 0 else 0.0
        print(
            f"[Maya2 STT] Transcribed {duration_s:.2f}s audio in {elapsed:.1f}s ({speed:.1f}x realtime) "
            f"via {engine}: {len(segments_out)} segments"
            + (f", {word_count} words" if word_timestamps else "")
            + f" | language={detected_language or 'unknown'}"
        )
        if text:
            preview = text if len(text) <= 120 else text[:117] + "..."
            print(f"[Maya2 STT] Transcript preview: {preview}")

        # Release the model after the run unless asked to keep it - frees
        # VRAM/RAM at the cost of a reload on the next run.
        if keep_model_loaded:
            print("[Maya2 STT] Keeping Whisper model loaded for the next run (keep_model_loaded=true)")
        else:
            # It is reloaded (from local/HF cache) on the next run.
            self._release_models()

        # audio passthrough lets the node sit inline in a pipeline
        return (text, segments_json, srt_text, timestamped_text, detected_language, audio)

    # ------------------------------------------------------------------
    # Engine 1: faster-whisper (CTranslate2)
    # ------------------------------------------------------------------
    def _transcribe_faster(self, audio_np, model_name, language, task, device,
                           word_timestamps, beam_size, temperature, initial_prompt,
                           model_path, compute_type, cpu_threads):
        model = self._get_faster_whisper_model(model_name, device, compute_type, cpu_threads, model_path or "")

        print(
            f"[Maya2 STT] Audio: {len(audio_np) / 16000.0:.2f}s @16kHz mono | engine=faster-whisper | "
            f"model={model_name} | language={language or 'auto-detect'} | task={task}"
        )

        # faster-whisper temperature: float or tuple of fallbacks
        segments_iter, info = model.transcribe(
            audio_np,
            language=language,
            task=task,
            word_timestamps=word_timestamps,
            beam_size=beam_size,
            temperature=float(temperature),
            initial_prompt=(initial_prompt or "").strip() or None,
            vad_filter=True,          # trims silence/noise → big speedup on real recordings
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
        )

        segments_out = []
        for seg in segments_iter:
            seg_text = (seg.text or "").strip()
            if not seg_text:
                continue
            entry = {
                "id": int(seg.id),
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "text": seg_text,
            }
            if word_timestamps and seg.words:
                entry["words"] = [
                    {
                        "word": (w.word or "").strip(),
                        "start": round(float(w.start), 3),
                        "end": round(float(w.end), 3),
                        "probability": round(float(w.probability or 0.0), 4),
                    }
                    for w in seg.words
                    if w.start is not None and w.end is not None
                ]
            segments_out.append(entry)

        text = " ".join(seg["text"] for seg in segments_out).strip()
        detected = (info.language or "") if info and info.language else ""
        return segments_out, text, detected

    # ------------------------------------------------------------------
    # Engine 2: openai-whisper (fallback)
    # ------------------------------------------------------------------
    def _transcribe_openai(self, audio_np, model_name, language, task, device,
                           word_timestamps, beam_size, temperature, initial_prompt,
                           model_path):
        model = self._get_openai_model(model_name, device, model_path or "")

        print(
            f"[Maya2 STT] Audio: {len(audio_np) / 16000.0:.2f}s @16kHz mono | engine=openai-whisper | "
            f"model={model_name} | language={language or 'auto-detect'} | task={task} | device={device}"
        )

        result = model.transcribe(
            audio_np,
            language=language,
            task=task,
            word_timestamps=word_timestamps,
            beam_size=beam_size,
            temperature=temperature,
            initial_prompt=(initial_prompt or "").strip() or None,
            fp16=(device != "cpu"),
            verbose=False,
            condition_on_previous_text=False,
        )

        segments_raw = result.get("segments", []) or []
        text = (result.get("text") or "").strip()
        detected = (result.get("language") or "").strip()

        # Build normalized segment list (with optional word-level timestamps).
        # Whisper occasionally emits empty segments (no text/words, zero length)
        # which would produce empty SRT blocks - they are filtered out here.
        segments_out = []
        for seg in segments_raw:
            seg_text = (seg.get("text") or "").strip()
            if not seg_text:
                continue
            entry = {
                "id": int(seg.get("id", len(segments_out))),
                "start": round(float(seg.get("start", 0.0)), 3),
                "end": round(float(seg.get("end", 0.0)), 3),
                "text": seg_text,
            }
            words = seg.get("words")
            if word_timestamps and words:
                entry["words"] = [
                    {
                        "word": (w.get("word") or "").strip(),
                        "start": round(float(w["start"]), 3),
                        "end": round(float(w["end"]), 3),
                        "probability": round(float(w.get("probability", 0.0)), 4),
                    }
                    for w in words
                    if w.get("start") is not None and w.get("end") is not None
                ]
            segments_out.append(entry)

        return segments_out, text, detected


# Node mappings (mirrors what __init__.py registers)
NODE_CLASS_MAPPINGS = {
    "Maya2NativeTranscribe": Maya2NativeTranscribe
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Maya2NativeTranscribe": "Maya2 Native Whisper Transcription"
}
