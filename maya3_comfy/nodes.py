"""ComfyUI nodes for Maya2 Native (Maya3 TTS engine)."""
from __future__ import annotations

import os
import sys

from .audio import load_npy_audio, save_audio_cache, to_comfy_audio
from .client import Maya3Engine, get_engine, reset_engines
from .paths import DEFAULT_REPO_ID, home_dir, tmp_dir
from .runtime import ensure_runtime
from .transcribe import Maya2NativeTranscribe

LANGUAGES = ["as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]
LANGUAGE_NAMES = {
    "as": "Assamese",
    "bn": "Bengali",
    "en": "English",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "or": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
}
LANGUAGE_TOOLTIP = "Language of the text: " + ", ".join(f"{k}={v}" for k, v in LANGUAGE_NAMES.items())

# Dropdown labels show the language names; the map converts a selection back
# to the language code the engine expects. Bare codes still work (typed
# workflows / old saved workflows keep loading unchanged).
LANGUAGE_CHOICES = [f"{name} ({code})" for code, name in LANGUAGE_NAMES.items()]
LANGUAGE_CHOICE_TO_CODE = {f"{name} ({code})": code for code, name in LANGUAGE_NAMES.items()}


def _language_code(value) -> str:
    """Normalize a widget selection ('Hindi (hi)' or 'hi') to a language code."""
    text = str(value or "").strip()
    if text in LANGUAGES:
        return text
    return LANGUAGE_CHOICE_TO_CODE.get(text, "en")


NAMED_VOICES = [
    "plain",
    "Ahana",
    "Vaidehi",
    "Malini",
    "Charvi",
    "Jahnavi",
    "Anirban",
    "Nakul",
    "Vasant",
    "Siddhant",
    "Satvik",
    "Abhay",
    "Nivedita",
    "Ishwari",
    "Ritvik",
    "Vrinda",
    "Sreekar",
    "Kayal",
    "Shaunak",
    "Akshat",
    "Tanushree",
]
VOICE_TOOLTIP = (
    "Built-in voice. 'plain' has no speaker conditioning; the named voices are the "
    "model's 20 trained personas. Do not put the voice name in the text."
)

DEFAULT_TEXT = "Hello! This is Maya2 Native, running locally inside ComfyUI."


def _interrupt_check() -> bool:
    try:
        import comfy.model_management as mm

        if mm.processing_interrupted():
            return True
    except Exception:
        pass
    try:
        import execution

        if execution.interruption_requested():
            return True
    except Exception:
        pass
    return False


def _raise_interrupt() -> None:
    try:
        import comfy.model_management as mm

        exc = getattr(mm, "InterruptProcessingException", None)
        if exc is not None:
            raise exc()
    except Exception:
        pass
    raise InterruptedError("Maya2 generation cancelled")


def _format_info(info: dict, result: dict) -> str:
    parts = []
    for key in ("reason", "frames", "pieces"):
        value = info.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    for key, fmt in (("audio_s", "{:.2f}s"), ("ttfa_s", "{:.3f}s"), ("wall_s", "{:.2f}s")):
        value = info.get(key)
        if isinstance(value, (int, float)):
            parts.append(f"{key}=" + fmt.format(value))
    parts.append(f"wall={result.get('wall_s')}s")
    parts.append(f"seed={result.get('seed')}")
    return " ".join(parts)


class Maya2NativeLoader:
    DESCRIPTION = (
        "Downloads (optional) and loads the Maya2 Native / Maya3 TTS model in an "
        "isolated worker process.  The checkpoint, Mimi codec and ECAPA encoder "
        "come from maya-research/maya2-native on Hugging Face."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {
                    "default": DEFAULT_REPO_ID,
                    "tooltip": "Hugging Face repository id.",
                }),
                "revision": ("STRING", {
                    "default": "",
                    "tooltip": "Optional revision/branch/commit of the release repo. Empty = main.",
                }),
                "performance": (["auto", "fast", "safe"], {
                    "default": "auto",
                    "tooltip": "auto = safe on Windows, fast elsewhere; fast = CUDA graphs + Triton kernels; "
                               "safe = portable gather attention without graphs. The worker retries safe on OOM.",
                }),
                "max_seats": ("INT", {
                    "default": 1, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Concurrent synthesis seats. Keep at 1 on 8-12 GB GPUs.",
                }),
                "capacity": ("INT", {
                    "default": 1024, "min": 512, "max": 3072, "step": 64,
                    "tooltip": "KV tokens per seat. 1024 is a good fit for 8 GB; lower it if loading OOMs.",
                }),
                "depth_dtype": (["fp16", "bf16", "fp32"], {
                    "default": "fp16",
                    "tooltip": "Depth-decoder dtype. fp16 is the model default; fp32 uses much more VRAM.",
                }),
                "auto_download": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Download the release, Mimi and ECAPA from Hugging Face on first use.",
                }),
                "free_vram_first": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Ask ComfyUI to unload its models before the Maya2 worker starts.",
                }),
                "idle_unload_s": ("INT", {
                    "default": 600, "min": 0, "max": 86400, "step": 30,
                    "tooltip": "Unload the worker after this many idle seconds (0 = keep it loaded).",
                }),
            },
            "optional": {
                "force_reload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force a fresh worker/model load even if the cached engine is alive.",
                }),
            },
        }

    RETURN_TYPES = ("MAYA2_ENGINE", "STRING", "STRING")
    RETURN_NAMES = ("engine", "status", "voices")
    OUTPUT_TOOLTIPS = (
        "Connect to the Maya2 TTS or Voice Clone node.",
        "Human-readable load status.",
        "Available voices, one per line.",
    )
    FUNCTION = "load"
    CATEGORY = "Maya2 Native"

    @classmethod
    def IS_CHANGED(cls, force_reload=False, **kwargs):
        return float("nan") if force_reload else ""

    def load(
        self,
        repo_id,
        revision,
        performance,
        max_seats,
        capacity,
        depth_dtype,
        auto_download,
        free_vram_first,
        idle_unload_s,
        force_reload=False,
    ):
        if force_reload:
            reset_engines()

        site = ensure_runtime(log=lambda message: print(message, flush=True))
        python = os.environ.get("MAYA3_COMFY_PYTHON") or sys.executable
        config = {
            "repo_id": str(repo_id).strip() or DEFAULT_REPO_ID,
            "revision": str(revision).strip(),
            "performance": str(performance),
            "max_seats": int(max_seats),
            "capacity": int(capacity),
            "depth_dtype": str(depth_dtype),
            "auto_download": bool(auto_download),
            "free_vram_first": bool(free_vram_first),
            "idle_unload_s": int(idle_unload_s),
        }

        engine = get_engine(config, python, site, home_dir())
        try:
            info = engine.start(cancel_check=_interrupt_check)
        except InterruptedError:
            _raise_interrupt()
            raise

        cfg = info.get("engine_config") or {}
        status = "\n".join([
            "Maya2 Native ready",
            f"repo: {info.get('repo')} @ {info.get('revision')}",
            f"snapshot: {info.get('model_dir')}",
            f"preset: {info.get('preset')} (attention={cfg.get('attention')}, graphs={cfg.get('use_graphs')})",
            f"gpu: {info.get('gpu')}",
            f"torch: {info.get('torch')} | transformers: {info.get('transformers')}",
            f"load: {info.get('load_s')} s (warmup {info.get('warmup_s')} s)",
            f"voices: {len(info.get('voices') or [])} (plain + named)",
        ])
        voices = "\n".join(info.get("voices") or ["plain"])
        return (engine, status, voices)


class Maya2NativeTTS:
    DESCRIPTION = (
        "Synthesize speech with a loaded Maya2 Native model. Output is 24 kHz mono "
        "float audio in ComfyUI's AUDIO format."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "engine": ("MAYA2_ENGINE",),
                "text": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_TEXT,
                    "dynamicPrompts": False,
                    "tooltip": "Text to speak. Inline emotion tags like [laughs] or [whispers] are best-effort.",
                }),
                "language": (LANGUAGE_CHOICES, {
                    "default": "English (en)",
                    "tooltip": LANGUAGE_TOOLTIP,
                }),
                "voice": (NAMED_VOICES, {"default": "Ahana", "tooltip": VOICE_TOOLTIP}),
                "normalize_text": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Expand numbers, currency, dates, URLs and ALL-CAPS names before synthesis.",
                }),
                "max_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 120.0, "step": 0.5,
                    "tooltip": "0 = engine default (hard cap 120 s). Otherwise stop after this many seconds.",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1,
                    "tooltip": "0 = random seed each run; 1+ = reproducible sampling (when the request runs alone).",
                }),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.05, "max": 2.0, "step": 0.05}),
                "top_k": ("INT", {"default": 64, "min": 1, "max": 2049, "step": 1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 5.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("audio", "duration_s", "seed_used", "info")
    FUNCTION = "generate"
    CATEGORY = "Maya2 Native"

    def generate(
        self,
        engine: Maya3Engine,
        text,
        language,
        voice,
        normalize_text,
        max_seconds,
        seed,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
    ):
        payload = {
            "text": text,
            "language": _language_code(language),
            "voice": voice,
            "normalize_text": bool(normalize_text),
            "max_seconds": float(max_seconds),
            "seed": int(seed),
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "repetition_penalty": float(repetition_penalty),
        }
        try:
            result = engine.run("generate", payload, timeout=900.0, cancel_check=_interrupt_check)
        except InterruptedError:
            _raise_interrupt()
            raise
        wav_path = result["wav_path"]
        try:
            wav = load_npy_audio(wav_path)
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass
        audio = to_comfy_audio(wav, int(result.get("sample_rate", 24000)))
        info = result.get("info") or {}
        return (audio, float(result.get("duration_s", len(wav) / 24000.0)), int(result.get("seed", 0)), _format_info(info, result))


class Maya2NativeVoiceClone:
    DESCRIPTION = (
        "Clone a voice from a reference clip (3-10 s of clean speech) and synthesize "
        "text in any supported language.  Only clone voices you have permission to use."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "engine": ("MAYA2_ENGINE",),
                "text": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_TEXT,
                    "dynamicPrompts": False,
                }),
                "language": (LANGUAGE_CHOICES, {
                    "default": "English (en)",
                    "tooltip": LANGUAGE_TOOLTIP,
                }),
                "ref_audio": ("AUDIO", {
                    "tooltip": "3-10 s clean reference speech (longer than 30 s is rejected).",
                }),
                "ref_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "dynamicPrompts": False,
                    "tooltip": "Exact transcript of ref_audio. Required for icl and icl_spk; unused for spk.",
                }),
                "ref_language": (LANGUAGE_CHOICES, {
                    "default": "English (en)",
                    "tooltip": "Language of the reference clip (may differ from the target language).",
                }),
                "mode": (["icl_spk", "icl", "spk"], {
                    "default": "icl_spk",
                    "tooltip": "icl_spk = reference prompt + speaker embedding (best); icl = prompt only; "
                               "spk = speaker embedding only (no transcript).",
                }),
                "normalize_text": ("BOOLEAN", {"default": True}),
                "max_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 120.0, "step": 0.5,
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1,
                }),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.05, "max": 2.0, "step": 0.05}),
                "top_k": ("INT", {"default": 64, "min": 1, "max": 2049, "step": 1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 5.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("audio", "duration_s", "seed_used", "info")
    FUNCTION = "clone"
    CATEGORY = "Maya2 Native"

    def clone(
        self,
        engine: Maya3Engine,
        text,
        language,
        ref_audio,
        ref_text,
        ref_language,
        mode,
        normalize_text,
        max_seconds,
        seed,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
    ):
        if mode in ("icl", "icl_spk") and not str(ref_text).strip():
            raise ValueError(f"mode={mode} needs the exact transcript of the reference clip in ref_text")
        if not str(text).strip():
            raise ValueError("text is empty")

        # Content-addressed reference path: the engine caches voices by file
        # stat, so an identical clip reuses its Mimi codes and ECAPA embedding.
        ref_path, ref_sample_rate = save_audio_cache(ref_audio, tmp_dir())
        payload = {
            "text": text,
            "language": _language_code(language),
            "ref_audio_path": str(ref_path),
            "ref_sample_rate": ref_sample_rate,
            "ref_text": str(ref_text),
            "ref_language": _language_code(ref_language),
            "mode": mode,
            "normalize_text": bool(normalize_text),
            "max_seconds": float(max_seconds),
            "seed": int(seed),
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "repetition_penalty": float(repetition_penalty),
        }
        try:
            result = engine.run("clone", payload, timeout=900.0, cancel_check=_interrupt_check)
        except InterruptedError:
            _raise_interrupt()
            raise

        wav_path = result["wav_path"]
        try:
            wav = load_npy_audio(wav_path)
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass
        audio = to_comfy_audio(wav, int(result.get("sample_rate", 24000)))
        info = result.get("info") or {}
        info_text = f"mode={mode} " + _format_info(info, result)
        return (audio, float(result.get("duration_s", len(wav) / 24000.0)), int(result.get("seed", 0)), info_text)


class Maya2NativeUnload:
    DESCRIPTION = "Unload the Maya2 Native worker and release its VRAM immediately."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"engine": ("MAYA2_ENGINE",)}}

    RETURN_TYPES = ("MAYA2_ENGINE", "STRING")
    RETURN_NAMES = ("engine", "status")
    FUNCTION = "unload"
    CATEGORY = "Maya2 Native"

    def unload(self, engine: Maya3Engine):
        engine.unload()
        return (engine, "Maya2 worker unloaded")


NODE_CLASS_MAPPINGS = {
    "Maya2NativeLoader": Maya2NativeLoader,
    "Maya2NativeTTS": Maya2NativeTTS,
    "Maya2NativeVoiceClone": Maya2NativeVoiceClone,
    "Maya2NativeTranscribe": Maya2NativeTranscribe,
    "Maya2NativeUnload": Maya2NativeUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Maya2NativeLoader": "Maya2 Native Model Loader",
    "Maya2NativeTTS": "Maya2 Native Text to Speech",
    "Maya2NativeVoiceClone": "Maya2 Native Voice Clone",
    "Maya2NativeTranscribe": "Maya2 Native Transcribe",
    "Maya2NativeUnload": "Maya2 Native Unload",
}
