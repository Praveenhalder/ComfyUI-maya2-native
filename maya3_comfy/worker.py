"""Maya2 Native worker: runs the Maya3 engine in an isolated interpreter.

Protocol: one JSON object per line on stdout.  Every other byte (logs,
download progress bars, tracebacks) goes to stderr, which the parent forwards
to the ComfyUI console.

Requests:
    {"id": 1, "op": "load",     "config": {...}}
    {"id": 2, "op": "generate", "text": ..., "language": "en", ...}
    {"id": 3, "op": "clone",    "ref_audio_path": ..., ...}
    {"id": 4, "op": "unload"}
    {"id": 5, "op": "status"}
    {"id": 6, "op": "shutdown"}

Replies:
    {"type": "ready",  ...}
    {"type": "result", "id": 2, ...}
    {"type": "error",  "id": 2, "error": "..."}
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
import traceback
import uuid
from pathlib import Path

import numpy as np

PROTO = sys.stdout

SR = 24000
FRAME = 1920
FRAMES_PER_S = SR / FRAME  # 12.5 Mimi frames per second
DEFAULT_REPO_ID = "maya-research/maya2-native"


def log(*args) -> None:
    print(*args, file=sys.stderr, flush=True)


def send(**message) -> None:
    PROTO.write(json.dumps(message, ensure_ascii=False) + "\n")
    PROTO.flush()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def clear_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def cleanup_tmp(tmp: Path, max_age_hours: float = 24.0) -> None:
    try:
        cutoff = time.time() - max_age_hours * 3600
        for p in tmp.glob("*"):
            try:
                if p.is_file() and p.stat().st_mtime < cutoff:
                    p.unlink()
            except OSError:
                pass
    except Exception:
        pass


def _preset_config(cfg: dict) -> tuple[dict, str]:
    preset = str(cfg.get("performance") or "auto").lower()
    if preset == "auto":
        preset = "safe" if sys.platform == "win32" else "fast"
    if preset not in ("fast", "safe"):
        raise ValueError(f"unknown performance preset {preset!r}; use auto, fast or safe")

    engine_cfg = {
        "max_seats": int(cfg.get("max_seats", 1)),
        "capacity": int(cfg.get("capacity", 1024)),
        "depth_dtype": str(cfg.get("depth_dtype", "fp16")),
        "free_reference_depth": True,
    }
    if preset == "fast":
        engine_cfg.update(
            use_graphs=True,
            fused_depth=True,
            fuse_rope=True,
            attention="triton",
            fuse_proj=False,
        )
    else:
        engine_cfg.update(
            use_graphs=False,
            fused_depth=False,
            fuse_rope=False,
            attention="gather",
            fuse_proj=False,
        )
    return engine_cfg, preset


def _safe_config(base: dict) -> dict:
    safe = dict(base)
    safe.update(
        use_graphs=False,
        fused_depth=False,
        fuse_rope=False,
        attention="gather",
        fuse_proj=False,
    )
    if safe.get("capacity", 1024) > 1024:
        safe["capacity"] = 1024
    return safe


class Worker:
    def __init__(self, home: Path, cache: Path):
        self.home = home
        self.cache = cache
        self.tmp = home / "tmp"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.tts = None
        self.load_info: dict = {}
        self.engine_cfg: dict = {}
        self.preset = ""
        cleanup_tmp(self.tmp)

    # ------------------------------------------------------------------ model
    def load(self, payload: dict) -> dict:
        from maya3 import hub
        from maya3.api import Maya3TTS

        cfg = dict(payload.get("config") or {})
        repo = str(cfg.get("repo_id") or DEFAULT_REPO_ID)
        revision = str(cfg.get("revision") or "") or None
        auto_download = bool(cfg.get("auto_download", True))

        log(f"resolving {repo}@{revision or 'main'} (auto_download={auto_download})")
        log(f"HF cache: {self.cache}")
        try:
            paths = hub.resolve(
                repo,
                cache_dir=str(self.cache),
                revision=revision,
                local_files_only=not auto_download,
            )
        except Exception as exc:
            hint = (
                " Downloads are disabled (auto_download=False); enable the node's "
                "auto_download input or pre-populate the HF cache."
                if not auto_download
                else ""
            )
            raise RuntimeError(f"could not resolve/download {repo}: {exc}.{hint}") from exc

        log(f"release: {paths['ckpt']}")
        log(f"backbone: {paths['backbone_dir']}")
        log(f"mimi: {paths['mimi_dir']}")
        log(f"ecapa: {paths['ecapa_dir']}")

        engine_cfg, preset = _preset_config(cfg)
        attempts = [engine_cfg]
        if preset != "safe":
            attempts.append(_safe_config(engine_cfg))

        last_error: Exception | None = None
        for index, attempt in enumerate(attempts):
            use_preset = preset if index == 0 else "safe"
            candidate = None
            try:
                clear_cuda()
                log(
                    "loading engine "
                    f"(preset={use_preset}, attention={attempt['attention']}, "
                    f"graphs={attempt['use_graphs']}, capacity={attempt['capacity']}, "
                    f"max_seats={attempt['max_seats']}, depth={attempt['depth_dtype']})"
                )
                started = time.perf_counter()
                candidate = Maya3TTS.from_pretrained(
                    paths["ckpt"],
                    backbone_dir=paths["backbone_dir"],
                    mimi_dir=paths["mimi_dir"],
                    ecapa_dir=paths["ecapa_dir"],
                    device="cuda",
                    voices=False,
                    **attempt,
                )
                if paths.get("voices_json"):
                    ids = candidate.load_named_voices(paths["voices_json"])
                    log(f"named voices loaded: {len(ids)}")
                self.tts = candidate
                self.engine_cfg = dict(attempt)
                self.preset = use_preset
                self.load_s = time.perf_counter() - started
                break
            except Exception as exc:
                last_error = exc
                log(f"engine load failed with preset={use_preset}: {type(exc).__name__}: {exc}")
                log(traceback.format_exc())
                try:
                    if candidate is not None:
                        candidate.close()
                except Exception:
                    pass
                clear_cuda()
                if index + 1 < len(attempts):
                    log("retrying with the portable safe configuration")
                else:
                    raise

        if self.tts is None:
            raise RuntimeError(f"Maya2 engine failed to load: {last_error}")

        import torch
        import transformers

        health = {}
        try:
            health = jsonable(self.tts.engine.health())
        except Exception:
            pass
        voices = list(self.tts.voices)
        self.load_info = {
            "repo": repo,
            "revision": revision or "main",
            "model_dir": paths["ckpt"],
            "paths": jsonable(paths),
            "preset": self.preset,
            "engine_config": jsonable(self.engine_cfg),
            "load_s": round(float(self.load_s), 2),
            "warmup_s": round(float(getattr(self.tts.engine, "warmup_s", 0.0)), 2),
            "voices": voices,
            "device": "cuda",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "health": health,
        }
        log(
            f"engine ready in {self.load_info['load_s']:.1f} s "
            f"({len(voices)} voices, preset={self.preset})"
        )
        return self.load_info

    # ------------------------------------------------------------------ synthesis
    def _sampling(self, payload: dict) -> dict:
        seed = int(payload.get("seed") or 0)
        if seed <= 0:
            seed = random.randrange(1, 2**31 - 1)
        return {
            "seed": seed,
            "temperature": float(payload.get("temperature", 0.9)),
            "top_k": int(payload.get("top_k", 64)),
            "top_p": float(payload.get("top_p", 1.0)),
            "repetition_penalty": float(payload.get("repetition_penalty", 0.8)),
        }

    def _synthesize(self, payload: dict, clone: bool) -> dict:
        if self.tts is None:
            raise RuntimeError("Maya2 model is not loaded; run the model loader first")

        text = str(payload.get("text") or "").strip()
        if not text:
            raise ValueError("text is empty")
        language = str(payload.get("language") or "en")
        sampling = self._sampling(payload)

        max_seconds = float(payload.get("max_seconds") or 0.0)
        max_frames = max(1, int(round(max_seconds * FRAMES_PER_S))) if max_seconds > 0 else None

        kwargs = {
            "lang": language,
            "max_frames": max_frames,
            "return_info": True,
            "normalize": bool(payload.get("normalize_text", True)),
            **sampling,
        }
        if clone:
            ref_path = Path(str(payload.get("ref_audio_path") or ""))
            if not ref_path.is_file():
                raise FileNotFoundError(f"reference audio not found: {ref_path}")
            ref_audio = np.load(str(ref_path), allow_pickle=False).astype(np.float32, copy=False)
            if ref_audio.ndim > 1:
                ref_audio = ref_audio.reshape(-1)
            ref_sr = int(payload.get("ref_sample_rate") or SR)
            mode = str(payload.get("mode") or "icl_spk")
            ref_text = str(payload.get("ref_text") or "").strip() or None
            if mode in ("icl", "icl_spk") and not ref_text:
                raise ValueError(f"mode={mode} requires the reference transcript (ref_text)")
            kwargs.update(
                voice=None,
                ref_audio=(ref_audio, ref_sr),
                ref_text=ref_text,
                ref_lang=str(payload.get("ref_language") or language),
                mode=mode,
            )
        else:
            kwargs.update(voice=str(payload.get("voice") or "plain"), ref_audio=None)

        started = time.perf_counter()
        wav, info = self.tts.generate(text, **kwargs)
        wall_s = time.perf_counter() - started
        wav = np.ascontiguousarray(wav, dtype=np.float32)
        if wav.size == 0:
            raise RuntimeError("the engine produced no audio")

        out = self.tmp / f"maya2_{uuid.uuid4().hex}.npy"
        np.save(str(out), wav)
        duration = float(wav.size) / SR
        log(
            f"synthesized {duration:.2f} s in {wall_s:.2f} s "
            f"(seed={sampling['seed']}, reason={info.get('reason')})"
        )
        return {
            "wav_path": str(out),
            "sample_rate": SR,
            "duration_s": round(duration, 4),
            "wall_s": round(wall_s, 3),
            "seed": sampling["seed"],
            "info": jsonable(info),
        }

    # ------------------------------------------------------------------ lifecycle
    def unload(self) -> dict:
        if self.tts is not None:
            try:
                self.tts.close()
            except Exception as exc:
                log(f"engine close failed: {exc}")
            self.tts = None
        clear_cuda()
        self.load_info = {}
        return {"unloaded": True}

    def status(self) -> dict:
        if self.tts is None:
            return {"loaded": False}
        try:
            health = jsonable(self.tts.engine.health())
        except Exception as exc:
            health = {"error": str(exc)}
        return {"loaded": True, "preset": self.preset, "engine_config": self.engine_cfg, "health": health}

    def handle(self, message: dict) -> tuple[dict, bool]:
        op = message.get("op")
        if op == "load":
            return self.load(message), False
        if op == "generate":
            return self._synthesize(message, clone=False), False
        if op == "clone":
            return self._synthesize(message, clone=True), False
        if op == "unload":
            return self.unload(), False
        if op == "status":
            return self.status(), False
        if op == "shutdown":
            self.unload()
            return {"bye": True}, True
        raise ValueError(f"unknown op {op!r}")


def _disable_broken_torchvision() -> None:
    """Hide torchvision when it cannot be used with the loaded torch.

    transformers 5.x imports ComfyUI's torchvision from its vision helpers even
    though the Maya3 engine only needs the Mimi audio codec.  When the installed
    torchvision was built for a different torch, that import fails inside
    ``torch.library`` with ``operator torchvision::nms does not exist`` and
    takes the whole transformers import down.  Pretending torchvision is not
    installed lets transformers take its audio-only path.
    """
    try:
        import torchvision  # noqa: F401
    except Exception as exc:
        for name in [n for n in sys.modules if n == "torchvision" or n.startswith("torchvision.")]:
            sys.modules.pop(name, None)
        sys.modules["torchvision"] = None
        log(
            f"torchvision is unusable ({type(exc).__name__}: {exc}); "
            "disabling it for transformers"
        )


def preload_runtime() -> None:
    """Import the transformers surface before speechbrain is ever touched.

    On Python 3.13, ComfyUI's ``hqq`` package runs ``torch.compile`` while
    ``transformers.cache_utils`` imports.  That import path calls
    ``inspect.getframeinfo``, which trips speechbrain's lazy-module
    ``__getattr__`` into infinite recursion if speechbrain is already loaded.
    Resolving the 5.x cache API here (before any speechbrain import) keeps the
    worker deterministic; voice cloning imports speechbrain only later.
    """
    import torch

    log(f"torch {torch.__version__} | cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"gpu: {torch.cuda.get_device_name(0)} | vram={torch.cuda.get_device_properties(0).total_memory / 2**30:.2f} GiB")

    _disable_broken_torchvision()

    # ComfyUI's optional hqq package is irrelevant here and its import-time
    # torch.compile() is a large, fragile side effect.  Hide it before the
    # transformers cache module probes for it (cache_utils imports the name
    # from transformers.utils, so patch both export points).
    try:
        import transformers.utils as _tu
        import transformers.utils.import_utils as _iu

        _no_hqq = lambda *a, **k: False  # noqa: E731
        _tu.is_hqq_available = _no_hqq
        _iu.is_hqq_available = _no_hqq
    except Exception:
        pass

    import transformers

    log(f"transformers {transformers.__version__} | {Path(transformers.__file__).parent}")
    from transformers import AutoConfig, AutoModel, MimiModel  # noqa: F401
    from transformers.cache_utils import DynamicLayer  # noqa: F401

    import maya3.api  # noqa: F401  (light module: no transformers import at top level)

    # The depth decoder's cb0 sampler is Triton-only; swap in the GPU torch
    # implementation on Windows (or when MAYA3_COMFY_TORCH_SAMPLING=1).
    from maya3_comfy.triton_fallback import install as install_sampling_fallback

    install_sampling_fallback(log=log)

    log("transformers 5.x cache API available")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--site", default="")
    args = parser.parse_args()

    # Keep a private handle for the protocol, then send every stray print to
    # stderr so it cannot corrupt the JSON stream.
    global PROTO
    PROTO = sys.stdout
    sys.stdout = sys.stderr
    for stream in (sys.stdin, sys.stderr, PROTO):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    home = Path(args.home).resolve()
    cache = Path(args.cache).resolve()
    home.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_HUB_CACHE", str(cache))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    repo_root = Path(__file__).resolve().parent.parent

    # Bind ComfyUI's torch before the shadow site joins sys.path.  The shadow
    # site is only meant to override transformers/tokenizers/speechbrain; a
    # stale torch copy inside it must not shadow ComfyUI's CUDA build (and the
    # torchvision compiled against it).
    try:
        import torch  # noqa: F401
    except Exception as exc:
        log(f"could not preload ComfyUI torch: {exc}")

    if args.site:
        sys.path.insert(0, str(Path(args.site).resolve()))
    sys.path.insert(0, str(repo_root))

    try:
        preload_runtime()
    except Exception as exc:
        log(traceback.format_exc())
        send(type="fatal", error=f"{type(exc).__name__}: {exc}")
        return 1

    send(
        type="ready",
        home=str(home),
        cache=str(cache),
        site=args.site,
        pid=os.getpid(),
        python=sys.version,
    )

    worker = Worker(home, cache)
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except Exception as exc:
                log(f"bad request line: {exc}")
                send(type="error", id=None, error=f"bad JSON request: {exc}")
                continue
            req_id = message.get("id")
            try:
                result, stop = worker.handle(message)
                send(type="result", id=req_id, **result)
                if stop:
                    break
            except Exception as exc:
                log(traceback.format_exc())
                send(type="error", id=req_id, error=f"{type(exc).__name__}: {exc}")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            worker.unload()
        except Exception:
            pass
        log("worker exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
