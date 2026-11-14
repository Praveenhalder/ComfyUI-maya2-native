"""Filesystem layout used by the ComfyUI integration.

Everything the node needs at runtime lives under::

    ComfyUI/models/maya3/
        hf/                         Hugging Face cache (release + Mimi + ECAPA)
        runtime/<py>-torch<x>/site  shadow site-packages for the worker
        tmp/                        audio hand-off files

Override the base directory with ``MAYA3_COMFY_HOME`` (or the legacy
``MAYA2_COMFY_HOME``), and the ComfyUI models directory with
``MAYA3_COMFY_MODELS_DIR``.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# ``.../custom_nodes/ComfyUI-maya2-native/maya3_comfy/paths.py`` -> repo root
NODE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_REPO_ID = "maya-research/maya2-native"


def _env_path(*names: str) -> Path | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return Path(os.path.expanduser(value)).resolve()
    return None


def comfy_models_dir() -> Path:
    """ComfyUI's ``models`` directory, without requiring ``folder_paths``."""
    override = _env_path("MAYA3_COMFY_MODELS_DIR", "MAYA2_COMFY_MODELS_DIR")
    if override is not None:
        return override
    try:
        import folder_paths  # type: ignore

        return Path(folder_paths.models_dir).resolve()
    except Exception:
        pass
    for parent in NODE_DIR.parents:
        if (parent / "models").is_dir() and (parent / "custom_nodes").is_dir():
            return (parent / "models").resolve()
    return (NODE_DIR.parent.parent / "models").resolve()


def home_dir() -> Path:
    override = _env_path("MAYA3_COMFY_HOME", "MAYA2_COMFY_HOME")
    return override if override is not None else comfy_models_dir() / "maya3"


def hf_cache_dir() -> Path:
    return home_dir() / "hf"


def runtime_root() -> Path:
    return home_dir() / "runtime"


def tmp_dir() -> Path:
    return home_dir() / "tmp"


def _torch_version() -> str:
    try:
        from importlib.metadata import version

        return version("torch").split("+", 1)[0]
    except Exception:
        return "unknown"


def runtime_tag() -> str:
    """Tag that keeps shadow sites apart for different Python/torch builds."""
    py = f"py{sys.version_info.major}{sys.version_info.minor}"
    return f"{py}-torch{_torch_version()}"


def shadow_site_dir() -> Path:
    return runtime_root() / runtime_tag() / "site"


def worker_script() -> Path:
    return Path(__file__).resolve().parent / "worker.py"


def ensure_dirs() -> None:
    for p in (home_dir(), hf_cache_dir(), runtime_root(), tmp_dir()):
        p.mkdir(parents=True, exist_ok=True)


def normalize_dist_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.lower())
