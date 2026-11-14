"""ComfyUI integration for Maya2 Native (Maya3 engine).

This package is imported by ComfyUI's parent process only.  Nothing in here
imports ``transformers``/``maya3`` at module level: the model runs in a
subprocess with its own shadow ``sys.path`` (see ``runtime.py`` / ``worker.py``).
"""

from __future__ import annotations

__all__ = ["paths", "runtime", "client", "audio", "nodes", "transcribe", "triton_fallback"]
