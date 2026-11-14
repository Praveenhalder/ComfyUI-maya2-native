"""ComfyUI entry point for Maya2 Native (Maya3 TTS).

The heavy engine (transformers 5.x, speechbrain, CUDA graphs) runs in an
isolated worker subprocess; importing this package does not change ComfyUI's
own Python environment.
"""

from __future__ import annotations

from .maya3_comfy.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
