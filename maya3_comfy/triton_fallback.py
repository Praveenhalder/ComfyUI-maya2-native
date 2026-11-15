"""Pure-PyTorch replacement for the Triton-only exponential sampler.

The Maya3 depth decoder's controlled-stop sampler
(:meth:`maya3.engine.Engine._cb0`) calls
``serve.fast_depth_kernels.row_exponential`` for every generated frame, even
when ``fused_depth=False``.  That kernel is Triton-only, and Triton on Windows
often cannot compile its CUDA helper (the ComfyUI embeddable Python ships no
``Python.h``), which makes the *safe* preset fail at engine warmup.

This module swaps in an exponential-race sampler built from deterministic
counter-based hashes.  It needs no Triton, runs on the GPU, never synchronises
with the host, and is reproducible for a given ``(seed, counter, level)``:

    E[row, c] = -log(U),  U = hash(seed[row], (counter[row] * 32 + level) * 4096 + c)

Sampling by ``argmax(q / E)`` is equivalent to sampling from ``q`` (the
exponential-race / Gumbel-max trick), so statistically this matches the Triton
kernel; only the random stream differs.

Enable/disable with ``MAYA3_COMFY_TORCH_SAMPLING``: ``1`` forces the fallback
on every platform, ``0`` always keeps Triton.  Unset means "fallback on Windows
or wherever Triton is unavailable, Triton elsewhere".
"""
from __future__ import annotations

import os
import sys

_MASK = 0xFFFFFFFF
_SCALE = 1.0 / 4294967296.0
_INDEX_CACHE: dict[tuple, "object"] = {}


def _log_default(message: str) -> None:
    print(f"[Maya2] {message}", flush=True)


def _mix32(x):
    """SplitMix32-style integer hash (tensor in, tensor out, values < 2**32)."""
    x = (x ^ (x >> 16)) * 0x7FEB352D
    x = x & _MASK
    x = (x ^ (x >> 15)) * 0x846CA68B
    x = x & _MASK
    return x ^ (x >> 16)


def _index(device, length: int):
    key = (device.type, device.index, int(length))
    cached = _INDEX_CACHE.get(key)
    if cached is None:
        import torch

        cached = torch.arange(int(length), device=device, dtype=torch.int64)
        _INDEX_CACHE[key] = cached
    return cached


def torch_row_exponential(B, V, seeds, ctrs, level=0):
    """Drop-in for ``serve.fast_depth_kernels.row_exponential``."""
    import torch

    if B <= 0:
        return torch.empty((0, int(V)), device=seeds.device, dtype=torch.float32)
    seed = seeds.to(torch.int64).reshape(-1)
    ctr = ctrs.to(torch.int64).reshape(-1)
    if seed.numel() == 1 and B != 1:
        seed = seed.expand(B)
    if ctr.numel() == 1 and B != 1:
        ctr = ctr.expand(B)

    base = ((ctr * 32 + int(level)) * 4096).reshape(B, 1)
    keys = (base + _index(seeds.device, V).reshape(1, V)) & _MASK
    mixed = _mix32(keys ^ seed.reshape(B, 1))
    # Map the 32-bit integer to (0, 1) strictly inside the interval.
    u = (mixed.to(torch.float32) + 0.5) * _SCALE
    return -torch.log(u.clamp_min(1e-12))


def _stub_kernels_module():
    """Register a stand-in for ``serve.fast_depth_kernels`` without Triton.

    The real module imports ``triton`` at import time, but ``maya3.engine``
    only needs ``row_exponential`` from it and the pure-PyTorch replacement is
    enough for the safe preset.  Putting the stub in ``sys.modules`` makes the
    engine's import succeed even when Triton is absent (the usual case on the
    Windows embeddable Python).
    """
    import types

    import serve

    name = "serve.fast_depth_kernels"
    module = types.ModuleType(name)
    module.row_exponential = torch_row_exponential
    sys.modules[name] = module
    setattr(serve, "fast_depth_kernels", module)
    return module


def install(force: bool | None = None, log=_log_default) -> bool:
    """Replace the Triton sampler with :func:`torch_row_exponential`.

    Returns True when the fallback was installed.  Idempotent.
    """
    mode = os.environ.get("MAYA3_COMFY_TORCH_SAMPLING", "").strip().lower()
    if mode in ("1", "true", "yes", "on", "always"):
        force = True
    elif mode in ("0", "false", "no", "off", "never"):
        return False
    elif force is None:
        force = sys.platform == "win32"
        if not force:
            # Triton is the default path, but the engine imports its kernels
            # unconditionally.  Use the fallback anywhere Triton cannot be
            # imported so the safe preset still works.
            try:
                import triton  # noqa: F401
            except Exception:
                force = True
    if not force:
        return False

    try:
        import serve.fast_depth_kernels as kernels
    except Exception as exc:
        # The module's only hard dependency is Triton.  If Triton itself is
        # missing (or unusable), use a stub module so the engine's lazy import
        # of row_exponential keeps working.
        try:
            import triton  # noqa: F401
        except Exception:
            _stub_kernels_module()
            log("using the pure-PyTorch exponential sampler (Triton not installed)")
            return True
        log(f"could not install the PyTorch sampling fallback: {exc}")
        return False

    if getattr(kernels, "row_exponential", None) is torch_row_exponential:
        return True
    kernels.row_exponential = torch_row_exponential
    log("using the pure-PyTorch exponential sampler (Triton not required)")
    return True
