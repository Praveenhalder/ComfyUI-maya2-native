"""Prepare the isolated worker interpreter for Maya2/Maya3.

ComfyUI ships ``transformers`` 4.x while the Maya3 engine needs the 5.x cache
API (``transformers.cache_utils.DynamicLayer``).  Python can only load one
version of a package per process, so the model runs in a subprocess and this
module prepares a *shadow site-packages* directory containing only the
conflicting packages.  The worker prepends that directory to ``sys.path``;
ComfyUI's own process is never touched.

The shadow directory lives at ``ComfyUI/models/maya3/runtime/<tag>/site`` and
is populated on demand with ``pip install --target`` using ComfyUI's own
Python.  Torch/CUDA are reused from ComfyUI (they are not copied).

Escape hatches:
    MAYA3_COMFY_PYTHON   use a different Python (e.g. a hand-made venv).
    MAYA3_COMFY_SITE     use a specific shadow site-packages directory.
    MAYA3_COMFY_SKIP_INSTALL=1
                         never run pip; fail with the missing package list.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from importlib.metadata import distributions
from pathlib import Path

from .paths import (
    NODE_DIR,
    ensure_dirs,
    hf_cache_dir,
    home_dir,
    normalize_dist_name,
    shadow_site_dir,
)

# Versions the Maya3 engine is built against.  tokenizers/hub follow from
# transformers 5.6.2; speechbrain is only needed for voice cloning.
PINNED = {
    "transformers": "5.6.2",
    "tokenizers": "0.22.2",
    "huggingface-hub": "1.33.0",
    "speechbrain": "1.1.2",
}

INSTALL_SPECS = [
    "transformers==5.6.2",
    "tokenizers==0.22.2",
    "huggingface-hub==1.33.0",
    "speechbrain==1.1.2",
]

# speechbrain declares torch/torchaudio as requirements.  Installing it with
# ``--no-deps`` and these explicit non-torch dependencies keeps pip from copying
# a CPU torch into the shadow site, where it would shadow ComfyUI's CUDA build.
SPEECHBRAIN_DEPS = [
    "hyperpyyaml",
    "joblib",
    "scipy",
    "sentencepiece",
    "soundfile",
    "tqdm",
]

# Binary packages that must always come from ComfyUI's interpreter: they are
# built against ComfyUI's torch/CUDA, so a copy in the shadow site breaks the
# matching torchvision (``operator torchvision::nms does not exist``).
HOST_PROVIDED = ("torch", "torchaudio", "torchvision", "torchgen", "functorch")

MIN_TORCH = "2.7.1"
MIN_PYTHON = (3, 10)


def _log_default(message: str) -> None:
    print(f"[Maya2] {message}", flush=True)


def _dist_versions(site: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    if not site.is_dir():
        return found
    try:
        for dist in distributions(path=[str(site)]):
            name = dist.metadata.get("Name")
            if name:
                found[normalize_dist_name(name)] = dist.version
    except Exception:
        pass
    return found


def _host_dist_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _version_tuple(value: str) -> tuple[int, ...]:
    parts = []
    for piece in value.split("+"):
        for bit in piece.split("."):
            bit = "".join(ch for ch in bit if ch.isdigit())
            if bit:
                parts.append(int(bit))
    return tuple(parts)


def site_report(site: Path | None = None) -> tuple[dict[str, str], list[str]]:
    """Return (versions found in the shadow site, missing/mismatched specs)."""
    site = site or shadow_site_dir()
    found = _dist_versions(site)
    missing = []
    for name, want in PINNED.items():
        got = found.get(name)
        if got != want:
            missing.append(f"{name}=={want} (found {got or 'nothing'})")
    return found, missing


def _check_host() -> None:
    if sys.version_info < MIN_PYTHON:
        raise RuntimeError(
            f"Maya2 needs Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}; this ComfyUI runs "
            f"{sys.version_info.major}.{sys.version_info.minor}. Set MAYA3_COMFY_PYTHON "
            "to a compatible Python executable (see COMFYUI.md)."
        )
    torch_version = _host_dist_version("torch")
    if torch_version is None:
        raise RuntimeError(
            "Maya2 needs PyTorch, but no 'torch' distribution was found in this "
            "Python environment. Install ComfyUI's CUDA torch first."
        )
    if _version_tuple(torch_version) < _version_tuple(MIN_TORCH):
        raise RuntimeError(
            f"Maya2 needs torch >= {MIN_TORCH}; this environment has {torch_version}. "
            "Upgrade ComfyUI's torch or set MAYA3_COMFY_PYTHON to a venv with a "
            "compatible CUDA torch build."
        )
    if _host_dist_version("torchaudio") is None:
        raise RuntimeError(
            "Maya2 needs torchaudio (speechbrain imports it for voice cloning). "
            "Install the torchaudio build that matches this ComfyUI's torch."
        )
    if not (NODE_DIR / "maya3").is_dir():
        raise RuntimeError(f"Maya3 engine package not found under {NODE_DIR}")


def _drop_path(path: Path) -> bool:
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
        return True
    except OSError:
        return False


def prune_host_packages(site: Path, log=None) -> list[str]:
    """Remove torch builds pip may have copied into the shadow site.

    ``pip install --target`` resolves speechbrain's ``torch``/``torchaudio``
    requirements even though the worker is meant to reuse ComfyUI's CUDA
    builds.  A CPU torch in the shadow site then shadows ComfyUI's torch while
    torchvision is still loaded from ComfyUI, so its C extension refuses to
    initialise and transformers dies with
    ``RuntimeError: operator torchvision::nms does not exist``.  ComfyUI's
    torch, torchaudio and torchvision are a matched set; keep them together.
    """
    removed: list[str] = []
    if not site.is_dir():
        return removed
    host_names = set(HOST_PROVIDED)
    for entry in sorted(site.iterdir(), key=lambda p: p.name):
        lower = entry.name.lower()
        if lower.endswith(".dist-info"):
            distribution = lower[: -len(".dist-info")].split("-", 1)[0]
        else:
            distribution = lower
        if distribution in host_names:
            if _drop_path(entry):
                removed.append(entry.name)
            elif log is not None:
                log(
                    f"could not remove the stale {entry.name} from the Maya2 shadow site; "
                    "restart ComfyUI to release it, or delete it manually"
                )
    if removed and log is not None:
        log(
            "removed torch packages from the Maya2 shadow site "
            f"({', '.join(removed)}); using ComfyUI's matched CUDA builds"
        )
    return removed


def _pip_install(site: Path, specs: list[str], log, *, no_deps: bool = False) -> None:
    site.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--target",
        str(site),
        "--upgrade",
        "--no-compile",
        "--disable-pip-version-check",
        "--no-warn-script-location",
    ]
    if no_deps:
        cmd.append("--no-deps")
    cmd.extend(specs)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Could not launch 'pip' with {sys.executable}. Install pip or set "
            "MAYA3_COMFY_SITE to a prepared site-packages directory."
        ) from exc
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        tail.append(line)
        tail = tail[-40:]
        log("  " + line)
    code = proc.wait()
    if code != 0:
        raise RuntimeError(
            "pip failed to prepare the Maya2 runtime (exit code "
            f"{code}). Last output:\n" + "\n".join(tail)
        )


def _run_pip(site: Path, log) -> None:
    log("installing isolated Maya2 runtime (one-time, ~1-3 minutes)")
    log("  " + " ".join(INSTALL_SPECS))
    # Install speechbrain itself without dependencies (it would drag a CPU
    # torch into the shadow site), then add its non-torch requirements.
    plain_specs = [spec for spec in INSTALL_SPECS if not spec.startswith("speechbrain")]
    _pip_install(site, plain_specs, log)
    _pip_install(site, ["speechbrain==1.1.2"], log, no_deps=True)
    _pip_install(site, SPEECHBRAIN_DEPS, log)
    prune_host_packages(site, log)


def ensure_runtime(log=None) -> Path | None:
    """Return the shadow site to put on the worker's ``sys.path``.

    Returns ``None`` when the user pointed ``MAYA3_COMFY_PYTHON`` at a
    self-contained environment (the worker then imports from that venv).
    """
    log = log or _log_default
    ensure_dirs()

    explicit_site = os.environ.get("MAYA3_COMFY_SITE")
    explicit_python = os.environ.get("MAYA3_COMFY_PYTHON")

    if explicit_site:
        site = Path(os.path.expanduser(explicit_site)).resolve()
        _, missing = site_report(site)
        if missing:
            raise RuntimeError(
                "MAYA3_COMFY_SITE does not provide the pinned Maya2 packages: "
                + ", ".join(missing)
            )
        log(f"using MAYA3_COMFY_SITE={site}")
        return site

    if explicit_python:
        log(
            "using MAYA3_COMFY_PYTHON=%s (make sure it already has the Maya2 "
            "dependencies installed)" % explicit_python
        )
        return None

    _check_host()
    site = shadow_site_dir()
    prune_host_packages(site, log)
    _, missing = site_report(site)
    if not missing:
        return site

    if os.environ.get("MAYA3_COMFY_SKIP_INSTALL", "").strip().lower() in ("1", "true", "yes", "on"):
        raise RuntimeError(
            "Maya2 runtime is incomplete and MAYA3_COMFY_SKIP_INSTALL is set. Missing: "
            + ", ".join(missing)
        )

    _run_pip(site, log)
    found, missing = site_report(site)
    if missing:
        raise RuntimeError(
            "Maya2 runtime installation finished but the pinned packages are "
            "still missing: " + ", ".join(missing)
        )
    try:
        (site.parent / "runtime.json").write_text(
            json.dumps(
                {
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "python": sys.executable,
                    "python_version": sys.version,
                    "torch": _host_dist_version("torch"),
                    "packages": found,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass
    log(f"runtime ready: {site}")
    return site


def worker_command(site: Path | None, python: str | None = None) -> tuple[str, list[str]]:
    """Return (python executable, argv for worker.py)."""
    python = python or os.environ.get("MAYA3_COMFY_PYTHON") or sys.executable
    cmd = [
        python,
        "-u",
        str(NODE_DIR / "maya3_comfy" / "worker.py"),
        "--home",
        str(home_dir()),
        "--cache",
        str(hf_cache_dir()),
    ]
    if site is not None:
        cmd += ["--site", str(site)]
    return python, cmd
