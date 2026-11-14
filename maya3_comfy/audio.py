"""Audio helpers shared by the ComfyUI nodes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

SAMPLE_RATE = 24000


def to_comfy_audio(waveform: np.ndarray | torch.Tensor, sample_rate: int = SAMPLE_RATE) -> dict:
    """Convert mono float audio into ComfyUI's AUDIO dict (``[1, 1, T]``)."""
    if isinstance(waveform, torch.Tensor):
        tensor = waveform.detach().float().cpu()
    else:
        tensor = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))
    if tensor.ndim == 1:
        tensor = tensor[None, None, :]
    elif tensor.ndim == 2:
        tensor = tensor[None, :, :]
    elif tensor.ndim != 3:
        raise ValueError(f"unsupported waveform shape {tuple(tensor.shape)}")
    return {"waveform": tensor.clamp(-1.0, 1.0), "sample_rate": int(sample_rate)}


def load_npy_audio(path: str | Path) -> np.ndarray:
    audio = np.load(str(path), allow_pickle=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.reshape(-1)
    return audio


def audio_to_mono_array(audio: dict) -> tuple[np.ndarray, int]:
    """Return ``(mono float32 samples, sample_rate)`` for a ComfyUI AUDIO dict."""
    waveform = audio.get("waveform")
    if waveform is None:
        raise ValueError("reference AUDIO input has no 'waveform' tensor")
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    wav = waveform.detach().float().cpu()
    if wav.ndim == 3:
        wav = wav[0]
    if wav.ndim == 2:
        wav = wav.mean(dim=0) if wav.shape[0] <= 8 else wav.mean(dim=1)
    if wav.ndim != 1:
        raise ValueError(f"cannot convert AUDIO of shape {tuple(waveform.shape)} to mono")
    array = np.ascontiguousarray(wav.numpy(), dtype=np.float32)
    return array, int(audio.get("sample_rate", SAMPLE_RATE))


def save_audio_cache(audio: dict, directory: str | Path) -> tuple[Path, int]:
    """Save a reference clip under a content-addressed name.

    A stable path/size/mtime lets the engine's per-voice cache (keyed on the
    file stat) hit when the same clip is reused, instead of recomputing the
    Mimi codes and ECAPA embedding on every prompt.  The worker cleans files
    older than a day on startup.
    """
    import hashlib

    array, sample_rate = audio_to_mono_array(audio)
    digest = hashlib.sha1(array.tobytes()).hexdigest()[:20]
    path = Path(directory) / f"maya2_ref_{sample_rate}_{digest}.npy"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), array)
    return path, sample_rate


def save_audio_tensor(audio: dict, path: str | Path) -> tuple[int, int]:
    """Save a ComfyUI AUDIO value as mono float32 ``.npy``.

    Returns ``(num_samples, sample_rate)``.  Multiple channels are averaged;
    only the first item of a batched waveform is used.
    """
    array, sample_rate = audio_to_mono_array(audio)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), array)
    return int(array.shape[0]), sample_rate
