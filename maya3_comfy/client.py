"""Parent-side worker manager for the Maya2 Native ComfyUI nodes.

The ComfyUI process never imports the Maya3 engine.  A subprocess owns the
model; this module starts it, talks JSON-lines to it, forwards its logs, and
unloads it after the configured idle period (or when ComfyUI shuts down).
"""
from __future__ import annotations

import atexit
import collections
import gc
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from .paths import NODE_DIR, ensure_dirs, hf_cache_dir
from .runtime import worker_command

_LOG_PREFIX = "[Maya2]"
_LOAD_TIMEOUT = 3600.0
_GENERATE_TIMEOUT = 900.0
_CANCEL_POLL_S = 0.25


def _say(message: str) -> None:
    """Print without letting console encoding errors kill a reader thread."""
    stream = sys.stdout
    try:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        try:
            message = message.encode(encoding, "replace").decode(encoding, "replace")
        except Exception:
            pass
        print(message, flush=True)
    except Exception:
        try:
            buffer = getattr(stream, "buffer", None)
            if buffer is not None:
                buffer.write(message.encode("utf-8", "replace") + b"\n")
                buffer.flush()
        except Exception:
            pass



class WorkerError(RuntimeError):
    pass


def free_comfy_vram() -> None:
    """Ask ComfyUI to unload its models so the worker has room to load."""
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


class Maya3Worker:
    """A single worker.py subprocess."""

    def __init__(self, config: dict, python: str, site: Path | None, home: Path):
        self.config = config
        self.python = python
        self.site = site
        self.home = home
        self._proc: subprocess.Popen | None = None
        self._lock = threading.RLock()
        self._responses: queue.Queue = queue.Queue()
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=80)
        self._next_id = 0
        self._closed = False
        self.on_log = None

    # ------------------------------------------------------------------ process
    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        with self._lock:
            if self.alive:
                return
            self._spawn_locked()

    def _spawn_locked(self) -> None:
        if self._closed:
            raise WorkerError("worker has been closed")
        ensure_dirs()
        home = self.home
        python, cmd = worker_command(self.site, self.python)
        env = os.environ.copy()
        env.update(
            PYTHONUTF8="1",
            PYTHONIOENCODING="utf-8",
            HF_HOME=str(hf_cache_dir()),
            HF_HUB_CACHE=str(hf_cache_dir()),
            TOKENIZERS_PARALLELISM="false",
            HF_HUB_DISABLE_SYMLINKS_WARNING="1",
            HF_HUB_DISABLE_PROGRESS_BARS="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            MAYA3_COMFY_HOME=str(home),
        )
        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._responses = queue.Queue()
        self._stderr_tail.clear()
        _say(f"{_LOG_PREFIX} starting worker: {python}")
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=str(NODE_DIR),
                env=env,
                creationflags=creationflags,
            )
        except FileNotFoundError as exc:
            raise WorkerError(f"could not launch Maya2 worker ({python}): {exc}") from exc
        self._proc = proc
        responses = self._responses
        threading.Thread(target=self._read_stdout, args=(proc, responses), name="maya2-stdout", daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(proc,), name="maya2-stderr", daemon=True).start()

        deadline = time.monotonic() + 300.0
        while True:
            try:
                message = self._responses.get(timeout=1.0)
            except queue.Empty:
                if proc.poll() is not None:
                    self._kill_locked()
                    raise WorkerError("Maya2 worker exited during startup\n" + self._tail_text())
                if time.monotonic() > deadline:
                    self._kill_locked()
                    raise WorkerError("Maya2 worker did not start within 5 minutes\n" + self._tail_text())
                continue
            kind = message.get("type")
            if kind == "ready":
                _say(
                    f"{_LOG_PREFIX} worker ready (pid={message.get('pid')}, site={message.get('site') or 'venv'})"
                )
                return
            if kind == "fatal":
                self._kill_locked()
                raise WorkerError("Maya2 worker failed to initialise: " + message.get("error", "unknown") + "\n" + self._tail_text())
            # Anything else before "ready" is unexpected; keep waiting.

    def _read_stdout(self, proc: subprocess.Popen, responses: queue.Queue) -> None:
        if proc.stdout is None:
            responses.put({"type": "eof"})
            return
        try:
            for line in proc.stdout:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except Exception:
                    _say(f"{_LOG_PREFIX} {line}")
                    continue
                kind = message.get("type")
                if kind == "log":
                    _say(f"{_LOG_PREFIX} {message.get('message', '')}")
                else:
                    responses.put(message)
        except Exception as exc:
            self._stderr_tail.append(f"stdout reader error: {exc}")
        finally:
            responses.put({"type": "eof"})

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        if proc.stderr is None:
            return
        try:
            while True:
                line = proc.stderr.readline()
                if not line:
                    break
                line = line.rstrip("\r\n")
                if not line:
                    continue
                self._stderr_tail.append(line)
                _say(f"{_LOG_PREFIX} {line}")
        except Exception as exc:
            self._stderr_tail.append(f"stderr reader error: {exc}")

    def _tail_text(self) -> str:
        if not self._stderr_tail:
            return "(no worker output)"
        return "worker output (last lines):\n" + "\n".join(self._stderr_tail)

    # ------------------------------------------------------------------ protocol
    def request(
        self,
        op: str,
        payload: dict | None = None,
        timeout: float = _GENERATE_TIMEOUT,
        cancel_check=None,
    ) -> dict:
        with self._lock:
            if not self.alive:
                self._spawn_locked()
            proc = self._proc
            assert proc is not None and proc.stdin is not None
            self._next_id += 1
            req_id = self._next_id
            message = {"id": req_id, "op": op}
            if payload:
                message.update(payload)
            try:
                proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
                proc.stdin.flush()
            except Exception as exc:
                self._kill_locked()
                raise WorkerError(f"could not send request to Maya2 worker: {exc}") from exc

            deadline = time.monotonic() + timeout
            while True:
                try:
                    response = self._responses.get(timeout=_CANCEL_POLL_S)
                except queue.Empty:
                    if cancel_check is not None and cancel_check():
                        self._kill_locked()
                        raise InterruptedError("Maya2 generation cancelled")
                    if proc.poll() is not None:
                        self._responses.put({"type": "eof"})
                        raise WorkerError(
                            f"Maya2 worker exited (code {proc.returncode}) while running {op}\n"
                            + self._tail_text()
                        )
                    if time.monotonic() > deadline:
                        self._kill_locked()
                        raise WorkerError(f"Maya2 worker timed out after {timeout:.0f}s on {op}\n" + self._tail_text())
                    continue
                kind = response.get("type")
                if kind == "eof":
                    raise WorkerError(f"Maya2 worker closed the protocol stream during {op}\n" + self._tail_text())
                if kind == "error" and response.get("id") in (req_id, None):
                    raise WorkerError(str(response.get("error") or "unknown worker error"))
                if kind == "result":
                    if response.get("id") == req_id:
                        return response
                    continue
                if kind == "fatal":
                    raise WorkerError(str(response.get("error") or "worker failed"))

    # ------------------------------------------------------------------ teardown
    def _kill_locked(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        except Exception:
            pass

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if not self.alive:
                self._kill_locked()
                return
            try:
                proc = self._proc
                assert proc is not None and proc.stdin is not None
                self._next_id += 1
                req_id = self._next_id
                proc.stdin.write(json.dumps({"id": req_id, "op": "shutdown"}) + "\n")
                proc.stdin.flush()
                deadline = time.monotonic() + 15.0
                while time.monotonic() < deadline:
                    try:
                        message = self._responses.get(timeout=0.25)
                    except queue.Empty:
                        if not self.alive:
                            break
                        continue
                    if message.get("type") == "result" and message.get("id") == req_id:
                        break
            except Exception:
                pass
            finally:
                self._kill_locked()


class Maya3Engine:
    """Handle passed between ComfyUI nodes; owns one worker subprocess."""

    def __init__(self, config: dict, python: str, site: Path | None, home: Path):
        self.config = dict(config)
        self.python = python
        self.site = site
        self.home = home
        self._lock = threading.RLock()
        self._worker: Maya3Worker | None = None
        self.info: dict = {}
        self._last_used = time.monotonic()
        self._idle_stop = threading.Event()
        self._idle_thread = threading.Thread(target=self._idle_loop, name="maya2-idle", daemon=True)
        self._idle_thread.start()

    # ------------------------------------------------------------------ loading
    def _ensure_locked(self, cancel_check=None) -> None:
        if self._worker is not None and self._worker.alive and self.info:
            return
        if self._worker is not None:
            try:
                self._worker.close()
            except Exception:
                pass
            self._worker = None
        if self.config.get("free_vram_first", True):
            free_comfy_vram()
        worker = Maya3Worker(self.config, self.python, self.site, self.home)
        worker.start()
        try:
            info = worker.request(
                "load",
                {"config": self.config},
                timeout=_LOAD_TIMEOUT,
                cancel_check=cancel_check,
            )
        except BaseException:
            try:
                worker.close()
            except Exception:
                pass
            raise
        self._worker = worker
        self.info = info

    def start(self, cancel_check=None) -> dict:
        with self._lock:
            self._last_used = time.monotonic()
            self._ensure_locked(cancel_check=cancel_check)
            return self.info

    def run(self, op: str, payload: dict | None = None, timeout: float = _GENERATE_TIMEOUT, cancel_check=None) -> dict:
        with self._lock:
            self._last_used = time.monotonic()
            self._ensure_locked(cancel_check=cancel_check)
            assert self._worker is not None
            try:
                return self._worker.request(op, payload, timeout=timeout, cancel_check=cancel_check)
            finally:
                self._last_used = time.monotonic()

    # ------------------------------------------------------------------ idle
    def _idle_loop(self) -> None:
        while not self._idle_stop.wait(5.0):
            idle_s = int(self.config.get("idle_unload_s", 600) or 0)
            if idle_s <= 0:
                continue
            if time.monotonic() - self._last_used < idle_s:
                continue
            with self._lock:
                if self._idle_stop.is_set():
                    return
                if time.monotonic() - self._last_used < idle_s:
                    continue
                if self._worker is not None:
                    _say(f"{_LOG_PREFIX} idle for {idle_s}s, unloading Maya2 model")
                    self.unload()

    def unload(self) -> None:
        """Close the worker/model but keep this engine handle reusable."""
        with self._lock:
            worker = self._worker
            self._worker = None
            self.info = {}
            self._last_used = time.monotonic()
            if worker is not None:
                try:
                    worker.close()
                except Exception:
                    pass

    def stop(self) -> None:
        self._idle_stop.set()
        self.unload()
        if self._idle_thread.is_alive() and threading.current_thread() is not self._idle_thread:
            self._idle_thread.join(timeout=2.0)


# ---------------------------------------------------------------------- registry
_ENGINES: dict[str, Maya3Engine] = {}
_ENGINES_LOCK = threading.Lock()


def get_engine(config: dict, python: str, site: Path | None, home: Path) -> Maya3Engine:
    """One live engine per configuration; a new configuration evicts the old one."""
    key = json.dumps(
        {
            "config": {k: config[k] for k in sorted(config)},
            "python": python,
            "site": str(site) if site else "",
            "home": str(home),
        },
        sort_keys=True,
        default=str,
    )
    with _ENGINES_LOCK:
        engine = _ENGINES.get(key)
        if engine is not None:
            return engine
        for old in list(_ENGINES.values()):
            try:
                old.stop()
            except Exception:
                pass
        _ENGINES.clear()
        engine = Maya3Engine(config, python, site, home)
        _ENGINES[key] = engine
        return engine


def reset_engines() -> None:
    with _ENGINES_LOCK:
        engines = list(_ENGINES.values())
        _ENGINES.clear()
    for engine in engines:
        try:
            engine.stop()
        except Exception:
            pass


@atexit.register
def _shutdown_engines() -> None:
    reset_engines()
