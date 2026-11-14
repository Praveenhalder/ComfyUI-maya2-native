# Maya2 Native for ComfyUI

ComfyUI nodes for **Maya2 Native** (the Maya3 TTS engine): 12 languages, 20 built-in voices,
zero-shot voice cloning, and automatic download of the weights from
[`maya-research/maya2-native`](https://huggingface.co/maya-research/maya2-native).

The upstream engine is *not* modified beyond one small `EngineConfig` addition. ComfyUI keeps its
own Python environment; the model runs in an isolated worker process.

---

## Requirements

- ComfyUI with **Python >= 3.10** and **PyTorch >= 2.7.1 + torchaudio** (CUDA build).
  The Windows portable build with torch 2.9.1 + cu130 is what this integration was tested on.
- An NVIDIA GPU. **8 GB VRAM works** with `max_seats=1`, `capacity=1024` and the `safe` preset;
  12 GB+ is comfortable. 6 GB is not enough for the 2.5B model.
- ~6 GB free disk for the model + ~200 MB for the isolated runtime.
- Internet on first use (unless the cache is pre-populated).

The root `requirements.txt` exists for the optional **Transcribe** node and contains only
`faster-whisper>=1.2.0`. Install it with ComfyUI's Python if the node reports the package missing:

```bat
python_embeded\python.exe -m pip install -r custom_nodes\ComfyUI-maya2-native\requirements.txt
```

Do **not** add torch, torchaudio, transformers, tokenizers or huggingface-hub to that file, and do
not upgrade ComfyUI's global `transformers` to 5.x; that would break other nodes and ComfyUI's text
encoders. The Maya3 engine's transformers 5.x lives in its own shadow runtime (see below).

---

## Install

1. Put this directory under `ComfyUI/custom_nodes/` (the folder name can stay
   `ComfyUI-maya2-native`).
2. *(Optional)* Install the transcription dependency:
   `python_embeded\python.exe -m pip install -r custom_nodes\ComfyUI-maya2-native\requirements.txt`.
   Skip this if you do not need the **Maya2 Native Transcribe** node.
3. Restart ComfyUI.
4. Add **Maya2 Native Model Loader** and queue once.

The first run does two things automatically:

- installs the isolated runtime into `ComfyUI/models/maya3/runtime/<py>-torch<x>/site`
  (`transformers 5.6.2`, `tokenizers 0.22.2`, `huggingface_hub 1.33.0`, `speechbrain 1.1.2`);
- downloads the release, the Mimi codec and the ECAPA speaker encoder into
  `ComfyUI/models/maya3/hf` (~5.6 GB).

Both steps are resumable. Logs appear in the ComfyUI console.

---

## Why an isolated worker?

ComfyUI currently ships `transformers` 4.46.x. The Maya3 engine imports
`transformers.cache_utils.DynamicLayer`, which only exists in transformers 5.x, and needs
`tokenizers` 0.22 / `huggingface_hub` 1.x. Python can only load one version of a package per
process, so the engine runs in a subprocess with a **shadow site-packages** directory prepended to
`sys.path`. Torch and CUDA are reused from ComfyUI; only the conflicting packages are shadowed.

Nothing in `maya3_comfy/` imports `transformers`, `speechbrain` or `maya3` in ComfyUI's process.

---

## Nodes

### Maya2 Native Model Loader

| input | default | notes |
|---|---|---|
| `repo_id` | `maya-research/maya2-native` | Hugging Face release repo |
| `revision` | *(empty)* | branch/tag/commit; empty = main |
| `performance` | `auto` | `auto` = `safe` on Windows, `fast` elsewhere |
| `max_seats` | `1` | concurrent synthesis seats; keep 1 on 8 GB |
| `capacity` | `1024` | KV tokens per seat; lower it if loading OOMs |
| `depth_dtype` | `fp16` | `bf16`/`fp32` also possible; fp32 is much heavier |
| `auto_download` | `true` | download from Hugging Face on first use |
| `free_vram_first` | `true` | unload ComfyUI models before the worker loads |
| `idle_unload_s` | `600` | unload after this idle time; `0` keeps it loaded |
| `force_reload` | `false` | force a fresh worker/model load |

Outputs: `engine` (connect to the other nodes), `status` (human-readable load report),
`voices` (one voice per line).

**Presets**

- `fast`: CUDA graphs + Triton kernels (`attention=triton`, fused depth). Best on Linux; may work on
  Windows with `triton-windows`.
- `safe`: no backbone graphs, gather attention, no fused Triton kernels. The depth sampler falls back to
  a pure-PyTorch exponential sampler, so **no Triton is required at all**. Portable path, slower per
  frame but much lower risk on Windows / small GPUs.
- `auto`: tries `fast` on Linux and `safe` on Windows. If a load fails, the worker automatically
  retries `safe` once (capacity capped to 1024) before reporting an error.

The Triton fallback is controlled by `MAYA3_COMFY_TORCH_SAMPLING` (`1` = force it on any platform,
`0` = always keep Triton; unset = on for Windows, Triton elsewhere).

### Maya2 Native Text to Speech

Inputs: `engine`, `text`, `language` (dropdown shows the language names, e.g. `Hindi (hi)`; all
12: as bn en gu hi kn ml mr or pa ta te), `voice` (`plain` +
20 named voices), `normalize_text`, `max_seconds` (0 = engine default), `seed` (0 = random),
`temperature`, `top_k`, `top_p`, `repetition_penalty`.

Outputs: `audio` (24 kHz mono AUDIO), `duration_s`, `seed_used`, `info` (reason, frames, TTFA,
wall time, seed).

- Pass only your text. Do **not** prepend the voice name; the engine builds the named-voice prompt.
- Long text is split internally at sentence boundaries (~20 s pieces); the engine gap is 0.15 s.
- Inline emotion tags such as `[laughs]` / `[whispers]` are best-effort.
- `normalize_text` expands numbers, currency, dates, URLs and ALL-CAPS names.

### Maya2 Native Voice Clone

Inputs: `engine`, `text`, `language`, `ref_audio` (AUDIO), `ref_text`, `ref_language`, `mode`,
plus the same sampling controls.

- `icl_spk` (default): reference prompt + ECAPA speaker embedding. Best similarity; needs an exact
  transcript.
- `icl`: reference prompt only; needs the transcript.
- `spk`: speaker embedding only; no transcript needed.
- 3-10 s of clean speech works best; clips longer than 30 s are rejected.
- The reference language may differ from the target language.
- The cloned voice is cached by the worker for the session, so reusing the same reference clip and
  transcript is cheap.
- Only clone voices you have consent to use.

### Maya2 Native Transcribe (faster-whisper)

Transcribes any ComfyUI `AUDIO` input with **faster-whisper only** (CTranslate2). There is no
openai-whisper fallback and no Triton dependency, so it works on the Windows portable/embedded
ComfyUI build. Requires the root `requirements.txt`.

| input | default | notes |
|---|---|---|
| `audio` | — | any ComfyUI AUDIO value (stereo is downmixed, resampled to 16 kHz) |
| `whisper_model` | `base` | `tiny` / `base` / `small` / `medium` / `large-v3`; multilingual checkpoints |
| `language` | `Auto-detect` | dropdown shows the names (`Auto-detect` / `English` / `Hindi`) |
| `task` | `transcribe` | `translate` turns any language into English text |
| `device` | `auto` | `auto` = CUDA when available |
| `word_timestamps` | `true` | per-word start/end/probability in `segments_json` |
| `beam_size` | `5` | 1-10 |
| `temperature` | `0.0` | 0 = greedy/deterministic |
| `initial_prompt` | *(empty)* | optional hint (names, spellings, punctuation) |
| `model_path` | *(empty)* | local faster-whisper folder containing `model.bin` |
| `compute_type` | `auto` | `auto` = float16 on CUDA, int8 on CPU |
| `cpu_threads` | `0` | 0 = CTranslate2 default |
| `keep_model_loaded` | `true` | keep the model in memory between runs (off = unload after every run) |

Outputs: `text`, `segments_json` (segments with optional word timestamps), `srt`,
`timestamped_text` (inline `[MM:SS.mmm - MM:SS.mmm]`), `detected_language`, and the input
`audio` passed through unchanged (so it can feed a clone node directly).

- Checkpoints are downloaded to `ComfyUI/models/maya3/whisper` on first use
  (`Systran/faster-whisper-<model>`).
- With `keep_model_loaded=true` (default) the model stays in memory for the next run.
  Turn it off to **unload it after every run** (`gc.collect()` + `torch.cuda.empty_cache()`),
  so it does not hold VRAM while Maya2 is loaded; expect a short reload on the next
  transcription.
- Silero VAD is enabled (`min_silence_duration_ms=500`) and
  `condition_on_previous_text=False`, which keeps long recordings stable.
- On a machine where CTranslate2 cannot use CUDA, set `device=cpu` and keep `compute_type=auto`
  (int8). CPU needs no GPU memory; with `tiny`/`base`/`small` it can even be faster on short clips,
  while CUDA wins for larger models and long recordings.

### Maya2 Native Unload

Takes `engine` and immediately stops the worker, freeing VRAM. The next generation with the same
engine handle reloads it automatically.

---

## VRAM tips (8 GB)

1. Use the defaults: `performance=auto` (safe on Windows), `max_seats=1`, `capacity=1024`.
2. Keep `free_vram_first=true`; ComfyUI will unload its own models before Maya2 loads.
3. If the loader still OOMs, try `capacity=768` or `512`, and `depth_dtype=fp16`.
4. `fast` uses more VRAM (CUDA graphs); do not use it on 8 GB unless `safe` is too slow.
5. Idle unloading (`idle_unload_s=600`) avoids holding 6+ GB between prompts. The trade-off is a
   ~25 s reload on the next prompt (measured on the test machine); set it to `0` while iterating.

### Measured on Windows 11 + RTX 3060 Ti 8 GB

`safe`, `max_seats=1`, `capacity=1024`, `depth_dtype=fp16`, torch 2.9.1+cu130:

| step | result |
|---|---|
| engine load + 20 named voices | 23.0 s (warmup 3.2 s) |
| VRAM after load | 5.93 GB allocated / 6.35 GB reserved |
| English, 8.00 s audio | 9.9 s (1.24× real time), first audio 0.19 s |
| Hindi, 3.28 s audio | 3.5 s (1.06× real time), first audio 0.09 s |
| Transcribe `base`, CUDA float16, 8 s clip | 1.5 s load + 2.4 s (3.3× real time), 17 words |
| Transcribe `base`, CPU int8, 8 s clip | 0.8 s load + 1.5 s (5.4× real time) |

The transcription rows are steady-state: the node releases the model after every run, so each run
pays that reload cost. The very first run additionally downloads the checkpoint (~145 MB for `base`).

Generation is slightly slower than real time on this card. `fast` on a Linux box with a working
Triton is the upstream-optimised path; 12 GB+ cards have comfortable headroom.

---

## Files and environment variables

```
ComfyUI/models/maya3/
  hf/                          Hugging Face cache (release + Mimi + ECAPA)
  runtime/<tag>/site/          shadow site-packages for the worker
  tmp/                         short-lived audio hand-off files
  whisper/                     faster-whisper checkpoints (Transcribe node)
```

| variable | meaning |
|---|---|
| `MAYA3_COMFY_HOME` | override the `models/maya3` base directory |
| `MAYA3_COMFY_MODELS_DIR` | ComfyUI `models` directory if `folder_paths` is unavailable |
| `MAYA3_COMFY_SITE` | use an existing shadow site instead of installing |
| `MAYA3_COMFY_PYTHON` | use a different Python (e.g. a venv that already has the deps) |
| `MAYA3_COMFY_SKIP_INSTALL` | `1` = never run pip, fail with the missing package list |
| `MAYA3_COMFY_TORCH_SAMPLING` | `1` = use the PyTorch depth sampler (no Triton); `0` = force Triton |
| `HF_HUB_OFFLINE` | `1` = use only the local cache |

If you set `MAYA3_COMFY_PYTHON`, that interpreter must already have `transformers 5.6.2`,
`tokenizers 0.22.2`, `huggingface_hub >= 1.5`, `speechbrain 1.1.2`, CUDA torch and torchaudio.
The worker inserts the repository root into `sys.path` itself, so the upstream project does not
need to be pip-installed.

To pre-populate the cache manually:

```bat
set HF_HUB_CACHE=C:\...\ComfyUI\models\maya3\hf
python -m huggingface_hub.commands.huggingface_cli download maya-research/maya2-native
python -m huggingface_hub.commands.huggingface_cli download kyutai/mimi
python -m huggingface_hub.commands.huggingface_cli download speechbrain/spkrec-ecapa-voxceleb
```

Then set the loader's `auto_download=false` (or `HF_HUB_OFFLINE=1`) to forbid network access.

---

## Troubleshooting

**"Maya2 needs torch >= 2.7.1"** — update ComfyUI's torch, or point `MAYA3_COMFY_PYTHON` at a
venv with a compatible CUDA torch + torchaudio.

**"Maya2 needs torchaudio"** — install the torchaudio build matching ComfyUI's torch.

**pip install fails / no network** — the runtime can be prepared on another machine: set
`MAYA3_COMFY_SITE` to a directory containing the four pinned packages, or set
`MAYA3_COMFY_PYTHON` to a prepared venv. `MAYA3_COMFY_SKIP_INSTALL=1` gives a precise missing list.

**Worker OOM at load** — lower `capacity`, keep `performance=safe`, close other GPU apps, or raise
`free_vram_first`. The worker retries the safe preset automatically once.

**Triton / `Python.h` errors** — the ComfyUI embeddable Python has no C headers, so Triton cannot
compile its CUDA helper on Windows. The safe preset avoids Triton entirely (the depth sampler uses a
pure-PyTorch fallback). If you see a Triton traceback in `fast` mode, switch to `safe`, or install a
working `triton-windows` build. Set `MAYA3_COMFY_TORCH_SAMPLING=1` to force the fallback elsewhere.

**"faster-whisper is not installed"** — install the optional transcription dependency with ComfyUI's
own Python, then restart ComfyUI:

```bat
python_embeded\python.exe -m pip install -r custom_nodes\ComfyUI-maya2-native\requirements.txt
```

Do not install `openai-whisper` instead; this node only uses faster-whisper.

**Transcription is slow / first run takes long** — the first run downloads the faster-whisper
checkpoint into `ComfyUI/models/maya3/whisper`; later runs reuse it. The node releases the model after
every run to free VRAM, so each run also reloads it (~1.5 s for `base` on CUDA, ~0.8 s on CPU). For
short clips with `tiny`/`base`/`small`, CPU int8 is often faster overall; use CUDA for `medium` /
`large-v3` or long recordings. Keep `word_timestamps=false` for the quickest turnaround.

**Worker crashes** — the ComfyUI console shows the worker's stderr, including the Python traceback.
The next queue run starts a fresh worker.

**Cancel** — cancelling a ComfyUI prompt kills the worker subprocess (the engine is inside a
blocking request). Partial downloads are resumed; the next run reloads the model.

**Audio is slow / real-time factor > 1** — check that the worker log says `cuda=True`. On 8 GB with
`safe`, generation is slower than on the upstream H100 numbers; try `performance=fast` if your
driver/triton combination is stable.

---

## Uninstall

Delete this custom node directory, then delete `ComfyUI/models/maya3` to reclaim ~6 GB.
ComfyUI's own Python environment was never modified.

---

## Publishing to the Comfy Registry

The node is published to the [Comfy Registry](https://registry.comfy.org) as `maya2-native`:

```bash
comfy node publish          # or: push a version bump in pyproject.toml to main
```

- `pyproject.toml` holds the registry metadata (`[tool.comfy]`: `PublisherId`, `DisplayName`).
  Its `dependencies` come from `requirements.txt` — the ComfyUI-safe list. The engine pins
  (torch, transformers 5.x, speechbrain) are deliberately **not** installed into ComfyUI's
  Python; the worker builds its own shadow runtime under `ComfyUI/models/maya3/`.
- `.comfyignore` keeps development-only files (tests, benchmarks, tools, docs) out of the
  published archive. `.github/workflows/publish_action.yml` publishes automatically when
  `pyproject.toml` changes on `main`; it needs the `REGISTRY_ACCESS_TOKEN` repository secret
  (an API key created at registry.comfy.org).
- Before the first publish, check that `PublisherId` in `pyproject.toml` matches the publisher
  account on the registry website, and that the node id `maya2-native` is still free.

---

## Credits and licences

Maya2 Native / Maya3 is Apache-2.0; the weights have their own licence on the model card.
The engine builds on Qwen3 (Alibaba), Mimi (Kyutai) and SpeechBrain ECAPA-TDNN. The ComfyUI
integration lives in `maya3_comfy/` and only adds an `EngineConfig.attention` switch to
`maya3/engine.py` so the portable `gather` path can be selected without Triton.
