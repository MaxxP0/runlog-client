# runlog-managed — single-file wandb replacement. `python wandb.py --help`
"""wandb.py — a drop-in replacement for Weights & Biases.

Put this file next to a training script and `import wandb` resolves to it
instead of the installed package. Nothing else is installed: this is one
stdlib-only file, so copying it to a GPU box is the entire deployment.

    python wandb.py configure                                   # ask for the address and keys
    python wandb.py --link ~/code/trainer                       # copy into a repo
    python wandb.py --smoke                                     # verify the pipeline
    python wandb.py --status                                    # show config

Every call an ordinary trainer makes is accepted with the
same signature — login, init, config.update, log, watch, finish, and the
Image / Video / Audio / Html / Table wrappers — so no training file changes.

Metrics never pass through the API server: init() asks it once for presigned
PUT URLs and everything after that goes straight to B2. A training loop cannot
slow down or take down the website, and there is no local agent to hang at exit.

Credentials come from ~/.runlog.json (mode 600) or RUNLOG_URL / RUNLOG_API_KEY.
Deliberately never from this file, which travels with the code.
"""
from __future__ import annotations

import argparse
import atexit
import base64
import concurrent.futures
import html as _html
import io
import json
import math
import os
import platform
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import http.client
import ssl
import urllib.error
import urllib.parse
import urllib.request
import wave
from typing import Any

__all__ = [
    "init", "log", "login", "watch", "finish", "config", "summary",
    "Image", "Video", "Audio", "Html", "Table", "Run", "run",
    "list_projects", "list_runs", "get_run", "history",
]




# MARK: - Machine and environment capture

# Probes are external commands on a background thread. Every one is bounded:
# an `nvidia-smi` wedged behind a hung driver must not stall the flush loop.
PROBE_TIMEOUT = 5.0

# A driver hang (Xid error, GPU off the bus) leaves nvidia-smi in
# uninterruptible sleep, where subprocess's timeout kill does nothing and the
# wait() blocks this thread. That thread also sends heartbeats, so a wedged
# probe would make a perfectly healthy run show as "stale". After two timeouts
# the process stops shelling out for the rest of the run.
_TIMEOUTS = {"count": 0}

_MAX_TIMEOUTS = 2

def _run(command: list[str], timeout: float = PROBE_TIMEOUT) -> str:
    if _TIMEOUTS["count"] >= _MAX_TIMEOUTS:
        return ""
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return done.stdout.strip() if done.returncode == 0 else ""
    except subprocess.TimeoutExpired:
        _TIMEOUTS["count"] += 1
        if _TIMEOUTS["count"] >= _MAX_TIMEOUTS:
            print(f"runlog: {command[0]} timed out twice; disabling external probes",
                  file=sys.__stderr__)
        return ""
    except Exception:
        return ""

def _sysctl(key: str) -> str:
    return _run(["sysctl", "-n", key]) if sys.platform == "darwin" else ""

def _cpu_brand() -> str:
    if sys.platform == "darwin":
        return _sysctl("machdep.cpu.brand_string")
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()

def _ram_bytes() -> int:
    if sys.platform == "darwin":
        value = _sysctl("hw.memsize")
        return int(value) if value.isdigit() else 0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0

def _physical_cores() -> int:
    if sys.platform == "darwin":
        value = _sysctl("hw.physicalcpu")
        return int(value) if value.isdigit() else 0
    try:
        ids = set()
        core = package = None
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("core id"):
                    core = line.split(":")[1].strip()
                elif line.startswith("physical id"):
                    package = line.split(":")[1].strip()
                elif not line.strip() and core is not None:
                    ids.add((package, core))
                    core = package = None
        return len(ids)
    except OSError:
        return 0

def _git_info(path: str) -> dict:
    """Commit the run was launched from. Worth more than any hyperparameter
    when you come back to a six-week-old run and ask what the code did."""
    if not shutil.which("git"):
        return {}
    inside = _run(["git", "-C", path, "rev-parse", "--is-inside-work-tree"])
    if inside != "true":
        return {}
    info = {
        "commit": _run(["git", "-C", path, "rev-parse", "HEAD"])[:12],
        "branch": _run(["git", "-C", path, "rev-parse", "--abbrev-ref", "HEAD"]),
    }
    status = _run(["git", "-C", path, "status", "--porcelain"])
    info["dirty"] = bool(status)
    if status:
        info["dirty_files"] = len(status.splitlines())
    return {k: v for k, v in info.items() if v != ""}

def _torch_info() -> dict:
    """Never imports torch — only reports it if the trainer already has."""
    torch = sys.modules.get("torch")
    if torch is None:
        return {}
    info = {"torch": getattr(torch, "__version__", "")}
    try:
        version = getattr(torch, "version", None)
        if getattr(version, "cuda", None):
            info["cuda"] = version.cuda
        backends = getattr(torch, "backends", None)
        cudnn = getattr(backends, "cudnn", None)
        if cudnn is not None and callable(getattr(cudnn, "version", None)):
            cudnn_version = cudnn.version()
            if cudnn_version:
                info["cudnn"] = str(cudnn_version)
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            gpus = []
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                gpus.append({
                    "name": props.name,
                    "memory_gb": round(props.total_memory / 1e9, 1),
                    "capability": f"{props.major}.{props.minor}",
                    "multi_processors": getattr(props, "multi_processor_count", None),
                })
            info["gpus"] = gpus
        elif getattr(getattr(backends, "mps", None), "is_available", lambda: False)():
            info["accelerator"] = "mps"
    except Exception:
        pass
    return info

def physical_index(logical: int) -> int:
    """Map a torch device index to the physical GPU nvidia-smi reports.

    Under CUDA_VISIBLE_DEVICES=2,3 the process's cuda:0 IS physical GPU 2, so
    anything read from nvidia-smi has to be looked up by the physical index or
    it describes the wrong card.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return logical
    parts = [p.strip() for p in visible.split(",") if p.strip()]
    try:
        return int(parts[logical])
    except (IndexError, ValueError):
        return logical

# Power limits are fixed per card, so they are read once and reused rather than
# re-forked on every sample.
_POWER_LIMITS: dict[int, float] = {}

def _read_power_limits() -> dict[int, float]:
    if not shutil.which("nvidia-smi"):
        return {}
    output = _run(["nvidia-smi", "--query-gpu=index,power.limit",
                   "--format=csv,noheader,nounits"])
    limits: dict[int, float] = {}
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            limits[int(parts[0])] = float(parts[1])
        except ValueError:
            continue   # "[N/A]" on cards that do not report a limit
    return limits

def _nvidia_driver() -> str:
    if not shutil.which("nvidia-smi"):
        return ""
    lines = _run(["nvidia-smi", "--query-gpu=driver_version",
                  "--format=csv,noheader"]).splitlines()
    return lines[0].strip() if lines else ""

def collect_system_info() -> dict:
    """Static description of the machine and code this run started on."""
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "cpu": _cpu_brand(),
        "cpu_cores_physical": _physical_cores() or None,
        "cpu_cores_logical": os.cpu_count(),
        "ram_gb": round(_ram_bytes() / 1e9, 1) or None,
        "cwd": os.getcwd(),
        "command": " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:])[:400],
    }
    info.update(_torch_info())
    driver = _nvidia_driver()
    if driver:
        info["nvidia_driver"] = driver
    global _POWER_LIMITS
    _POWER_LIMITS = _read_power_limits()
    if _POWER_LIMITS:
        info["gpu_power_limits_w"] = ", ".join(
            f"gpu{i}: {int(w)}W" for i, w in sorted(_POWER_LIMITS.items()))
    handles = _nvml_handles()
    if handles:
        info["gpus_on_host"] = len(handles)
    owned = visible_physical()
    if owned:
        # Telemetry covers every card on the box; this says which ones the run
        # is entitled to, so eight charted GPUs are not read as eight in use.
        info["gpus_used_by_this_process"] = ", ".join(f"gpu{i}" for i in owned)
    # Multi-node launchers set these; without them a run from one rank of a
    # distributed job is indistinguishable from a single-GPU run.
    for key in ("SLURM_JOB_ID", "WORLD_SIZE", "RANK", "LOCAL_RANK", "CUDA_VISIBLE_DEVICES"):
        if os.environ.get(key):
            info[key.lower()] = os.environ[key]
    git = _git_info(os.getcwd())
    if git:
        info["git"] = git
    return {k: v for k, v in info.items() if v not in (None, "", {}, [])}

_NVIDIA_QUERY = (
    "index,utilization.gpu,utilization.memory,memory.used,memory.total,"
    "temperature.gpu,power.draw"
)

def _nvidia_sample() -> dict:
    if not shutil.which("nvidia-smi"):
        return {}
    output = _run(["nvidia-smi", f"--query-gpu={_NVIDIA_QUERY}",
                   "--format=csv,noheader,nounits"])
    if not output:
        return {}
    metrics: dict[str, float] = {}
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        index = parts[0]
        for name, raw in (
            (f"system/gpu{index}_util_pct", parts[1]),
            (f"system/gpu{index}_mem_util_pct", parts[2]),
            (f"system/gpu{index}_mem_used_mb", parts[3]),
            (f"system/gpu{index}_temp_c", parts[5]),
            (f"system/gpu{index}_power_w", parts[6]),
            (f"system/gpu{index}_mem_total_mb", parts[4]),
        ):
            try:
                metrics[name] = float(raw)
            except ValueError:
                continue  # "[N/A]" on cards that do not report a field
        used = metrics.get(f"system/gpu{index}_mem_used_mb")
        total = metrics.pop(f"system/gpu{index}_mem_total_mb", None)
        if used is not None and total:
            metrics[f"system/gpu{index}_mem_pct"] = 100.0 * used / total
        limit = _POWER_LIMITS.get(int(index)) if index.isdigit() else None
        drawn = metrics.get(f"system/gpu{index}_power_w")
        if limit and drawn is not None:
            metrics[f"system/gpu{index}_power_pct"] = 100.0 * drawn / limit
    return metrics

# NVML is the right interface for multi-GPU: it reads driver state, so it can
# report EVERY card on the box without creating a CUDA context on any of them.
# That matters under DDP, where each rank has CUDA_VISIBLE_DEVICES pinned to one
# GPU — only rank 0 usually logs, and without NVML that run would chart a single
# card out of eight.
_NVML = {"tried": False, "handles": [], "module": None}

LOG_ALL_GPUS = os.environ.get("RUNLOG_GPU_ALL", "1") != "0"

def _nvml_handles():
    if _NVML["tried"]:
        return _NVML["handles"]
    _NVML["tried"] = True
    try:
        import pynvml            # ships with torch as nvidia-ml-py
    except ImportError:
        return []
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        _NVML["module"] = pynvml
        _NVML["handles"] = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
    except Exception:
        _NVML["handles"] = []
    return _NVML["handles"]

def visible_physical() -> list[int]:
    """Physical indices this process is allowed to use."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return []
    out = []
    for part in visible.split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return out

def _nvml_sample() -> dict:
    handles = _nvml_handles()
    if not handles:
        return {}
    pynvml = _NVML["module"]
    allowed = visible_physical()
    metrics: dict[str, float] = {}
    for index, handle in enumerate(handles):
        if not LOG_ALL_GPUS and allowed and index not in allowed:
            continue
        prefix = f"system/gpu{index}_"
        try:
            rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            metrics[prefix + "util_pct"] = float(rates.gpu)
            metrics[prefix + "mem_util_pct"] = float(rates.memory)
        except Exception:
            pass
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            metrics[prefix + "mem_used_gb"] = mem.used / 1e9
            if mem.total:
                metrics[prefix + "mem_pct"] = 100.0 * mem.used / mem.total
        except Exception:
            pass
        try:
            metrics[prefix + "temp_c"] = float(
                pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
        except Exception:
            pass
        try:
            # NVML reports power in milliwatts.
            drawn = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            metrics[prefix + "power_w"] = drawn
            limit = _POWER_LIMITS.get(index)
            if limit is None:
                try:
                    limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
                    _POWER_LIMITS[index] = limit
                except Exception:
                    limit = None
            if limit:
                metrics[prefix + "power_pct"] = 100.0 * drawn / limit
        except Exception:
            pass
        try:
            metrics[prefix + "clock_mhz"] = float(
                pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
        except Exception:
            pass
    return metrics

def _torch_gpu_sample() -> dict:
    """Allocator figures for the device this process is actually training on.

    Deliberately NOT looped over every index: `memory_allocated` and
    `mem_get_info` go through the CUDA runtime and would create a context on
    each idle GPU — hundreds of MB of VRAM per card, multiplied by every rank.
    NVML above already covers the other cards.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return {}
    metrics: dict[str, float] = {}
    try:
        if not torch.cuda.is_available():
            if hasattr(torch, "mps") and callable(getattr(torch.mps, "current_allocated_memory", None)):
                metrics["system/mps_alloc_gb"] = torch.mps.current_allocated_memory() / 1e9
            return metrics
        logical = torch.cuda.current_device()
        index = physical_index(logical)
        metrics[f"system/gpu{index}_torch_alloc_gb"] = torch.cuda.memory_allocated(logical) / 1e9
        metrics[f"system/gpu{index}_torch_reserved_gb"] = torch.cuda.memory_reserved(logical) / 1e9
    except Exception:
        pass
    return metrics

def sample() -> dict:
    """Point-in-time machine metrics, as plain scalars ready for log()."""
    metrics = _nvml_sample()
    # nvidia-smi is the fallback when pynvml is absent. It also enumerates every
    # card, and it ignores CUDA_VISIBLE_DEVICES, so it reports physical indices
    # exactly like NVML does.
    if not metrics:
        metrics = _nvidia_sample()
    for key, value in _torch_gpu_sample().items():
        metrics.setdefault(key, value)

    # psutil is present in the training env but is NOT a dependency; the tracker
    # stays importable on a bare interpreter.
    try:
        import psutil

        metrics["system/cpu_pct"] = psutil.cpu_percent(interval=None)
        virtual = psutil.virtual_memory()
        metrics["system/ram_used_gb"] = virtual.used / 1e9
        metrics["system/ram_pct"] = virtual.percent
        process = psutil.Process()
        metrics["system/proc_rss_gb"] = process.memory_info().rss / 1e9
    except Exception:
        pass

    return metrics



# MARK: - Parameter and gradient histograms

BINS = int(os.environ.get("RUNLOG_HIST_BINS", "64"))

# Wall-clock floor between snapshots, independent of log_freq.
MIN_INTERVAL = float(os.environ.get("RUNLOG_HIST_INTERVAL", "60"))

# Above this many named tensors the snapshot is subsampled. High enough to
# cover a large transformer (several hundred named tensors) without truncation.
MAX_TENSORS = int(os.environ.get("RUNLOG_HIST_MAX_TENSORS", "1200"))

# Gradient magnitudes are heavy-tailed: linear min..max bins put essentially
# all the mass in one bin, which is a useless panel. Fixed log2|g| bins spread
# it out AND keep the y axis identical across snapshots, which is the only
# reason a step x bin heatmap is readable at all.
GRAD_LOG2_MIN = float(os.environ.get("RUNLOG_HIST_LOG2_MIN", "-40"))

GRAD_LOG2_MAX = float(os.environ.get("RUNLOG_HIST_LOG2_MAX", "8"))

def clean_name(name: str) -> str:
    """Strip wrapper prefixes so a tensor keeps one identity across runs.

    DDP adds `module.` and torch.compile adds `_orig_mod.` (DDP followed by
    compile gives `_orig_mod.module.…`). Without this, the same tensor
    from a single-GPU rerun gets a different key and the two runs cannot be
    compared.
    """
    for _ in range(4):
        for prefix in ("_orig_mod.", "module."):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        else:
            break
    return name

def histogram(tensor, bins: int = BINS, scale: str = "linear") -> dict | None:
    """One tensor -> {min, max, counts, ...}. None if unusable.

    `scale="log2"` histograms log2 of the magnitude over a FIXED window rather
    than the tensor's own min..max — see GRAD_LOG2_MIN above. Non-finite values
    are stripped rather than fatal: `torch.histc` raises on a NaN range, so a
    single inf would otherwise drop the tensor from the snapshot entirely,
    precisely when you most want to see it.
    """
    try:
        import torch
    except ImportError:
        return None
    try:
        flat = tensor.detach().reshape(-1)
        if flat.numel() == 0:
            return None
        if flat.dtype not in (torch.float32, torch.float64):
            flat = flat.float()

        total = int(flat.numel())
        finite_mask = torch.isfinite(flat)
        nonfinite = total - int(finite_mask.sum())
        if nonfinite:
            flat = flat[finite_mask]
            if flat.numel() == 0:
                return {"min": 0.0, "max": 0.0, "counts": [0] * bins,
                        "n": total, "nonfinite": nonfinite, "scale": scale}

        if scale == "log2":
            magnitude = flat.abs()
            positive = magnitude > 0
            zeros = int(magnitude.numel() - int(positive.sum()))
            exponents = torch.log2(magnitude[positive])
            # histc silently DROPS values outside [min,max], so anything
            # outside the window has to be counted here or a run whose
            # gradients left the window would look empty rather than broken.
            under = int((exponents < GRAD_LOG2_MIN).sum())
            over = int((exponents > GRAD_LOG2_MAX).sum())
            counts = torch.histc(exponents, bins=bins,
                                 min=GRAD_LOG2_MIN, max=GRAD_LOG2_MAX)
            entry = {
                "min": GRAD_LOG2_MIN, "max": GRAD_LOG2_MAX,
                "counts": [int(c) for c in counts.to("cpu").tolist()],
                "n": total, "scale": "log2", "zeros": zeros,
            }
            if under:
                entry["under"] = under
            if over:
                entry["over"] = over
        else:
            low = float(flat.min())
            high = float(flat.max())
            if high == low:
                # Constant tensor (zero-init, or dead). Worth showing as a flat
                # line rather than dropping.
                high = low + 1e-8
            counts = torch.histc(flat, bins=bins, min=low, max=high)
            entry = {
                "min": low, "max": high,
                "counts": [int(c) for c in counts.to("cpu").tolist()],
                "n": total, "scale": "linear",
            }
        if nonfinite:
            entry["nonfinite"] = nonfinite
        return entry
    except Exception:
        return None

def _select(named: list) -> list:
    """Bound the tensor count without truncating by module order.

    `named[:MAX]` would keep the embedding and the first blocks and drop the
    decoder and output head — exactly where exploding gradients show up. An
    even stride keeps coverage across the whole model.
    """
    if len(named) <= MAX_TENSORS:
        return named
    stride = len(named) / MAX_TENSORS
    return [named[min(len(named) - 1, int(i * stride))] for i in range(MAX_TENSORS)]

class Watcher:
    """Holds the backward hooks for one watched model.

    Hooks are always installed but do nothing unless armed, so the per-step
    cost when no snapshot is due is one boolean test per tensor per backward.
    """

    def __init__(self, model, mode: str = "all", bins: int = BINS) -> None:
        self.model = model
        self.mode = mode or "all"
        self.bins = bins
        self.truncated = False
        self._grad_hists: dict[str, dict] = {}
        self._grad_sq = 0.0
        self._arm_hist = False
        self._arm_norm = False
        self._handles: list = []
        if self.mode in ("all", "gradients"):
            self._install()

    def _install(self) -> None:
        try:
            named = list(self.model.named_parameters())
        except Exception:
            return
        selected = _select(named)
        self.truncated = len(selected) != len(named)
        for name, param in selected:
            if not getattr(param, "requires_grad", False):
                continue
            try:
                self._handles.append(param.register_hook(self._hook(clean_name(name))))
            except Exception:
                continue

    def _hook(self, name: str):
        def capture(grad):
            # Must never raise and must never return a tensor: returning a
            # value here would REPLACE the gradient flowing backwards.
            try:
                if self._arm_norm:
                    self._grad_sq += float(grad.detach().float().pow(2).sum())
                if self._arm_hist:
                    entry = histogram(grad, self.bins, scale="log2")
                    if entry:
                        self._grad_hists[name] = entry
            except Exception:
                pass
            return None
        return capture

    def arm(self, want_hist: bool) -> None:
        """Ask the next backward pass to record."""
        self._arm_norm = True
        self._arm_hist = want_hist or self._arm_hist
        if want_hist:
            self._grad_hists = {}
        self._grad_sq = 0.0

    @property
    def has_gradients(self) -> bool:
        return bool(self._grad_hists)

    def grad_norm(self) -> float | None:
        if not self._arm_norm:
            return None
        value = self._grad_sq ** 0.5
        self._arm_norm = False
        return value

    def param_norm(self) -> float | None:
        try:
            total = 0.0
            for _, param in self.model.named_parameters():
                total += float(param.detach().float().pow(2).sum())
            return total ** 0.5
        except Exception:
            return None

    def collect(self) -> dict:
        """Build the snapshot: parameters read now, gradients from the hooks."""
        out: dict[str, dict] = {}
        if self.mode in ("all", "parameters"):
            try:
                for name, param in _select(list(self.model.named_parameters())):
                    entry = histogram(param, self.bins)
                    if entry:
                        out.setdefault(clean_name(name), {})["p"] = entry
            except Exception:
                pass
        for name, entry in self._grad_hists.items():
            out.setdefault(name, {})["g"] = entry
        self._grad_hists = {}
        self._arm_hist = False
        return out

    def close(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles = []

def snapshot(model, log: str = "all", bins: int = BINS) -> dict:
    """One-shot snapshot without hooks.

    Only sees gradients if `.grad` still holds them, which after the usual
    `optimizer.zero_grad()` before logging it does not. Kept for callers that manage their own timing; `Watcher` is what
    the tracker uses.
    """
    watcher = Watcher.__new__(Watcher)
    watcher.model, watcher.mode, watcher.bins = model, log or "all", bins
    watcher._grad_hists, watcher._arm_hist = {}, False
    if log in ("all", "gradients"):
        try:
            for name, param in _select(list(model.named_parameters())):
                if getattr(param, "grad", None) is not None:
                    entry = histogram(param.grad, bins, scale="log2")
                    if entry:
                        watcher._grad_hists[clean_name(name)] = entry
        except Exception:
            pass
    return watcher.collect()



# MARK: - Media encoders and the wandb-shaped wrappers

def _as_numpy(value):
    """Torch tensor / numpy array / anything array-like -> numpy, on CPU."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy") and not hasattr(value, "dtype"):
        value = value.numpy()
    try:
        import numpy as np
    except ImportError as exc:
        raise TypeError("numpy is required to log arrays") from exc
    return np.asarray(value)

def _to_uint8_hwc(array):
    """Normalize the many shapes training code produces into HWC uint8.

    Accepts CHW or HWC, float in [0,1] or [0,255], and single-channel; these
    all show up across the two repos and guessing wrong produces a black or
    scrambled panel rather than an error, so the rules are explicit.
    """
    import numpy as np

    array = _as_numpy(array)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = array.transpose(1, 2, 0)  # CHW -> HWC
    if array.dtype != np.uint8:
        finite = array[np.isfinite(array)]
        peak = float(finite.max()) if finite.size else 1.0
        array = np.clip(array * (255.0 if peak <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return array

def encode_image(value) -> tuple[bytes, str]:
    """PIL image, array, encoded bytes or a path -> (bytes, extension)."""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value), "png"
    if isinstance(value, str) or hasattr(value, "__fspath__"):
        path = os.fspath(value)
        with open(path, "rb") as handle:
            return handle.read(), (path.rsplit(".", 1)[-1].lower() or "png")
    if hasattr(value, "save") and not hasattr(value, "shape"):  # PIL.Image
        buffer = io.BytesIO()
        value.save(buffer, format="PNG")
        return buffer.getvalue(), "png"
    try:
        from PIL import Image as PILImage
    except ImportError as exc:
        raise TypeError("Pillow is required to log arrays as images") from exc
    buffer = io.BytesIO()
    # compress_level=1: measured ~2x faster than the default 6 for ~5% more
    # bytes. These are sample dumps, not archives.
    PILImage.fromarray(_to_uint8_hwc(value)).save(
        buffer, format="PNG", compress_level=1)
    return buffer.getvalue(), "png"

def _ffmpeg_binary() -> str | None:
    """imageio-ffmpeg ships a binary; otherwise use whatever is on PATH."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return shutil.which("ffmpeg")

def encode_video(frames, fps: int = 8) -> tuple[bytes, str]:
    """Frames [T,H,W,C] or [T,C,H,W] uint8/float -> (bytes, "mp4" or "gif").

    Prefers h264 through ffmpeg because a recon grid over many steps is large
    and a GIF of the same content is several times the size at worse quality.
    Falls back to an animated GIF when no ffmpeg is reachable, so a machine
    without it still logs something viewable rather than raising mid-training.
    """
    import numpy as np

    array = _as_numpy(frames)
    if array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 4:
        raise TypeError(f"Video needs 4 dims [T,H,W,C] or [T,C,H,W], got {array.shape}")
    if array.shape[1] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = array.transpose(0, 2, 3, 1)  # TCHW -> THWC
    if array.dtype != np.uint8:
        peak = float(array.max()) if array.size else 1.0
        array = np.clip(array * (255.0 if peak <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    array = array[:, :, :, :3]

    # yuv420p requires even dimensions; an odd grid silently fails to encode.
    height, width = array.shape[1], array.shape[2]
    if height % 2 or width % 2:
        array = array[:, : height - (height % 2), : width - (width % 2), :]
        height, width = array.shape[1], array.shape[2]

    binary = _ffmpeg_binary()
    if binary:
        # Output goes to a temp FILE, not a pipe: the mp4 muxer rewrites the
        # moov atom after the last frame, so it needs a seekable target. Piped
        # to stdout ffmpeg fails and every video silently became a GIF.
        with tempfile.TemporaryDirectory() as workdir:
            target = os.path.join(workdir, "clip.mp4")
            command = [
                binary, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", str(max(1, int(fps))),
                "-i", "pipe:0",
                "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                target,
            ]
            try:
                done = subprocess.run(
                    command, input=array.tobytes(),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600,
                )
                if done.returncode == 0 and os.path.getsize(target):
                    with open(target, "rb") as handle:
                        return handle.read(), "mp4"
            except Exception:
                pass  # fall through to GIF

    try:
        from PIL import Image as PILImage
    except ImportError as exc:
        raise RuntimeError(
            "Logging video needs ffmpeg (pip install imageio-ffmpeg) or Pillow"
        ) from exc
    buffer = io.BytesIO()
    images = [PILImage.fromarray(frame) for frame in array]
    images[0].save(
        buffer, format="GIF", save_all=True, append_images=images[1:],
        duration=max(20, int(1000 / max(1, fps))), loop=0, optimize=True,
    )
    return buffer.getvalue(), "gif"

def encode_audio(samples, sample_rate: int = 24000) -> tuple[bytes, str]:
    """Float or int16 waveform -> 16-bit PCM WAV. Stdlib only, no deps."""
    import numpy as np

    array = _as_numpy(samples)
    array = np.squeeze(array)
    if array.ndim > 2:
        raise TypeError(f"Audio needs 1 or 2 dims, got {array.shape}")
    if array.ndim == 2 and array.shape[0] < array.shape[1]:
        array = array.T  # channels-first -> frames-first
    channels = 1 if array.ndim == 1 else array.shape[1]
    if array.dtype != np.int16:
        array = np.clip(np.nan_to_num(array.astype(np.float32)), -1.0, 1.0)
        array = (array * 32767.0).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(array.tobytes())
    return buffer.getvalue(), "wav"

class Media:
    """Base: something `log()` turns into one file plus a bit of metadata."""

    kind = "file"

    def __init__(self, caption: str = "") -> None:
        self.caption = caption or ""

    def encode(self) -> tuple[bytes, str]:
        raise NotImplementedError

class Image(Media):
    kind = "image"

    def __init__(self, data, caption: str = "", **_ignored) -> None:
        super().__init__(caption)
        self._data = data

    def encode(self) -> tuple[bytes, str]:
        return encode_image(self._data)

class Video(Media):
    kind = "video"

    def __init__(self, data, fps: int = 8, format: str = "mp4", caption: str = "", **_ignored) -> None:
        super().__init__(caption)
        self._data = data
        self.fps = fps or 8
        self._format = format

    def encode(self) -> tuple[bytes, str]:
        # A path or already-encoded bytes passes straight through; only arrays
        # go to the encoder.
        if isinstance(self._data, (bytes, bytearray)):
            return bytes(self._data), (self._format or "mp4")
        if isinstance(self._data, str) or hasattr(self._data, "__fspath__"):
            path = os.fspath(self._data)
            with open(path, "rb") as handle:
                return handle.read(), (path.rsplit(".", 1)[-1].lower() or "mp4")
        return encode_video(self._data, self.fps)

class Audio(Media):
    kind = "audio"

    def __init__(self, data, sample_rate: int = 24000, caption: str = "", **_ignored) -> None:
        super().__init__(caption)
        self._data = data
        self.sample_rate = sample_rate

    def encode(self) -> tuple[bytes, str]:
        if isinstance(self._data, (bytes, bytearray)):
            return bytes(self._data), "wav"
        if isinstance(self._data, str) or hasattr(self._data, "__fspath__"):
            path = os.fspath(self._data)
            with open(path, "rb") as handle:
                return handle.read(), (path.rsplit(".", 1)[-1].lower() or "wav")
        return encode_audio(self._data, self.sample_rate)

class Html(Media):
    kind = "html"

    def __init__(self, data, caption: str = "", **_ignored) -> None:
        super().__init__(caption)
        self._data = data if isinstance(data, str) else str(data)

    def encode(self) -> tuple[bytes, str]:
        return self._data.encode("utf-8", "replace"), "html"

class Table(Media):
    """W&B's Table, rendered to an HTML table.

    The dashboard has one text/HTML viewer rather than a bespoke table widget,
    and these tables are read, not queried — caption predictions and optimizer
    outlier dumps — so HTML loses nothing that was being used.
    """

    kind = "html"

    def __init__(self, columns=None, data=None, caption: str = "", **_ignored) -> None:
        super().__init__(caption)
        self.columns = list(columns or [])
        self.data = [list(row) for row in (data or [])]

    def add_data(self, *values) -> None:
        self.data.append(list(values))

    def encode(self) -> tuple[bytes, str]:
        def cell(value):
            return _html.escape(str(value))

        head = "".join(f"<th>{cell(c)}</th>" for c in self.columns)
        body = "".join(
            "<tr>" + "".join(f"<td>{cell(v)}</td>" for v in row) + "</tr>"
            for row in self.data
        )
        markup = (
            "<table border=1 style='border-collapse:collapse;font:12px ui-monospace,monospace'>"
            f"<tr>{head}</tr>{body}</table>"
        )
        return markup.encode("utf-8"), "html"



# MARK: - The run itself

from typing import Any

# Credentials live in ~/.runlog.json (mode 600), not in the training repos:
# a key written into a project ends up in its git history sooner or later.
# Environment variables still win, for CI and one-off overrides.
CONFIG_PATH = os.path.expanduser(os.environ.get("RUNLOG_CONFIG", "~/.runlog.json"))

# ---------------------------------------------------------------------------
# Baked-in address and credentials, filled by `python wandb.py --embed-key`.
# Both are blank in the published client: `configure` asks for them.
#
# This is a WRITE credential. Anyone holding it can create runs and overwrite
# existing ones inside the tracker's own B2 prefix. It cannot read the
# dashboard (that needs the password), cannot reach anything else in the bucket, and cannot
# delete anything. Bounded, but not nothing — keep this file out of git.
#
# Order of precedence: RUNLOG_* environment variables, then ~/.runlog.json,
# then these. So a box can still override without editing the file.
# ---------------------------------------------------------------------------
EMBEDDED_URL = ""
EMBEDDED_API_KEY = ""

def _load_config() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}

_CONFIG = _load_config()

_EMBEDDED = {"url": EMBEDDED_URL, "api_key": EMBEDDED_API_KEY}


def _setting(name: str, default: str = "") -> str:
    return (os.environ.get(f"RUNLOG_{name.upper()}")
            or _CONFIG.get(name, "")
            or _EMBEDDED.get(name, "")
            or default)

DEFAULT_URL = _setting("url", "http://127.0.0.1:8000")

# Flush cadence. A flush now ships only what is new, so this stays short for
# the whole run.
MIN_FLUSH_SECONDS = 5.0

# Deltas are small and never rewritten, so there is no reason to back off as a
# run grows — metrics reach the dashboard in seconds instead of minutes.
MAX_FLUSH_SECONDS = 15.0
# A full snapshot every N increments bounds how many objects a reader fetches.
# Compaction rewrites the ENTIRE history into one snapshot so a reader only
# needs that snapshot plus the increments after it. Triggering it every N
# flushes made the cost scale with flush count while the benefit scales with
# data size: a 6-hour run wrote 93 snapshots of 7.4 MB — 333 MB, 96% of
# everything it ever uploaded — to save re-reading 13.5 MB of increments.
#
# Trigger on BYTES instead: compact once the increments since the last snapshot
# are worth folding in, i.e. they approach a useful fraction of the snapshot
# itself. That spaces snapshots further apart as a run grows, which is exactly
# when they are most expensive.
# Console chunk names count down from here so the newest sorts first.
DESCENDING_BASE = 999_999_999
COMPACT_RATIO = float(os.environ.get("RUNLOG_COMPACT_RATIO", "0.5"))
COMPACT_MIN_BYTES = int(os.environ.get("RUNLOG_COMPACT_MIN_BYTES", str(1_000_000)))
# A hard ceiling on how many objects a reader has to fetch, whatever the sizes.
COMPACT_MAX_CHUNKS = max(2, int(os.environ.get("RUNLOG_COMPACT_MAX_CHUNKS", "320")))

HEARTBEAT_SECONDS = 30.0

# How often machine telemetry is sampled into system/* series. Sampling shells
# out to nvidia-smi, so this is deliberately coarser than the flush interval.
SYSTEM_SAMPLE_SECONDS = float(os.environ.get("RUNLOG_SYSTEM_INTERVAL", "30"))

URL_REFRESH_MARGIN = 0.6

CONSOLE_LIMIT = 400_000

MEDIA_BATCH = 24
# Uploads run concurrently. Each file is its own presigned PUT, so a batch of
# 16 samples was 16 sequential TLS handshakes and round trips; the payload is
# small but the latency is not. Bounded so a big eval batch cannot saturate the
# uplink and starve the heartbeat.
# Four, not eight: total throughput is capped by the uplink either way, and
# more concurrency mainly makes each individual request take longer — which is
# precisely what tips a large upload over its read timeout.
UPLOAD_WORKERS = max(1, int(os.environ.get("RUNLOG_UPLOAD_WORKERS", "4")))
# Ceiling on encoded media waiting to upload. On a slow uplink a trainer can
# queue samples faster than they drain; without a bound that backlog is
# unbounded RAM inside the training process. When it is exceeded the OLDEST
# pending items are dropped — the newest samples are the ones you opened the
# dashboard to look at.
MEDIA_QUEUE_MAX_MB = float(os.environ.get("RUNLOG_MEDIA_QUEUE_MB", "256"))
MEDIA_QUEUE_MB_LABEL = f"{MEDIA_QUEUE_MAX_MB:g} MB"

# Restarts. A job restarted from its checkpoint is a new run, usually with the
# same name, and Tidy has to guess which earlier run it continued. This machine
# knows: it remembers, per project and run name, the last run it created and
# the last step that run flushed, and a new run of that name says so in its
# meta.json ("previous_run"). Purely local and best effort: another machine or
# a wiped home directory just means no hint, and nothing here can fail a run.
RESTARTS_FILE = os.path.join("~", ".runlog", "restarts.json")
RESTARTS_MAX = 200                 # names remembered; the oldest go first
RESTARTS_WRITE_SECONDS = 60.0      # how often a run updates its last step


def _restarts_path() -> str:
    return os.path.expanduser(RESTARTS_FILE)


def _restarts_read() -> dict:
    try:
        with open(_restarts_path(), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        runs = data.get("runs") if isinstance(data, dict) else None
        return runs if isinstance(runs, dict) else {}
    except Exception:
        return {}


def _restarts_key(project: str, name: str) -> str:
    return json.dumps([project, name])


def _previous_run(project: str, name: str) -> dict | None:
    entry = _restarts_read().get(_restarts_key(project, name))
    if not isinstance(entry, dict) or not isinstance(entry.get("run_id"), str) or not entry["run_id"]:
        return None
    step = entry.get("last_step")
    return {"run_id": entry["run_id"],
            "last_step": step if isinstance(step, int) and not isinstance(step, bool) else None}


def _remember_run(project: str, name: str, run_id: str, last_step: int | None) -> None:
    """Record run_id as the latest run of this name. Never raises."""
    try:
        path = _restarts_path()
        folder = os.path.dirname(path)
        os.makedirs(folder, mode=0o700, exist_ok=True)
        # Kept in the order written, newest last, so trimming drops the names
        # not used for longest.
        key = _restarts_key(project, name)
        runs = {k: v for k, v in _restarts_read().items() if isinstance(v, dict) and k != key}
        runs[key] = {"run_id": run_id, "last_step": last_step, "at": int(time.time())}
        if len(runs) > RESTARTS_MAX:
            runs = dict(list(runs.items())[-RESTARTS_MAX:])
        # Written whole and moved into place, so a reader (another run
        # starting) never sees half a file. mkstemp creates it as 0600.
        fd, temp = tempfile.mkstemp(dir=folder, prefix=".restarts-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"runs": runs}, handle)
            os.replace(temp, path)
        except BaseException:
            try:
                os.unlink(temp)
            except OSError:
                pass
            raise
    except Exception:
        pass


_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

def _slug(value: str, fallback: str = "x") -> str:
    out = _SAFE.sub("-", str(value)).strip("-")
    return (out or fallback)[:48]

def _disabled() -> bool:
    """Honour wandb's own kill switch.

    Trainers commonly set WANDB_MODE=disabled inside DataLoader worker init to
    stop the wandb service from being inherited across a fork. Respecting the same
    variable means that workaround keeps doing the right thing here.
    """
    return (
        os.environ.get("RUNLOG_MODE", "").lower() in ("disabled", "offline")
        or os.environ.get("WANDB_MODE", "").lower() in ("disabled", "offline")
    )

# Transient failures are normal on this path: B2 documents occasional 500s that
# clients are expected to retry, and a 28-hour run will hit them. Retrying here
# rather than waiting for the next flush keeps them invisible instead of
# printing an alarming line into the training log every time.
RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
# A socket read timeout raises a bare TimeoutError, which is NOT a URLError —
# so without naming these explicitly the retry treated a slow upload as a
# permanent failure and gave up after one attempt.
TRANSIENT_ERRORS = (
    TimeoutError, ConnectionError, ssl.SSLError,
    http.client.RemoteDisconnected, http.client.IncompleteRead,
    http.client.BadStatusLine,
)
# 6 attempts with jitter, not 4 in lockstep. Two reasons: a B2 capacity blip
# ("503 ServiceUnavailable") usually clears in tens of seconds, and the upload
# pool runs several PUTs at once — undithered backoff makes all of them retry
# on the same instants, which is exactly the thundering herd the store is
# already struggling with. Worst case here is ~1+2+4+8+16 = 31s of retrying
# inside one call, still far below the flush interval.
RETRY_ATTEMPTS = 6
RETRY_MAX_DELAY = 16.0
# finish() gets its own, more patient budget: no later flush can cover for it.
FINAL_FLUSH_ATTEMPTS = 6
FINAL_FLUSH_BUDGET = 90.0


def _describe(exc: Exception) -> str:
    """A failure message that names the cause, not just the status number."""
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read(400).decode("utf-8", "replace").strip()
        except Exception:
            body = ""
        return f"HTTP {exc.code} {exc.reason or ''}".strip() + (f": {body}" if body else "")
    if isinstance(exc, urllib.error.URLError):
        return f"network error: {exc.reason}"
    return f"{type(exc).__name__}: {exc}"


def _send(build, timeout: float, what: str):
    """Run a request, backing off on transient failures."""
    delay = 1.0
    for attempt in range(RETRY_ATTEMPTS):
        try:
            with urllib.request.urlopen(build(), timeout=timeout) as response:
                return response.read()
        except Exception as exc:
            transient = (
                isinstance(exc, urllib.error.HTTPError) and exc.code in RETRY_STATUSES
            ) or (
                isinstance(exc, urllib.error.URLError)
                and not isinstance(exc, urllib.error.HTTPError)
            ) or isinstance(exc, TRANSIENT_ERRORS)
            if not transient or attempt == RETRY_ATTEMPTS - 1:
                raise RuntimeError(f"{what}: {_describe(exc)}") from None
            time.sleep(delay * (0.5 + random.random()))   # jitter: ±50%
            delay = min(RETRY_MAX_DELAY, delay * 2)
    return b""


def _post(url: str, payload: dict, api_key: str, timeout: float = 30.0) -> dict:
    def build():
        return urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"},
        )
    body = _send(build, timeout, f"POST {url.split('?')[0]}")
    return json.loads(body or b"{}")

# Kept below the join timeout in finish(), so shutdown cannot proceed while the
# loop thread is still inside a socket read.
PUT_TIMEOUT = 120.0
PUT_TIMEOUT_MAX = 900.0
# Assume an uplink at least this fast when budgeting a PUT's timeout.
MIN_UPLOAD_BYTES_PER_SEC = 25_000

def _put(url: str, data: bytes, content_type: str, timeout: float | None = None) -> None:
    def build():
        return urllib.request.Request(
            url, data=data, method="PUT", headers={"Content-Type": content_type}
        )
    # A flat timeout punishes big objects on a slow uplink: a 5 MB sample at
    # 100 KB/s legitimately needs longer than a 3 KB metric delta. Allow time
    # proportional to the payload, with a floor and a hard ceiling.
    if timeout is None:
        timeout = min(PUT_TIMEOUT_MAX,
                      max(PUT_TIMEOUT, len(data) / MIN_UPLOAD_BYTES_PER_SEC))
    # Name the host, not the signed URL: the query string carries the signature.
    host = url.split("//", 1)[-1].split("/", 1)[0]
    _send(build, timeout, f"PUT to {host}")

# ANSI styling, applied only when the destination is a real terminal. Training
# output is routinely piped into nohup.out, tee, CI logs and tqdm-wrapped
# pipes; escape codes written into those are noise nobody asked for. NO_COLOR
# and TERM=dumb are honoured because they are the conventions people already
# set when they want plain text.
_ANSI = {
    "dim": "\033[2m", "bold": "\033[1m", "cyan": "\033[36m",
    "yellow": "\033[33m", "red": "\033[31m", "under": "\033[4m", "off": "\033[0m",
}


def _color_ok(stream) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM", "") == "dumb":
        return False
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _paint(stream, text: str, *styles: str) -> str:
    if not styles or not _color_ok(stream):
        return text
    return "".join(_ANSI[s] for s in styles if s in _ANSI) + text + _ANSI["off"]


def _content_type(ext: str) -> str:
    return {
        "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "webp": "image/webp", "gif": "image/gif", "mp4": "video/mp4",
        "webm": "video/webm", "wav": "audio/wav", "mp3": "audio/mpeg",
        "flac": "audio/flac", "html": "text/html", "txt": "text/plain",
        "json": "application/json",
    }.get(ext.lower(), "application/octet-stream")

class _Tee(io.TextIOBase):
    """Mirror stdout/stderr into the run's console buffer without stealing it.

    Everything the wrapped stream exposes has to keep working, because callers
    inspect it. tqdm in particular reads `encoding` to decide whether it may
    draw its Unicode block bar: TextIOBase.encoding is None, so a bar that
    looked like "█████" downgraded to "#####" the moment output was captured.
    It reads isatty() and fileno() for the terminal width too.
    """

    def __init__(self, stream, run: "Run") -> None:
        self._stream = stream
        self._run = run

    def write(self, text: str) -> int:
        self._run._append_console(text)
        return self._stream.write(text)

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

    def fileno(self) -> int:
        return self._stream.fileno()

    # TextIOBase defines these, so __getattr__ below would never see them.
    @property
    def encoding(self):
        return getattr(self._stream, "encoding", None)

    @property
    def errors(self):
        return getattr(self._stream, "errors", None)

    @property
    def newlines(self):
        return getattr(self._stream, "newlines", None)

    @property
    def buffer(self):
        return getattr(self._stream, "buffer", None)

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    @property
    def closed(self) -> bool:
        return getattr(self._stream, "closed", False)

    def close(self) -> None:
        # The real stream is the caller's; capturing must not close it.
        pass

    def __getattr__(self, name):
        # Anything else (mode, name, reconfigure, ...) belongs to the stream.
        return getattr(self._stream, name)

class Run:
    def __init__(
        self,
        project: str = "default",
        name: str | None = None,
        config: dict | None = None,
        notes: str = "",
        url: str | None = None,
        api_key: str | None = None,
        capture_output: bool = True,
        system_info: bool = True,
        system_metrics: bool = True,
        **_wandb_kwargs: Any,   # entity, tags, group, resume, settings, …
    ) -> None:
        self._system_metrics = system_metrics
        self.base_url = (url or DEFAULT_URL).rstrip("/")
        self.api_key = api_key or _setting("api_key")
        if not self.api_key:
            raise RuntimeError(
                "No API key. Either set RUNLOG_URL and RUNLOG_API_KEY, or bake "
                "them into this file on a machine that has them: "
                "python wandb.py --embed-key"
            )

        self.project = _slug(project or "default", "default")
        self._name = name or time.strftime("%m-%d %H:%M")
        self.config = dict(config or {})
        self.notes = notes
        self.summary: dict[str, Any] = {}

        # Everything that follows a fork must be inert; a DataLoader worker
        # inheriting this object would otherwise run atexit and mark the
        # parent's still-running run as finished.
        self._pid = os.getpid()

        self._lock = threading.Lock()
        self._metrics: dict[str, dict[str, list]] = {}
        self._media_queue: list[tuple[str, bytes, str]] = []
        self._media_index: dict[str, dict[str, Any]] = {}
        # Monotonic name counter per (label, step); never rewound by trimming.
        self._media_seq: dict[tuple, int] = {}
        self._media_dirty = False
        self._dropped_media = 0
        self._dropped_warned = 0.0
        self._meta_dirty = False
        self._console = io.StringIO()
        self._console_dirty = False
        self._series_bytes = 0
        self._series_seq = 0
        self._console_seq = 0
        self._snapshot_bytes = 0      # size of the newest full snapshot
        self._delta_bytes = 0         # increment bytes written since it
        self._since_snapshot = 0      # increment COUNT since it
        self._console_sent = 0     # chars already shipped, for the append-only tail
        self._url_bank: dict[str, tuple[str, float]] = {}   # presigned PUTs, single-use
        self._fail_streak = 0            # consecutive failed flush ticks
        self._fail_last = ""             # last message printed, to suppress repeats
        self._sent: dict[str, int] = {}   # points already shipped, per metric
        self._step = 0
        # Telemetry rides on the training step, but until the first log() there
        # is no real step to ride on — and self._step starts at 0 even for a
        # run RESUMED at step 15000. A sample taken then lands at step 0 and
        # stretches every system chart across the whole run: one flat line from
        # 0 to the resume point, with the real telemetry crushed into the last
        # few percent of the panel.
        self._stepped = False
        self._closed = False
        self._last_flush = 0.0
        self._last_heartbeat = 0.0
        self._watching: list[tuple[Any, int]] = []
        self._last_system = 0.0
        self._hist_queue: list[tuple[str, bytes]] = []
        self._hist_steps: list[int] = []
        self._hist_tensors: list[str] = []
        self._hist_kinds: list[str] = []
        self._hist_dirty = False
        self._hist_truncated = False
        self._last_hist = 0.0
        # Captured before the first metric so a run that dies in its first
        # minute still records what machine it was on.
        self._name_ready = True
        self.system = collect_system_info() if system_info else {}

        created = _post(
            f"{self.base_url}/track/runs",
            {"project": self.project, "name": self.name,
             "config": self.config, "notes": self.notes},
            self.api_key,
        )
        self.run_id = self.id = created["run_id"]
        self.project = created["project"]
        self._urls = created["urls"]
        self._urls_expire = time.time() + created.get("expires_in", 7200) * URL_REFRESH_MARGIN
        self.started_at = int(time.time())
        # See RESTARTS_FILE. Only a name the script chose: a default timestamp
        # name says nothing about which job this is.
        self._restart_name = str(name) if name else None
        self._previous = _previous_run(self.project, self._restart_name) if name else None
        self._remembered: tuple[int | None, float] = (None, 0.0)

        # Written immediately so a run that dies in its first seconds is still
        # identifiable rather than an unnamed directory.
        self._write_meta()
        self._write_label()

        self._stdout = self._stderr = None
        if capture_output:
            self._stdout, self._stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = _Tee(sys.stdout, self), _Tee(sys.stderr, self)

        # atexit runs after CPython has cleared the exception, so sys.exc_info()
        # there is always empty and every crash was being recorded as a clean
        # finish. Catch it where it is still visible.
        self._prev_excepthook = sys.excepthook
        sys.excepthook = self._excepthook
        self._crash: str = ""

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="runlog", daemon=True)
        self._thread.start()
        atexit.register(self._atexit)
        self._urgent = False
        self._install_signal_handlers()

        # Printed like wandb's own startup line: a link straight to THIS run,
        # not to the dashboard root.
        self.url = f"{self.base_url}/runlog#{self.project}/{self.run_id}"
        stream = self._stdout or sys.__stdout__
        print(
            _paint(stream, "runlog:", "dim")
            + " view run "
            + _paint(stream, self.name, "bold")
            + " at "
            + _paint(stream, self.url, "cyan", "under"),
            file=stream,
        )

    # -- public API ---------------------------------------------------------

    def log(self, values: dict[str, Any], step: int | None = None, **_kwargs: Any) -> None:
        """Record scalars and/or  `step` defaults to an auto counter."""
        if self._closed or os.getpid() != self._pid or not isinstance(values, dict):
            return
        with self._lock:
            if step is None:
                step = self._step
                self._step += 1
            else:
                step = int(step)
                self._step = max(self._step, step + 1)
            self._stepped = True

        media: list[tuple[str, Any]] = []
        with self._lock:
            for key, value in values.items():
                key = str(key)
                if isinstance(value, Media):
                    media.append((key, value))
                    continue
                if isinstance(value, (list, tuple)):
                    if value and all(isinstance(v, Media) for v in value):
                        media.extend((key, v) for v in value)
                    continue
                if hasattr(value, "item"):  # 0-d tensor / numpy scalar
                    try:
                        value = value.item()
                    except Exception:
                        pass
                if isinstance(value, str):
                    self.summary[key] = value
                    continue
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                value = float(value)
                if value != value or value in (float("inf"), float("-inf")):
                    # NaN/inf would blow out the chart's y-range for every other
                    # point. Keeping them out of the series is more useful than
                    # a flat line at infinity; they still appear in the console.
                    continue
                entry = self._metrics.setdefault(key, {"steps": [], "values": [], "wall": []})
                entry["steps"].append(step)
                entry["values"].append(value)
                # Seconds since the run began, as an int. Costs a few bytes per
                # point and is the only way to offer a wall-clock x axis; the
                # step axis alone cannot show that throughput changed.
                entry.setdefault("wall", []).append(int(time.time() - self.started_at))
                self.summary[key] = value

        for label, item in media:
            self._queue_media_object(label, item, step)
        if self._watching:
            self._watch_step(step)

    def log_image(self, label: str, image: Any, step: int | None = None) -> None:
        self.log({label: Image(image)}, step=step)

    def log_text(self, label: str, text: str, step: int | None = None) -> None:
        self.log({label: Html("<pre>" + str(text) + "</pre>")}, step=step)

    def watch(self, model: Any, log: str | None = "all", log_freq: int = 1000, **_kwargs: Any) -> None:
        """Per-tensor histograms plus global norms, on wandb's vocabulary.

        `log` is 'parameters', 'gradients', 'all' or None. Norms are recorded
        every `log_freq` steps because they are two numbers; histograms are
        rate-limited separately by wall clock (see MIN_INTERVAL),
        since `log_freq` here derives from `save_every` and on a fast run would
        otherwise mean a snapshot every few seconds.
        """
        if model is None:
            return
        try:
            watcher = Watcher(model, log or "all")
        except Exception as exc:
            print(f"runlog: could not watch model: {exc}", file=sys.__stderr__)
            return
        self._watching.append((watcher, max(1, int(log_freq or 1000))))

    def _install_signal_handlers(self) -> None:
        """Finish the run when the process is told to stop.

        SIGTERM (kill, a scheduler's time limit, `docker stop`) and SIGHUP (the
        terminal or the SSH session went away) end the interpreter without
        running atexit. The run then never wrote its last points, never
        compacted its history into one object and never said it had ended: the
        dashboard showed it as "stale", and every later reader fetched one
        snapshot plus dozens of small increments instead of one object. Most
        runs ended this way.

        The handler does no work itself. It leaves through SystemExit, the way
        Ctrl-C leaves through KeyboardInterrupt, so `finally` blocks run and
        atexit finishes the run. A handler somebody else installed first still
        gets the signal and decides what happens; a signal the process ignores
        (nohup) stays ignored. Only the main thread can install handlers.
        """
        self._prev_signals: dict[int, Any] = {}
        if threading.current_thread() is not threading.main_thread():
            return
        for name in ("SIGTERM", "SIGHUP"):
            signum = getattr(signal, name, None)
            if signum is None:                      # no SIGHUP on Windows
                continue
            try:
                previous = signal.getsignal(signum)
                if previous is signal.SIG_IGN or previous is None:
                    continue
                signal.signal(signum, self._on_signal)
                self._prev_signals[signum] = previous
            except (ValueError, OSError):
                continue
        if self._prev_signals and hasattr(os, "register_at_fork"):
            # A forked child (a DataLoader worker) must die of SIGTERM exactly
            # as it did before: give it back the handlers this run replaced.
            os.register_at_fork(after_in_child=self._restore_signal_handlers)

    def _restore_signal_handlers(self) -> None:
        for signum, previous in list(getattr(self, "_prev_signals", {}).items()):
            try:
                if signal.getsignal(signum) == self._on_signal:
                    signal.signal(signum, previous)
            except (ValueError, OSError, TypeError):
                pass
        self._prev_signals = {}

    def _on_signal(self, signum, frame) -> None:
        previous = self._prev_signals.get(signum)

        def stopping() -> None:
            if os.getpid() != self._pid or self._closed:
                return
            try:
                label = signal.Signals(signum).name
            except ValueError:
                label = f"signal {signum}"
            # Recorded as an abnormal end, like Ctrl-C: the run did not finish.
            self._crash = self._crash or f"Stopped by {label}."
            self._urgent = True

        if callable(previous):
            try:
                previous(signum, frame)
            except BaseException:
                stopping()          # their handler ends the process: the run ends with it
                raise
            return                  # their handler chose to carry on, so the run does too
        stopping()
        raise SystemExit(128 + signum)

    def finish(self, status: str = "finished", error: str = "") -> None:
        if self._closed or os.getpid() != self._pid:
            return
        self._closed = True
        self._stop.set()
        urgent = self._urgent
        if self._thread.is_alive():
            self._thread.join(timeout=10 if urgent else PUT_TIMEOUT_MAX + 60)
        for watcher, _ in self._watching:
            watcher.close()
        if self._stdout is not None:
            sys.stdout, sys.stderr = self._stdout, self._stderr
        self._restore_signal_handlers()
        if urgent:
            # Told to stop: whoever sent the signal usually follows it with
            # SIGKILL seconds later. Do the cheap, important things first (the
            # last points, then the end marker) and the large upload last, so
            # being killed part-way costs only the compaction.
            try:
                self._flush(final=False)
            except Exception as exc:
                print(_paint(sys.__stderr__, f"runlog: last flush failed: {exc}", "red"), file=sys.__stderr__)
            self._write_final_status(status, error)
            try:
                self._flush(final=True)
            except Exception as exc:
                print(_paint(sys.__stderr__, f"runlog: final compaction failed: {exc}", "yellow"),
                      file=sys.__stderr__)
            sys.excepthook = self._prev_excepthook
            return
        # The final flush is the one that cannot be retried by the next tick —
        # there is no next tick. It is also the flush that matters most: it
        # compacts the whole history into one snapshot. A store having a bad
        # minute at exactly the wrong moment would otherwise truncate the run's
        # tail permanently, so persist here rather than giving up after one go.
        for attempt in range(FINAL_FLUSH_ATTEMPTS):
            try:
                self._flush(final=True)
                break
            except Exception as exc:
                last = exc
                if attempt < FINAL_FLUSH_ATTEMPTS - 1:
                    if attempt == 0:
                        print(_paint(sys.__stderr__,
                                     f"runlog: final flush failed ({exc}); "
                                     f"retrying for up to {FINAL_FLUSH_BUDGET:.0f}s",
                                     "yellow"), file=sys.__stderr__)
                    time.sleep(min(20.0, 2.0 * (attempt + 1)) * (0.5 + random.random()))
        else:  # never let teardown mask the real failure
            print(_paint(sys.__stderr__, f"runlog: final flush failed: {last}", "red"),
                  file=sys.__stderr__)
        try:
            self._write_final_status(status, error)
        finally:
            sys.excepthook = self._prev_excepthook

    def _write_final_status(self, status: str, error: str) -> None:
        try:
            # Separate from the final flush: if the last upload fails, the run
            # must still be marked finished, or the dashboard shows a completed
            # run as "stale" forever.
            key = "final_crash" if status == "crashed" else "final_ok"
            self._put_json(self._urls[key], {
                "status": status, "error": error,
                "ended_at": int(time.time()), "summary": self.summary,
            })
        except Exception as exc:
            print(_paint(sys.__stderr__, f"runlog: could not write final status: {exc}", "red"),
                  file=sys.__stderr__)

    # -- internals ----------------------------------------------------------

    def _write_label(self) -> None:
        """Publish the display name as an object KEY, not as file content.

        The run list is built from a single object listing with no GETs, so a
        name kept inside meta.json would be invisible there — which is why long
        checkpoint names showed up truncated to the 48-character slug baked
        into the run id. Percent-encoded so slashes in a checkpoint path do not
        create nested keys.
        """
        encoded = urllib.parse.quote(self.name, safe="")[:180]
        try:
            signed = _post(
                f"{self.base_url}/track/runs/{self.run_id}/urls",
                {"project": self.project, "label": encoded}, self.api_key,
            )
            target = (signed.get("label") or {}).get(encoded)
            if target:
                _put(target, b"", "text/plain", timeout=20)
        except Exception as exc:
            print(f"runlog: could not publish the run name: {exc}", file=sys.__stderr__)

    def rename(self, name: str) -> None:
        """Rename the run from code. `run.name = "..."` does the same."""
        name = str(name).strip()
        if not name or name == self.name or os.getpid() != self._pid:
            return
        # Not `self.name = name`: the property setter calls rename(), which
        # recursed until the interpreter gave up.
        object.__setattr__(self, "_name", name)
        self._meta_dirty = True
        self._write_label()

    def _write_meta(self) -> None:
        self._put_json(self._urls["meta"], {
            "name": self.name, "project": self.project, "run_id": self.run_id,
            "config": self.config, "notes": self.notes, "started_at": self.started_at,
            "system": self.system,
            **({"previous_run": self._previous} if getattr(self, "_previous", None) else {}),
        })
        self._meta_dirty = False

    def _remember_step(self, step: int | None, final: bool) -> None:
        """Make this run the one a restart of the same name continues.

        Only once it has flushed a step: a retry that dies before its first
        step must not hide the run that actually wrote the checkpoint. Then at
        most once a minute, and on the last flush.
        """
        if self._restart_name is None or step is None:
            return
        held, at = self._remembered
        now = time.time()
        if step == held or (not final and held is not None and now - at < RESTARTS_WRITE_SECONDS):
            return
        _remember_run(self.project, self._restart_name, self.run_id, step)
        self._remembered = (step, now)

    def _queue_media_object(self, label: str, item: Media, step: int) -> None:
        try:
            data, ext = item.encode()
        except Exception as exc:
            print(f"runlog: could not encode {label!r}: {exc}", file=sys.__stderr__)
            return
        with self._lock:
            # Zero-padded step first so a plain key listing already sorts
            # chronologically per label; the index below carries the parts that
            # do not survive slugification (the real label, the caption).
            #
            # The per-(label, step) counter is monotonic and kept separately on
            # purpose. It used to be derived by counting _media_index, which
            # _trim_media_queue() removes entries from — so as soon as the
            # backlog was trimmed the count went BACKWARDS, later clips reused
            # names earlier clips already had, and each collision silently
            # overwrote a file. A batch of 16 could land as 8.
            seq_key = (label, step)
            index = self._media_seq.get(seq_key, 0)
            self._media_seq[seq_key] = index + 1
            name = f"{step:09d}__{_slug(label, 'media')}__{index:03d}.{_slug(ext, 'bin')}"
            self._media_queue.append((name, data, ext))
            self._media_index[name] = {
                "label": label, "caption": item.caption,
                "step": step, "kind": item.kind,
            }
            self._media_dirty = True
            # Trim last, so the item just queued is accounted for.
            self._trim_media_queue()

    def _record(self, values: dict[str, float], step: int | None = None) -> None:
        """Append scalars at the run's current step without advancing it.

        Telemetry is sampled on a timer, not on a training step, so it rides
        along at whatever step the loop has reached. Advancing the counter here
        would desynchronise the auto-step used by callers that omit `step=`.
        """
        if not values:
            return
        with self._lock:
            at = self._step if step is None else int(step)
            for key, value in values.items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                value = float(value)
                if value != value or value in (float("inf"), float("-inf")):
                    continue
                entry = self._metrics.setdefault(key, {"steps": [], "values": [], "wall": []})
                entry["steps"].append(at)
                entry["values"].append(value)
                entry.setdefault("wall", []).append(int(time.time() - self.started_at))
                self.summary[key] = value

    def _watch_step(self, step: int) -> None:
        """Emit whatever the hooks captured, then arm them for the next pass.

        Order matters. A gradient histogram recorded during backward at step N
        is only available to the logging call that follows it, so each visit
        first drains what is ready and then requests the next capture.
        """
        for watcher, freq in self._watching:
            if watcher.has_gradients or (step % freq == 0 and watcher.mode == "parameters"):
                self._emit_histograms(watcher, step)

            grad_norm = watcher.grad_norm()
            if grad_norm is not None:
                param_norm = watcher.param_norm()
                values = {"watch/grad_norm": grad_norm}
                if param_norm is not None:
                    values["watch/param_norm"] = param_norm
                self._record(values, step)

            if step % freq:
                continue
            now = time.time()
            want_hist = now - self._last_hist >= MIN_INTERVAL
            if want_hist:
                self._last_hist = now
            watcher.arm(want_hist)

    def _emit_histograms(self, watcher: Any, step: int) -> None:
        try:
            data = watcher.collect()
        except Exception as exc:
            print(f"runlog: histogram snapshot failed: {exc}", file=sys.__stderr__)
            return
        if not data:
            return
        body = json.dumps({"step": step, "bins": watcher.bins, "tensors": data}).encode()
        with self._lock:
            self._hist_queue.append((f"{step:09d}.json", body))
            self._hist_steps.append(step)
            names = list(data.keys())
            if not self._hist_tensors:
                self._hist_tensors = names
            else:
                seen = set(self._hist_tensors)
                self._hist_tensors += [n for n in names if n not in seen]
            kinds = {k for entry in data.values() for k in entry}
            self._hist_kinds = sorted(set(self._hist_kinds) | kinds)
            self._hist_truncated = self._hist_truncated or bool(getattr(watcher, "truncated", False))
            self._hist_dirty = True

    def _trim_media_queue(self) -> None:
        """Bound the pending-upload backlog. Caller holds the lock."""
        limit = MEDIA_QUEUE_MAX_MB * 1_000_000
        total = sum(len(data) for _, data, _ in self._media_queue)
        if total <= limit:
            return
        dropped = 0
        while self._media_queue and total > limit:
            name, data, _ = self._media_queue.pop(0)
            self._media_index.pop(name, None)
            total -= len(data)
            dropped += 1
        self._dropped_media += dropped
        now = time.time()
        # One line a minute, not one per dropped file.
        if now - self._dropped_warned > 60:
            self._dropped_warned = now
            print(_paint(sys.__stderr__,
                         f"runlog: upload backlog over {MEDIA_QUEUE_MB_LABEL}; dropped "
                         f"{self._dropped_media} oldest samples so far. The uplink cannot "
                         f"keep up with how often this run logs media.", "yellow"),
                  file=sys.__stderr__)

    def _append_console(self, text: str) -> None:
        # Fork guard BEFORE the lock, not after. Only the forking thread
        # survives fork(), so a child that inherits _lock while the flush
        # thread held it can never acquire it — and every print() in a
        # DataLoader worker goes through here. The child simply does not
        # capture console output.
        if os.getpid() != self._pid:
            return
        with self._lock:
            # A progress bar redraws by returning to the start of the line, so
            # a 4-hour tqdm writes tens of thousands of near-identical frames.
            # Stored verbatim they crowd everything else out of the console
            # view and upload for no benefit. Keep only the newest frame of an
            # unfinished line, exactly as a terminal shows it.
            if "\r" in text:
                text = self._collapse_carriage_returns(text)
            self._console.write(text)
            self._console_dirty = True
            if self._console.tell() > CONSOLE_LIMIT * 2:
                whole = self._console.getvalue()
                tail = whole[-CONSOLE_LIMIT:]
                self._console = io.StringIO()
                self._console.write(tail)
                # The buffer just lost its head; the sent-marker indexes into
                # it, so shift it by the same amount or the next chunk would
                # re-ship text already uploaded.
                self._console_sent = max(0, self._console_sent - (len(whole) - len(tail)))

    def _collapse_carriage_returns(self, text: str) -> str:
        """Apply \r the way a terminal would: it rewinds, it does not scroll.

        Called with the lock held. Rewrites the tail of the buffer in place
        when the incoming chunk rewinds over a line already written.
        """
        head, sep, tail = text.rpartition("\n")
        if sep:                       # completed lines: only the tail rewinds
            head = head.rpartition("\r")[2] if "\r" in head else head
        current = tail.rpartition("\r")[2] if "\r" in tail else tail
        if sep:
            return head + "\n" + current
        # No newline at all: this chunk overwrites the unfinished line already
        # in the buffer, so drop that line before appending the new frame.
        if "\r" in text:
            buffered = self._console.getvalue()
            keep = buffered.rpartition("\n")[0]
            self._console = io.StringIO()
            if keep:
                self._console.write(keep + "\n")
            # Dropping the unfinished line shortens the buffer; clamp the
            # sent-marker so it never points past the end.
            self._console_sent = min(self._console_sent, self._console.tell())
        return current

    def _flush_interval(self) -> float:
        """Nearly constant now that a flush ships only what is new.

        The old backoff existed because every flush re-uploaded the entire
        history; a long run ended up waiting two minutes between updates. A
        delta is a few KB regardless of run length.
        """
        megabytes = self._series_bytes / 1_000_000
        return min(MAX_FLUSH_SECONDS, max(MIN_FLUSH_SECONDS, megabytes * 2.0))

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(1.0)
            now = time.time()
            try:
                if now - self._last_heartbeat >= HEARTBEAT_SECONDS:
                    self._refresh_urls_if_needed()
                    _put(self._urls["heartbeat"], b"", "text/plain", timeout=15)
                    self._last_heartbeat = now
                # Wait for a real step before sampling: see self._stepped.
                # Sampling resumes on the very next tick after the first log,
                # so at most one interval of telemetry is given up.
                if (self._system_metrics and self._stepped
                        and now - self._last_system >= SYSTEM_SAMPLE_SECONDS):
                    self._last_system = now
                    self._record(sample())
                # Media ships on its own, WITHOUT rewriting the series. These
                # were coupled, so logging one audio sample forced a full
                # re-upload of series.json on the next tick — 11 MB at 77k
                # steps, every time, which saturates the uplink and looks like
                # the logging call has frozen.
                if self._media_queue or self._hist_queue:
                    self._drain_attachments()
                if now - self._last_flush >= self._flush_interval():
                    self._flush()
                if self._fail_streak:
                    print(_paint(sys.__stderr__,
                                 f"runlog: uploads recovered after {self._fail_streak} "
                                 f"failed attempt{'s' if self._fail_streak > 1 else ''}",
                                 "green"), file=sys.__stderr__)
                    self._fail_streak, self._fail_last = 0, ""
            except Exception as exc:
                # A transient upload failure must not kill training. Everything
                # is requeued, so the next tick simply retries.
                #
                # Print ONE line per distinct problem rather than one per tick:
                # an object store can be unhappy for minutes, and a wall of
                # identical warnings buries the training log it is printed
                # into. Repeats are counted and re-surfaced as the outage
                # lengthens, and recovery says so explicitly.
                self._fail_streak += 1
                message = str(exc)
                milestone = self._fail_streak in (5, 20, 60, 200)
                if message != self._fail_last or milestone:
                    suffix = (f" (x{self._fail_streak}, still retrying)"
                              if self._fail_streak > 1 else " — queued, retrying")
                    print(_paint(sys.__stderr__,
                                 f"runlog: upload failed ({message}){suffix}", "yellow"),
                          file=sys.__stderr__)
                    self._fail_last = message

    def _drain_attachments(self) -> None:
        """Upload queued media and histogram snapshots only.

        Separate from _flush because these are immutable objects — each one is
        written once — whereas the series is rewritten whole. Coupling them
        made the cost of logging a sample scale with the length of the run.
        """
        self._refresh_urls_if_needed()
        with self._lock:
            pending, self._media_queue = self._media_queue, []
            hist_pending, self._hist_queue = self._hist_queue, []
            media_index = dict(self._media_index) if self._media_dirty else None
            self._media_dirty = False
            hist_index = {
                "steps": list(self._hist_steps), "tensors": list(self._hist_tensors),
                "kinds": list(self._hist_kinds), "bins": BINS,
                "truncated": self._hist_truncated,
            } if self._hist_dirty and self._hist_steps else None
            self._hist_dirty = False
        try:
            if pending:
                self._upload_media(pending)
                pending = []
            if hist_pending:
                self._upload_histograms(hist_pending)
                hist_pending = []
            if media_index is not None:
                self._put_json(self._urls["media_index"], media_index)
            if hist_index is not None:
                self._put_json(self._urls["hist_index"], hist_index)
        except Exception:
            with self._lock:
                self._media_queue = pending + self._media_queue
                self._hist_queue = hist_pending + self._hist_queue
                self._media_dirty = self._media_dirty or media_index is not None
                self._hist_dirty = self._hist_dirty or hist_index is not None
            raise

    def _flush(self, final: bool = False) -> None:
        self._refresh_urls_if_needed()
        with self._lock:
            # Append-only: each flush writes ONLY the points added since the
            # last one, under a key that is written exactly once. Nothing is
            # ever overwritten, so no old versions pile up — and the upload is
            # a few KB whether the run is at step 100 or step 100,000.
            self._series_seq += 1
            seq = self._series_seq
            # See COMPACT_RATIO: fold in the increments once they are worth
            # the rewrite, or once a reader would face too many objects.
            threshold = max(COMPACT_MIN_BYTES, int(self._snapshot_bytes * COMPACT_RATIO))
            compacting = (final or self._snapshot_bytes == 0
                          or self._delta_bytes >= threshold
                          or self._since_snapshot >= COMPACT_MAX_CHUNKS)
            delta: dict[str, dict] = {}
            # _sent is advanced below as if the upload had already succeeded.
            # If it has not, those points belong to a chunk that never landed,
            # and the next flush would start after them — leaving a hole in the
            # chart until the next compaction rewrote everything (up to
            # the next compaction, or never if the run dies first).
            sent_before = dict(self._sent)
            for key, entry in self._metrics.items():
                start = 0 if compacting else self._sent.get(key, 0)
                if start >= len(entry["steps"]) and not compacting:
                    continue
                delta[key] = {
                    "steps": list(entry["steps"][start:]),
                    "values": list(entry["values"][start:]),
                    "wall": list((entry.get("wall") or [])[start:]),
                }
                self._sent[key] = len(entry["steps"])
            payload = {
                "name": self.name, "run_id": self.run_id, "seq": seq,
                "started_at": self.started_at, "updated_at": int(time.time()),
                "config": self.config, "summary": dict(self.summary),
                "metrics": delta,
            }
            pending, self._media_queue = self._media_queue, []
            hist_pending, self._hist_queue = self._hist_queue, []
            hist_index = {
                "steps": list(self._hist_steps),
                "tensors": list(self._hist_tensors),
                "kinds": list(self._hist_kinds),
                "bins": BINS,
                "truncated": self._hist_truncated,
            } if (self._hist_dirty or final) and self._hist_steps else None
            self._hist_dirty = False
            media_index = dict(self._media_index) if (self._media_dirty or final) else None
            self._media_dirty = False
            # Append-only, exactly like the series. The whole buffer used to be
            # re-uploaded on every flush; with versioning on the bucket, every
            # one of those rewrites is retained forever — 4 GB of dead console
            # versions for a handful of runs. Ship only the new text.
            console = None
            if self._console_dirty or final:
                whole = self._console.getvalue()
                console = whole[self._console_sent:]
                self._console_sent = len(whole)
                if not console:
                    console = None
            self._console_dirty = False
            meta_dirty = self._meta_dirty
            last_step = self._step - 1 if self._stepped else None

        # Everything below can fail on a network blip. The queues were already
        # drained into locals, so ANY failure has to put them back — including
        # a failure in the series or console PUT, which happen first. Losing a
        # sample or a histogram snapshot to an unrelated upload error would be
        # silent and unrecoverable.
        series_ok = False
        try:
            if meta_dirty:
                self._write_meta()
            if payload["metrics"] or final:
                body = json.dumps(payload).encode()
                self._series_bytes = len(body)
                # `c` marks a full snapshot, `d` an increment. A reader takes
                # the newest snapshot plus every increment after it.
                name = f"{'c' if compacting else 'd'}{seq:06d}.json"
                self._put_series(name, body)
                if compacting:
                    self._snapshot_bytes = len(body)
                    self._delta_bytes = 0
                    self._since_snapshot = 0
                else:
                    self._delta_bytes += len(body)
                    self._since_snapshot += 1
            # Past this line the points are durably stored, so a later failure
            # in this same block (console, media, an index) must NOT rewind
            # _sent — that would resend points already on the server and
            # duplicate them in the merged history.
            series_ok = True
            self._remember_step(last_step, final)
            if console is not None:
                self._console_seq += 1
                self._put_console(self._console_seq, console)
            if pending:
                self._upload_media(pending)
                pending = []
            if hist_pending:
                self._upload_histograms(hist_pending)
                hist_pending = []
            if hist_index is not None:
                self._put_json(self._urls["hist_index"], hist_index)
            if media_index is not None:
                self._put_json(self._urls["media_index"], media_index)
        except Exception:
            with self._lock:   # requeue whatever did not make it; next tick retries
                self._media_queue = pending + self._media_queue
                self._hist_queue = hist_pending + self._hist_queue
                self._media_dirty = self._media_dirty or media_index is not None
                self._hist_dirty = self._hist_dirty or hist_index is not None
                # Rewind to what was actually shipped so the next flush resends
                # the delta — but only when the chunk itself did not land, or
                # the resend would duplicate points already stored. Points
                # appended meanwhile are still included: this is an index into
                # the history, not a copy of it. The consumed seq number is
                # deliberately NOT reused — a gap in the chunk sequence is
                # harmless to the reader, whereas reusing a key could overwrite
                # a chunk that landed but whose acknowledgement was lost.
                if not series_ok:
                    self._sent = sent_before
            raise

        self._last_flush = time.time()

    def _bank_urls(self, series_seq: int, console_seq: int) -> None:
        """One signing call that covers the next many flushes.

        Signing was requested per chunk: one serverless invocation per flush,
        two with console output — ~34,000 invocations a day from a single
        24/7 trainer, which alone exhausts a free hosting tier's CPU budget.
        Chunk names are deterministic, so ask for the next 16 of everything
        at once (both c- and d- series names, since which one is needed
        depends on when compaction triggers).
        """
        series_names = []
        for seq in range(series_seq, series_seq + 16):
            series_names += [f"c{seq:06d}.json", f"d{seq:06d}.json"]
        console_names = [f"{DESCENDING_BASE - s:09d}.txt"
                         for s in range(console_seq, console_seq + 16)]
        signed = _post(
            f"{self.base_url}/track/runs/{self.run_id}/urls",
            {"project": self.project, "series": series_names,
             "console_chunks": console_names}, self.api_key,
        )
        expires = time.time() + signed.get("expires_in", 7200) * URL_REFRESH_MARGIN
        for name, url in (signed.get("series") or {}).items():
            self._url_bank[f"series/{name}"] = (url, expires)
        for name, url in (signed.get("console_chunks") or {}).items():
            self._url_bank[f"clog/{name}"] = (url, expires)
        # Old servers know nothing of console_chunks and return series only;
        # the bank simply stays empty for those and the per-chunk path signs.

    def _banked(self, key: str) -> str | None:
        hit = self._url_bank.pop(key, None)      # single use: PUT once, gone
        if hit and hit[1] > time.time():
            return hit[0]
        return None

    def _put_console(self, seq: int, text: str) -> None:
        """Write one console chunk under a key that is written exactly once."""
        # Counted DOWN, so lexicographic order puts the NEWEST chunk first and
        # the dashboard can read the tail from a single page. Ascending names
        # meant listing every chunk of the run — 5,000 of them after a few
        # hours — just to find the end.
        name = f"{DESCENDING_BASE - seq:09d}.txt"
        target = self._banked(f"clog/{name}")
        if not target:
            self._bank_urls(self._series_seq + 1, seq)
            target = self._banked(f"clog/{name}")
        if not target:
            signed = _post(
                f"{self.base_url}/track/runs/{self.run_id}/urls",
                {"project": self.project, "console_chunk": name}, self.api_key,
            )
            target = (signed.get("console_chunks") or {}).get(name)
        if not target:
            # An older server that does not know about chunks: fall back to the
            # single rewritten object so logging still works.
            target = self._urls.get("console")
            if target:
                _put(target, self._console.getvalue().encode("utf-8", "replace"), "text/plain")
            return
        _put(target, text.encode("utf-8", "replace"), "text/plain")

    def _put_series(self, name: str, body: bytes) -> None:
        target = self._banked(f"series/{name}")
        if not target:
            self._bank_urls(self._series_seq, self._console_seq + 1)
            target = self._banked(f"series/{name}")
        if not target:                       # old server without batch support
            signed = _post(
                f"{self.base_url}/track/runs/{self.run_id}/urls",
                {"project": self.project, "series": [name]}, self.api_key,
            )
            target = (signed.get("series") or {}).get(name)
        if target:
            _put(target, body, "application/json")

    def _upload_media(self, pending: list[tuple[str, bytes, str]]) -> None:
        for start in range(0, len(pending), MEDIA_BATCH):
            chunk = pending[start:start + MEDIA_BATCH]
            signed = _post(
                f"{self.base_url}/track/runs/{self.run_id}/urls",
                {"project": self.project, "media": [name for name, _, _ in chunk]},
                self.api_key,
            )
            urls = signed.get("media", {})
            self._put_many([(urls.get(name), data, _content_type(ext))
                            for name, data, ext in chunk])

    def _put_many(self, items: list[tuple[str | None, bytes, str]]) -> None:
        """Upload a set of objects concurrently, surfacing the first failure.

        Each PUT already retries transient errors on its own; if one still
        fails the caller requeues the whole batch, which is safe because these
        are idempotent writes to fixed keys.
        """
        jobs = [(url, data, ctype) for url, data, ctype in items if url]
        if not jobs:
            return
        if len(jobs) == 1:
            _put(*jobs[0])
            return
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(UPLOAD_WORKERS, len(jobs))) as pool:
            for future in [pool.submit(_put, *job) for job in jobs]:
                future.result()

    def _upload_histograms(self, pending: list[tuple[str, bytes]]) -> None:
        for start in range(0, len(pending), MEDIA_BATCH):
            chunk = pending[start:start + MEDIA_BATCH]
            signed = _post(
                f"{self.base_url}/track/runs/{self.run_id}/urls",
                {"project": self.project, "hist": [name for name, _ in chunk]},
                self.api_key,
            )
            urls = signed.get("hist", {})
            self._put_many([(urls.get(name), body, "application/json")
                            for name, body in chunk])

    def _refresh_urls_if_needed(self, force: bool = False) -> None:
        if not force and time.time() < self._urls_expire:
            return
        refreshed = _post(
            f"{self.base_url}/track/runs/{self.run_id}/urls",
            {"project": self.project, "media": []}, self.api_key,
        )
        self._urls = refreshed["urls"]
        self._urls_expire = time.time() + refreshed.get("expires_in", 7200) * URL_REFRESH_MARGIN

    def _put_json(self, url: str, payload: dict) -> None:
        _put(url, json.dumps(payload, default=str).encode(), "application/json")

    def _excepthook(self, exc_type, exc, tb) -> None:
        self._crash = "".join(traceback.format_exception(exc_type, exc, tb))[-4000:]
        self._prev_excepthook(exc_type, exc, tb)

    def _atexit(self) -> None:
        if self._closed or os.getpid() != self._pid:
            return
        # A run that died is never mistaken for a completed one.
        if self._crash:
            self.finish("crashed", self._crash)
        else:
            self.finish("finished")

def _run_name_setter(self, value):
    # Plain attribute assignment is how people rename a run in wandb, so it has
    # to do more than set a field: the name lives in meta.json AND in a key.
    if getattr(self, "_name_ready", False):
        Run.rename(self, value)
    else:
        object.__setattr__(self, "_name", value)


Run.name = property(lambda self: getattr(self, "_name", ""), _run_name_setter)


class _NoOpRun:
    """What init() returns when logging is disabled. Swallows everything."""

    id = run_id = "disabled"
    name = "disabled"
    url = ""

    def __init__(self) -> None:
        self.config: dict[str, Any] = {}
        self.summary: dict[str, Any] = {}
        self._closed = True

    def log(self, *a: Any, **k: Any) -> None: pass
    def log_image(self, *a: Any, **k: Any) -> None: pass
    def log_text(self, *a: Any, **k: Any) -> None: pass
    def watch(self, *a: Any, **k: Any) -> None: pass
    def finish(self, *a: Any, **k: Any) -> None: pass

_active: Run | _NoOpRun | None = None

run: Run | _NoOpRun | None = None

class _LiveDict(dict):
    """A dict that also writes through to whichever run is active.

    `wandb.config` and `wandb.summary` are module-level in W&B but per-run in
    meaning; training code sets them before and after init() indiscriminately.
    """

    def __init__(self, attr: str) -> None:
        super().__init__()
        self._attr = attr

    def _target(self) -> dict | None:
        return getattr(_active, self._attr, None) if _active is not None else None

    def __setitem__(self, key: Any, value: Any) -> None:
        target = self._target()
        if target is not None:
            target[key] = value
            if self._attr == "config" and isinstance(_active, Run):
                _active._meta_dirty = True
        super().__setitem__(key, value)

    def update(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        merged = dict(*args, **kwargs)
        for key, value in merged.items():
            self[key] = value

config = _LiveDict("config")

summary = _LiveDict("summary")

def login(*_args: Any, **_kwargs: Any) -> bool:
    """No-op. There is no account here — auth is RUNLOG_API_KEY."""
    return True

def init(project: str = "default", **kwargs: Any) -> Run | _NoOpRun:
    global _active, run
    if _active is not None and not getattr(_active, "_closed", True):
        _active.finish("finished")
    if _disabled():
        _active = _NoOpRun()
    else:
        _active = Run(project=project, **kwargs)
        # Anything set on wandb.config before init() is real intent; carry it in
        # rather than dropping it on the floor.
        if config:
            _active.config.update(dict(config))
            _active._meta_dirty = True
    run = _active
    summary.clear()
    return _active

def current_run() -> Run | _NoOpRun | None:
    return _active

def _require() -> Run | _NoOpRun:
    if _active is None:
        raise RuntimeError("Call init(...) first")
    return _active

def log(values: dict, step: int | None = None, **kwargs: Any) -> None:
    _require().log(values, step, **kwargs)

def log_image(label: str, image: Any, step: int | None = None) -> None:
    _require().log_image(label, image, step)

def log_text(label: str, text: str, step: int | None = None) -> None:
    _require().log_text(label, text, step)

def watch(model: Any = None, log: str = "all", log_freq: int = 1000, **kwargs: Any) -> None:
    if _active is not None:
        _active.watch(model, log=log, log_freq=log_freq, **kwargs)

def finish(status: str = "finished", **_kwargs: Any) -> None:
    if _active is not None:
        _active.finish(status)


# MARK: - Reading runs back
#
# For scripts and coding agents: fetch what the dashboard shows, as plain JSON.
# Uses a READ-ONLY key (RUNLOG_READ_KEY / read_key in ~/.runlog.json), which is
# separate from the upload key baked into training repos and from the dashboard
# password. It can read runs and nothing else.


def _read(path: str) -> Any:
    key = _setting("read_key")
    if not key:
        raise RuntimeError(
            "No read key. Set RUNLOG_READ_KEY, or add \"read_key\" to "
            f"{CONFIG_PATH}. It is separate from the upload key on purpose."
        )
    url = f"{_setting('url', 'http://127.0.0.1:8000').rstrip('/')}{path}"

    def build():
        return urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})

    return json.loads(_send(build, 60.0, f"GET {path.split('?')[0]}") or b"{}")


def list_projects() -> list[str]:
    """Every project name."""
    return _read("/runlog/api/projects").get("projects", [])


def list_runs(project: str) -> list[dict]:
    """Runs in a project: id, name, state, start time, last activity."""
    return _read(f"/runlog/api/runs?project={urllib.parse.quote(project)}").get("runs", [])


def get_run(project: str, run_id: str) -> dict:
    """One run in full: config, machine, summary, metrics, samples, outcome."""
    return _read(f"/runlog/api/runs/{urllib.parse.quote(project)}/{urllib.parse.quote(run_id)}")


def history(project: str, run_id: str, metric: str | None = None) -> dict:
    """Metric series for a run: {name: {steps, values, wall}}.

    Downsampled by the server for very long runs, the same as the charts.
    """
    metrics = get_run(project, run_id).get("metrics", {})
    return metrics if metric is None else metrics.get(metric, {})


# MARK: - The rest of the wandb surface

# Corners of the wandb API a trainer may reach for that have no effect on
# training. These no-op instead of raising, because losing a 12-hour run to
# `AttributeError: alert` would be a much worse outcome than doing nothing.
_NOOP_CALLS = frozenset({
    "alert", "define_metric", "save", "restore", "unwatch", "mark_preempting",
    "log_artifact", "use_artifact", "log_model", "use_model", "link_model",
    "require", "setup", "teardown", "agent", "sweep", "controller",
    "init_run", "finish_artifact", "join", "termlog", "termwarn", "termerror",
})


def __getattr__(name: str):
    if name == "run":
        return current_run()
    if name in _NOOP_CALLS:
        return lambda *a, **k: None
    # Everything else raises. A blanket no-op made `hasattr(wandb, anything)`
    # true and turned typos and real feature gaps into silently-wrong values
    # rather than a visible error.
    raise AttributeError(f"this wandb replacement does not implement {name!r}")


class Artifact:  # pragma: no cover - accepted, not implemented
    def __init__(self, *a, **k) -> None: pass
    def add_file(self, *a, **k) -> None: pass
    def add_dir(self, *a, **k) -> None: pass


class Api:  # pragma: no cover - accepted, not implemented
    def __init__(self, *a, **k) -> None: pass


# `import wandb.sdk.lib.asyncio_manager` appears in some DataLoader worker
# setups, guarded by try/except. Registering empty modules makes that
# import succeed and its patch loop iterate over nothing, which is exactly the
# intent (never block on a service that does not exist here).
if __name__ == "wandb":
    import types as _types

    for _name in ("wandb.sdk", "wandb.sdk.lib", "wandb.sdk.lib.asyncio_manager"):
        sys.modules.setdefault(_name, _types.ModuleType(_name))
    sys.modules["wandb.sdk"].lib = sys.modules["wandb.sdk.lib"]
    sys.modules["wandb.sdk.lib"].asyncio_manager = sys.modules["wandb.sdk.lib.asyncio_manager"]


# MARK: - Command line: install this file, check config, verify the pipeline

MARKER = "# runlog-managed"
POINTER_MARKER = "# runlog-managed pointer"

# A subdirectory trainer cannot see a wandb.py in the repo root, because
# sys.path[0] is the directory of the script being run. Rather than leave a
# second 1900-line copy there to drift out of sync, deeper directories get a
# pointer that loads the nearest real copy above them.
POINTER = POINTER_MARKER + '''
"""Loads the real wandb.py from a parent directory.

This exists because Python puts the *script's* directory on sys.path, not the
repo root, so a trainer in a subdirectory cannot see the copy above it. Keeping
one real file per repo means there is nothing to drift.
"""
import importlib.util
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_found = None
_dir = _here
while True:
    _parent = os.path.dirname(_dir)
    if _parent == _dir:
        break
    _dir = _parent
    _candidate = os.path.join(_dir, "wandb.py")
    if os.path.exists(_candidate):
        with open(_candidate, "r", encoding="utf-8") as _handle:
            if "# runlog-managed pointer" not in _handle.read(200):
                _found = _candidate
                break

if _found is None:
    raise ImportError(
        "wandb.py pointer in %s found no real wandb.py in any parent directory. "
        "Copy the full file here, or re-run: python wandb.py --link <repo>" % _here
    )

_spec = importlib.util.spec_from_file_location(__name__, _found)
_module = importlib.util.module_from_spec(_spec)
sys.modules[__name__] = _module
_spec.loader.exec_module(_module)
'''

SKIP_DIRS = {".git", "__pycache__", "wandb", ".venv", "venv", "node_modules",
             ".ruff_cache", ".pytest_cache", "checkpoint_archive"}


def dirs_importing_wandb(repo: str) -> list[str]:
    """Directories holding a script that imports wandb.

    `sys.path[0]` is the directory of the script being RUN, so a copy in the
    repo root does nothing for a trainer that lives in a subdirectory — those
    runs would keep going to real W&B while looking like they had moved.
    """
    found = set()
    for root, dirnames, filenames in os.walk(repo):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for filename in filenames:
            if not filename.endswith(".py") or filename == "wandb.py":
                continue
            try:
                with open(os.path.join(root, filename), "r", encoding="utf-8", errors="ignore") as handle:
                    if "import wandb" in handle.read(200_000):
                        found.add(root)
                        break
            except OSError:
                continue
    return sorted(found)


def _ignore(directory: str) -> None:
    path = os.path.join(directory, ".gitignore")
    try:
        existing = open(path, "r", encoding="utf-8").read() if os.path.exists(path) else ""
        if any(line.strip() == "wandb.py" for line in existing.splitlines()):
            return
        with open(path, "a", encoding="utf-8") as handle:
            if existing and not existing.endswith("\n"):
                handle.write("\n")
            handle.write("wandb.py\n")
    except OSError:
        pass


def link(repo: str, force: bool = False) -> int:
    """Copy this file into every directory of `repo` that imports wandb."""
    repo = os.path.abspath(os.path.expanduser(repo))
    source = os.path.abspath(__file__)
    targets = dirs_importing_wandb(repo)
    if not targets:
        print(f"{repo}: nothing imports wandb")
        return 0
    # The shallowest directory gets the real file; everything below it points
    # at that one, so a repo never holds two copies that can diverge.
    root = min(targets, key=len)
    count = 0
    for directory in targets:
        target = os.path.join(directory, "wandb.py")
        if os.path.abspath(target) == source:
            continue
        if os.path.exists(target) and not force:
            with open(target, "r", encoding="utf-8") as handle:
                if MARKER not in handle.read(400):
                    print(f"skip {target}: a wandb.py already exists (use --force)")
                    continue
        if directory == root:
            shutil.copyfile(source, target)
            if EMBEDDED_API_KEY:
                # copyfile does not carry permissions, and this copy holds a
                # credential.
                os.chmod(target, 0o600)
            print(f"linked {target}" + (" (carries the key)" if EMBEDDED_API_KEY else ""))
        else:
            with open(target, "w", encoding="utf-8") as handle:
                handle.write(POINTER)
            print(f"linked {target} (pointer -> {os.path.relpath(root, directory)}/wandb.py)")
        _ignore(directory)
        count += 1
    return count


def unlink(repo: str) -> int:
    repo = os.path.abspath(os.path.expanduser(repo))
    removed = 0
    for root, dirnames, filenames in os.walk(repo):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        if "wandb.py" not in filenames:
            continue
        target = os.path.join(root, "wandb.py")
        with open(target, "r", encoding="utf-8") as handle:
            if MARKER not in handle.read(400):
                print(f"refusing to delete {target}: not ours")
                continue
        os.remove(target)
        print(f"unlinked {target}")
        removed += 1
    if not removed:
        print(f"{repo}: nothing to remove")
    return removed


def write_config(url: str | None, key: str | None, read_key: str | None = None) -> None:
    config = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
                config = json.load(handle)
        except (OSError, ValueError):
            config = {}
        if not isinstance(config, dict):
            config = {}
    if url:
        config["url"] = url.rstrip("/")
    if key:
        config["api_key"] = key
    if read_key:
        config["read_key"] = read_key
    # Written next to the file and moved over it: the keys are bearer tokens,
    # so the file is private from its first byte, and a training process that
    # starts during the write reads the old file or the new one, never half.
    directory = os.path.dirname(os.path.abspath(CONFIG_PATH))
    os.makedirs(directory, exist_ok=True)
    partial = f"{CONFIG_PATH}.{os.getpid()}.tmp"
    try:
        descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
            handle.write("\n")
        os.chmod(partial, 0o600)
        os.replace(partial, CONFIG_PATH)
    finally:
        if os.path.exists(partial):
            os.remove(partial)
    print(f"wrote {CONFIG_PATH} (mode 600)")


# MARK: - configure
#
# `runlog configure`, after `aws configure`: three questions, Enter keeps what
# is there, the answers go to ~/.runlog.json. A key shows as stars and is never
# printed in full.

CONFIGURE_HELP = """\
runlog configure         ask for the server address and the keys, save them in {path}
runlog configure list    show which address and keys this machine uses, and where each comes from

Enter keeps the value in brackets. A key shows as one star per character
while it is typed or pasted, never as itself. Afterwards the server is asked
once whether it takes the keys; add --no-check to skip that. For scripts there
are flags instead of questions: --url URL, --key-stdin, --read-key KEY.
"""

# A line copied whole from notes or a shell profile is taken for its value.
_PASTED_LINE = re.compile(
    r"^\s*(?:export\s+|set\s+|setx\s+|\$env:)?(?:RUNLOG|TRACKER)_[A-Za-z_]+\s*(?:=|\s)\s*(.+?)\s*;?\s*$")


def _answer(text: str) -> str:
    """What a typed or pasted answer means.

    Keys get pasted, and what is on the clipboard is often the whole line they
    were stored in: `$env:RUNLOG_API_KEY = "..."`, `export RUNLOG_API_KEY=...`,
    or the value still in its quotes.
    """
    text = text.strip()
    line = _PASTED_LINE.match(text)
    if line:
        text = line.group(1).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def _server_address(text: str) -> str:
    """The server's address from what was entered.

    The address at hand is usually the dashboard's, copied from the browser
    (https://host/runlog#project); the client needs only scheme and host.
    """
    url = _answer(text)
    if not url:
        return ""
    if "://" not in url:
        local = re.match(r"(localhost|127\.|0\.0\.0\.0|\[::1\])", url)
        url = ("http://" if local else "https://") + url
    parts = urllib.parse.urlsplit(url)
    path = re.sub(r"/(?:runlog|log|track)(?:/.*)?$", "", parts.path.rstrip("/"))
    return f"{parts.scheme}://{parts.netloc}{path}"


def _masked(value: str) -> str:
    """A secret as `aws configure` shows one: stars and its last four characters."""
    if not value:
        return "None"
    return "*" * 16 + (value[-4:] if len(value) >= 16 else "")


def _skip_escape_sequence(fd: int, select) -> None:
    """Swallow what follows an Esc byte: an arrow key or a paste marker is a
    sequence the terminal sent, not something that was typed into the key."""
    def more() -> bool:
        return bool(select.select([fd], [], [], 0.05)[0])
    if not more():
        return                               # a lone Esc
    if os.read(fd, 1) not in (b"[", b"O"):
        return
    while more():
        byte = os.read(fd, 1)
        if not byte or 0x40 <= byte[0] <= 0x7E:
            return


def _read_masked_windows(prompt: str) -> str:
    import msvcrt
    for ch in prompt:
        msvcrt.putwch(ch)
    chars: list[str] = []
    while True:
        ch = msvcrt.getwch()
        if ch in "\r\n":
            break
        if ch == "\x03":
            msvcrt.putwch("\r"); msvcrt.putwch("\n")
            raise KeyboardInterrupt
        if ch in ("\x00", "\xe0"):          # an arrow or function key: two codes
            msvcrt.getwch()
            continue
        if ch == "\x08":
            if chars:
                chars.pop()
                for erase in "\b \b":
                    msvcrt.putwch(erase)
            continue
        if ch < " ":
            continue
        chars.append(ch)
        msvcrt.putwch("*")
    msvcrt.putwch("\r"); msvcrt.putwch("\n")
    return "".join(chars)


def _read_masked(prompt: str) -> str:
    """Read a secret from the terminal, showing a star for each character.

    A prompt that shows nothing makes a paste look like it did not happen.
    Backspace takes a star away and Ctrl-U all of them; the key itself is never
    on screen.
    """
    if os.name == "nt":
        return _read_masked_windows(prompt)
    import codecs
    import select
    import termios
    import tty
    try:
        fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
        own = True
    except OSError:
        fd, own = sys.stdin.fileno(), False
    out = fd if own else sys.stderr.fileno()
    try:
        saved = termios.tcgetattr(fd)
    except termios.error:
        if own:
            os.close(fd)
        import getpass
        return getpass.getpass(prompt)       # not a terminal we can drive: no echo at all
    chars: list[str] = []
    decode = codecs.getincrementaldecoder("utf-8")("ignore").decode
    try:
        tty.setcbreak(fd)                    # one key at a time, unechoed; Ctrl-C still interrupts
        os.write(out, prompt.encode())
        while True:
            data = os.read(fd, 1)
            if not data:
                if chars:
                    break
                raise EOFError
            ch = decode(data)
            if not ch:
                continue                     # in the middle of a multi-byte character
            if ch in "\r\n":
                break
            if ch == "\x04":                 # Ctrl-D
                if chars:
                    break
                raise EOFError
            if ch in "\x7f\x08":
                if chars:
                    chars.pop()
                    os.write(out, b"\b \b")
            elif ch in "\x15\x17":           # Ctrl-U, Ctrl-W: start over
                os.write(out, b"\b \b" * len(chars))
                chars.clear()
            elif ch == "\x1b":
                _skip_escape_sequence(fd, select)
            elif ch >= " ":
                chars.append(ch)
                os.write(out, b"*")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        os.write(out, b"\n")
        if own:
            os.close(fd)
    return "".join(chars)


def _ask(label: str, shown: str, secret: bool) -> str:
    prompt = f"{label} [{shown}]: "
    if sys.stdin.isatty():
        if secret:
            sys.stdout.flush()
            return _read_masked(prompt)
        return input(prompt)
    # Piped in (a setup script, a test): one answer per line, nothing to hide.
    sys.stdout.write(prompt)
    sys.stdout.flush()
    line = sys.stdin.readline()
    sys.stdout.write("\n")
    return line


def _key_check(url: str, key: str, read: bool) -> tuple[bool | None, str]:
    """Ask the server whether it takes a key: (yes / no / could not tell, why).

    Nothing is created by asking. For the upload key this is the call that
    starts a run, which only signs upload addresses; a run exists once a client
    uploads to one, and this never does.
    """
    if read:
        request = urllib.request.Request(f"{url}/runlog/api/projects", headers={"Authorization": f"Bearer {key}"})
    else:
        request = urllib.request.Request(
            f"{url}/track/runs", method="POST",
            data=json.dumps({"project": "configure-check", "name": "configure-check"}).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            response.read()
        return True, ""
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return False, "the server refused it"
        if exc.code == 503:
            return None, "the server has no key set up (HTTP 503)"
        return None, f"the server answered HTTP {exc.code}"
    except Exception as exc:
        return None, _describe(exc)


def _report_checks(url: str, key: str, read_key: str) -> bool:
    """Print one line per key; False when the server refused one."""
    host = urllib.parse.urlsplit(url).netloc or url
    fine = True
    for label, value, read in (("API key", key, False), ("Read key", read_key, True)):
        if not value:
            continue
        accepted, why = _key_check(url, value, read)
        if accepted:
            print(f"  {label}: accepted by {host}")
        elif accepted is None:
            print(f"  {label}: could not be checked, {why}")
        else:
            fine = False
            print(f"  {label}: {why}. It is saved as entered ({len(value)} characters): check that the "
                  f"whole key was pasted, and that it is the {'read' if read else 'API'} key.")
    return fine


def _configured() -> list[tuple[str, str, str]]:
    """(name, value, where it comes from) for each setting, in the order a run resolves them."""
    stored = _load_config()
    rows = []
    for name in ("url", "api_key", "read_key"):
        variable = f"RUNLOG_{name.upper()}"
        if os.environ.get(variable):
            rows.append((name, os.environ[variable], f"environment variable {variable}"))
        elif stored.get(name):
            rows.append((name, str(stored[name]), CONFIG_PATH))
        elif _EMBEDDED.get(name):
            rows.append((name, _EMBEDDED[name], f"built into {os.path.abspath(__file__)}"))
        else:
            rows.append((name, "", ""))
    return rows


def _shadowed_by() -> str:
    """The file `import wandb` loads on this machine when that is not this
    client, else "". Installed next to the real wandb package, this module
    loses: Python takes the package, and training would log there instead.
    """
    try:
        import importlib.util
        spec = importlib.util.find_spec("wandb")
        origin = spec.origin if spec else None
    except Exception:
        return ""
    if not origin or os.path.realpath(origin) == os.path.realpath(__file__):
        return ""
    try:
        with open(origin, "r", encoding="utf-8", errors="replace") as handle:
            if MARKER in handle.read(400):
                return ""              # another copy of this client, or a pointer to one
    except OSError:
        pass
    return origin


def _warn_if_shadowed() -> None:
    other = _shadowed_by()
    if other:
        print(f"Note: `import wandb` loads {other} here, not this client, so training would not log to "
              "Runlog. If that is the real wandb package, remove it from this environment (pip uninstall wandb).")


def configure(argv: list[str] | None = None) -> int:
    argv = list(argv or [])
    if any(word in ("-h", "--help", "help") for word in argv):
        print(CONFIGURE_HELP.format(path=CONFIG_PATH), end="")
        return 0
    check = "--no-check" not in argv
    words = [word for word in argv if word != "--no-check"]
    if words == ["list"]:
        for name, value, source in _configured():
            shown = value if name == "url" else _masked(value)
            print(f"  {name:<9} {shown if value else 'None':<34} {source}".rstrip())
        _warn_if_shadowed()
        return 0
    if words:
        print(f"runlog configure: unknown argument {words[0]!r}\n", file=sys.stderr)
        print(CONFIGURE_HELP.format(path=CONFIG_PATH), end="", file=sys.stderr)
        return 2

    stored = _load_config()
    url_now = str(stored.get("url") or EMBEDDED_URL or "")
    key_now = str(stored.get("api_key") or "")
    read_now = str(stored.get("read_key") or "")
    try:
        url = _server_address(_ask("Runlog URL", url_now or "None", secret=False))
        key = _answer(_ask("API key", _masked(key_now), secret=True))
        read_key = _answer(_ask("Read key, optional", _masked(read_now), secret=True))
    except (KeyboardInterrupt, EOFError):
        print("\nNothing was changed.")
        return 130
    for label, value in (("Runlog URL", url), ("API key", key), ("Read key", read_key)):
        if any(ch.isspace() for ch in value):
            print(f"{label}: there is a space or a line break in what was entered. Nothing was changed.",
                  file=sys.stderr)
            return 2
    if url and not urllib.parse.urlsplit(url).netloc:
        print("Runlog URL: that is not an address (expected something like https://example.com). "
              "Nothing was changed.", file=sys.stderr)
        return 2

    # The address is stored even when Enter kept the built-in one, so this
    # machine no longer depends on which copy of the client it runs.
    url = url or url_now
    if key or read_key or url != str(stored.get("url") or ""):
        write_config(url, key, read_key)
    else:
        print(f"Nothing was changed in {CONFIG_PATH}.")
    key, read_key = key or key_now, read_key or read_now
    if os.environ.get("RUNLOG_URL") or os.environ.get("RUNLOG_API_KEY"):
        print("Note: RUNLOG_URL / RUNLOG_API_KEY are set in this shell and win over the file "
              "(runlog configure list shows which is used).")
    if not key:
        print("No API key yet: runs cannot be logged from this machine until one is entered.")
    _warn_if_shadowed()
    if check and url and (key or read_key):
        if not _report_checks(url, key, read_key):
            return 1
    return 0


def embed_key(strip: bool = False) -> None:
    """Write the configured URL and key into this file, or clear them.

    For rented boxes and containers, where there is no ~/.runlog.json and
    exporting two variables into every shell is friction. The value is read
    from the local config, so it never has to be typed or pasted anywhere.
    """
    path = os.path.abspath(__file__)
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()

    url = key = ""
    if not strip:
        if not os.path.exists(CONFIG_PATH):
            print(f"no {CONFIG_PATH}; run --url/--key first")
            return
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        url, key = config.get("url", ""), config.get("api_key", "")
        if not key:
            print(f"no api_key in {CONFIG_PATH}")
            return

    text = re.sub(r'^EMBEDDED_URL = ".*"$', f'EMBEDDED_URL = "{url}"', text, count=1, flags=re.M)
    text = re.sub(r'^EMBEDDED_API_KEY = ".*"$', f'EMBEDDED_API_KEY = "{key}"', text, count=1, flags=re.M)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.chmod(path, 0o600 if key else 0o644)
    if strip:
        print(f"cleared embedded credentials from {path}")
    else:
        print(f"embedded url + {len(key)}-char key into {path} (mode 600)")
        print("This file now carries a write credential. Re-run --link to push it")
        print("to the repos, and keep it out of git.")


def status(repos: list[str]) -> None:
    print(f"config: {CONFIG_PATH} {'present' if os.path.exists(CONFIG_PATH) else 'MISSING'}")
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
                config = json.load(handle)
            print(f"  url:     {config.get('url', '(unset)')}")
            key = config.get("api_key", "")
            print(f"  api_key: {'set (' + str(len(key)) + ' chars)' if key else '(unset)'}")
            rk = config.get("read_key", "")
            print(f"  read_key: {'set (' + str(len(rk)) + ' chars)' if rk else '(unset)'}")
            print(f"  mode:    {oct(os.stat(CONFIG_PATH).st_mode & 0o777)}")
        except (OSError, ValueError) as exc:
            print(f"  unreadable: {exc}")
    print(f"this file: {os.path.abspath(__file__)}")
    print(f"  embedded: {'url + key baked in' if EMBEDDED_API_KEY else 'none'}")
    for repo in repos:
        root = os.path.abspath(os.path.expanduser(repo))
        wanted = dirs_importing_wandb(root)
        linked = [d for d in wanted if os.path.exists(os.path.join(d, "wandb.py"))]
        print(f"  {repo}: {len(linked)}/{len(wanted)} directories linked")
        for directory in wanted:
            if directory not in linked:
                print(f"    not linked: {os.path.relpath(directory, root)}")


def smoke(project: str = "smoke") -> int:
    """Prove the whole pipeline works, in ~20 seconds.

    Exercises every path a real run uses — scalars, a per-device metric, an
    image, an HTML table, machine info, and (with torch) parameter and gradient
    histograms through the same backward hooks watch() installs.
    """
    try:
        active = Run(project=project, name="smoke-test",
                     config={"purpose": "pipeline check"})
    except Exception as exc:
        print(f"FAILED to start a run: {exc}")
        print("Check RUNLOG_URL / RUNLOG_API_KEY, or run: python wandb.py --status")
        return 1

    model = optimizer = None
    try:
        import torch
        import torch.nn as nn

        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 4))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        active.watch(model, log="all", log_freq=5)
        print("torch found: histograms will be exercised too")
    except ImportError:
        print("torch not installed: scalars and media only")

    for step in range(40):
        if model is not None:
            loss_t = ((model(torch.randn(16, 32)) - torch.randn(16, 4)) ** 2).mean()
            loss_t.backward()
            optimizer.step()
            # Mirrors the real trainers: zero BEFORE logging, which is what
            # makes hook-based gradient capture necessary.
            optimizer.zero_grad(set_to_none=True)
            loss = float(loss_t.detach())
        else:
            loss = 2.0 / (step + 1) + 0.05 * math.sin(step)
        values = {"loss": loss, "train/lr": 1e-2}
        for device in range(2):
            values[f"system/fake_gpu{device}_util_pct"] = 70 + 10 * math.sin(step / 5 + device)
        active.log(values, step=step)
        if step % 20 == 0:
            active.log({"notes/table": Html(
                f"<table><tr><th>step</th><th>loss</th></tr>"
                f"<tr><td>{step}</td><td>{loss:.4f}</td></tr></table>")}, step=step)
            try:
                import numpy as np
                active.log({"samples": [Image(np.random.rand(3, 32, 32), caption=f"step {step}")]},
                           step=step)
            except ImportError:
                pass

    time.sleep(2)      # let the background thread take a telemetry sample
    active.finish()
    print(f"\n  project : {active.project}")
    print(f"  run     : {active.run_id}")
    print(f"  machine : {active.system.get('cpu', '?')}")
    if active.system.get("gpus_on_host"):
        print(f"  gpus    : {active.system['gpus_on_host']} on host")
    print(f"  open    : {active.base_url}/runlog")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["configure"]:
        return configure(argv[1:])
    # Installed with pip this file is also the `runlog` command.
    started_as = os.path.basename(sys.argv[0] or "")
    parser = argparse.ArgumentParser(
        prog="python wandb.py" if started_as.endswith(".py") or not started_as else started_as,
        epilog="`configure` asks for the address and the keys instead (see `configure --help`).")
    parser.add_argument("--url", help="tracker base URL")
    parser.add_argument("--key", help="TRACKER_API_KEY value")
    parser.add_argument("--key-stdin", action="store_true",
                        help="read the key from stdin, keeping it out of shell history")
    parser.add_argument("--link", nargs="+", metavar="REPO", help="copy this file into these repos")
    parser.add_argument("--unlink", nargs="+", metavar="REPO", help="remove copies of this file")
    parser.add_argument("--status", nargs="*", metavar="REPO", help="show config and link state")
    parser.add_argument("--smoke", nargs="?", const="smoke", metavar="PROJECT",
                        help="run an end-to-end pipeline check")
    parser.add_argument("--read-key", help="store a read-only key for --list/--show")
    parser.add_argument("--list", nargs="?", const="", metavar="PROJECT",
                        help="list projects, or runs in a project, as JSON")
    parser.add_argument("--show", nargs=2, metavar=("PROJECT", "RUN"),
                        help="print one run as JSON")
    parser.add_argument("--force", action="store_true", help="overwrite an existing wandb.py")
    parser.add_argument("--embed-key", action="store_true",
                        help="bake the configured url+key into this file (for remote boxes)")
    parser.add_argument("--strip-key", action="store_true",
                        help="remove baked-in credentials from this file")
    args = parser.parse_args(argv)

    key = sys.stdin.read().strip() if args.key_stdin else args.key
    did = False
    if args.url or key or args.read_key:
        write_config(args.url, key, args.read_key); did = True
    if args.list is not None:
        print(json.dumps(list_runs(args.list) if args.list else list_projects(), indent=2))
        return 0
    if args.show:
        run = get_run(args.show[0], args.show[1])
        # Series are large and rarely what you want printed; summarise them.
        run["metrics"] = {k: {"points": len(v.get("steps") or []),
                              "last": (v.get("values") or [None])[-1]}
                          for k, v in (run.get("metrics") or {}).items()}
        print(json.dumps(run, indent=2))
        return 0
    for repo in args.link or []:
        link(repo, args.force); did = True
    for repo in args.unlink or []:
        unlink(repo); did = True
    if args.embed_key or args.strip_key:
        embed_key(strip=args.strip_key); did = True
    if args.status is not None:
        status(args.status or []); did = True
    if args.smoke:
        return smoke(args.smoke)
    if not did:
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
