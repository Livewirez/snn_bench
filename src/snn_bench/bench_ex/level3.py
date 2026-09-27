from __future__ import annotations

import math
import time
import threading
import copy
import psutil
import pynvml
 
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

from typing import Dict, List, Optional, Tuple, Union
from . import sram_access_energy, profile_topology
from .types import LayerSpec, EnergyModel, SYNAPTIC_TYPES, NEURON_TYPES, INT32_MODEL,BYTES_PER_WORD

from spikingjelly.activation_based import neuron, functional
from ..constants import _NVML, _HAS_NVML, _HAS_PSUTIL
 

class _PowerSampler(threading.Thread):
    def __init__(self, period_s: float = 0.02):
        super().__init__(daemon=True)
        self.period = period_s
        self.samples: List[float] = []
        self._stop = threading.Event()
 
    def run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(pynvml.nvmlDeviceGetPowerUsage(_NVML) / 1000.0)
            except Exception:
                pass
            time.sleep(self.period)
 
    def stop(self):
        self._stop.set(); self.join(timeout=1.0)
 
 
def _energy_J() -> Optional[float]:
    if not _HAS_NVML:
        return None
    try:
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(_NVML) / 1000.0
    except Exception:
        return None
 
 
def measure_idle_power(settle_s: float = 5.0, window_s: float = 5.0) -> float:
    """GPU is clocked down, then measured."""
    if not _HAS_NVML:
        return 0.0
    if torch.cuda.is_available():
        torch.cuda.synchronize(); torch.cuda.empty_cache()
    time.sleep(settle_s)
    e0 = _energy_J(); t0 = time.perf_counter()
    if e0 is not None:
        time.sleep(window_s)
        return (_energy_J() - e0) / (time.perf_counter() - t0)
    s = _PowerSampler(); s.start(); time.sleep(window_s); s.stop()
    return float(np.mean(s.samples)) if s.samples else 0.0
 
 
def measure_empirical(
    model: nn.Module, loader: Dataset, T: int, device: Union[torch.device, str],
    min_duration_s: float = 20.0,
    warmup_s: float = 3.0,
    repeats: int = 3,
    spiking: bool = True
) -> Dict:
    """
    Loops the dataset until min_duration_s elapses, so the measurement is long
    compared to NVML's sampling period.
    """
    model.eval()
    x0, _ = next(iter(loader))
    x0 = x0.to(device)
 
    t_end = time.perf_counter() + warmup_s
    with torch.no_grad():
        while time.perf_counter() < t_end:
            if spiking:
                functional.reset_net(model)
                for _ in range(T):
                    model(x0)
            else:
                model(x0)
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
 
    idle_W = measure_idle_power()
    trials = []
 
    for _ in range(repeats):
        if _HAS_PSUTIL:
            psutil.cpu_percent(interval=None)
        e0 = _energy_J()
        sampler = None
        if _HAS_NVML and e0 is None:
            sampler = _PowerSampler(); sampler.start()
 
        t0 = time.perf_counter(); seen = 0
        with torch.no_grad():
            while time.perf_counter() - t0 < min_duration_s:
                for x, _ in loader:
                    x = x.to(device, non_blocking=True)
                    if spiking:
                        functional.reset_net(model)
                        for _ in range(T):
                            model(x)
                    else:
                        model(x)
                    seen += x.size(0)
                    if time.perf_counter() - t0 >= min_duration_s:
                        break
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
 
        e1 = _energy_J()
        if sampler is not None:
            sampler.stop()
 
        wall = t1 - t0
        if e0 is not None and e1 is not None:
            gpu_J, src = e1 - e0, "nvml_energy_counter"
        elif sampler is not None and sampler.samples:
            gpu_J, src = float(np.mean(sampler.samples)) * wall, "nvml_power_integration"
        else:
            gpu_J, src = float("nan"), "unavailable"
 
        trials.append({
            "wall": wall, "seen": seen, "gpu_J": gpu_J,
            "marginal_J": gpu_J - idle_W * wall,
            "ms_per_sample": 1000.0 * wall / max(seen, 1),
            "power_W": gpu_J / wall if wall > 0 else float("nan"),
            "src": src,
        })
 
    med = lambda k: float(np.median([t[k] for t in trials]))
    seen = int(np.median([t["seen"] for t in trials]))
    return {
        "T": T, "n_samples_per_trial": seen, "repeats": repeats,
        "idle_power_W": idle_W,
        "wall_time_s": med("wall"),
        "time_per_sample_ms": med("ms_per_sample"),
        "gpu_energy_per_sample_J": med("gpu_J") / max(seen, 1),
        "gpu_energy_marginal_per_sample_J": med("marginal_J") / max(seen, 1),
        "avg_gpu_power_W": med("power_W"),
        "power_spread_W": float(np.ptp([t["power_W"] for t in trials])),
        "energy_source": trials[0]["src"],
        "cpu_util_pct": psutil.cpu_percent(interval=None) if _HAS_PSUTIL else float("nan"),
        "gpu_peak_mem_MB": (
            torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
        ),
    }