import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from spikingjelly.activation_based import monitor, neuron, functional
from collections import OrderedDict
from typing import Dict, Optional

import time
from typing import Dict, Optional, List
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil

from spikingjelly.activation_based import monitor, neuron, functional

from snn_bench.constants import (
    E_MAC, E_ACC, E_SRAM_READ,
    E_SRAM_WRITE, E_ADDRESS, _NVML_HANDLE,
    _HAS_NVML
)


# ------------------------------------------------------------------
# Level 1 – Handle Computational activity
# ------------------------------------------------------------------
class Level1:
    @staticmethod
    def count_spikes(snn: nn.Module, loader: DataLoader, T: int, device, max_batches: Optional[int] = None) -> Dict[str, float]:
        """Spikes per sample, per IF layer."""
        snn.eval()
        spike_mon = monitor.OutputMonitor(snn, neuron.IFNode)
        per_layer = {name: 0.0 for name in spike_mon.monitored_layers}
        n = 0

        with torch.no_grad():
            for b_idx, (x, _) in enumerate(loader):
                if max_batches is not None and b_idx >= max_batches:
                    break
                x = x.to(device)
                functional.reset_net(snn)
                spike_mon.clear_recorded_data()
                for _ in range(T):
                    snn(x)
                for name in spike_mon.monitored_layers:
                    per_layer[name] += sum(r.sum().item() for r in spike_mon[name])
                n += x.size(0)

        spike_mon.remove_hooks()
        if n == 0:
            return per_layer
        return {k: v / n for k, v in per_layer.items()}

    @staticmethod
    def get_neuron_counts(snn: nn.Module, example_input: torch.Tensor) -> Dict[str, int]:
        """Number of neurons in every IFNode."""
        snn.eval()
        counts = {}
        hooks = []

        def make_hook(name):
            def hook(module, inp, out):
                counts[name] = out[0].numel() if isinstance(out, (tuple, list)) else out.numel()
            return hook

        for name, mod in snn.named_modules():
            if isinstance(mod, neuron.IFNode):
                hooks.append(mod.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            functional.reset_net(snn)
            _ = snn(example_input.to(next(snn.parameters()).device))

        for h in hooks:
            h.remove()
        return counts

    @staticmethod
    def metrics(per_layer_spikes: Dict[str, float], neuron_counts: Dict[str, int], T: int) -> Dict:
        total_spikes = sum(per_layer_spikes.values())
        total_neurons = sum(neuron_counts.values()) if neuron_counts else 0
        spikes_per_neuron = total_spikes / total_neurons if total_neurons else 0.0
        spike_rate = total_spikes / T if T else 0.0
        return {
            "total_spikes_per_sample": total_spikes,
            "spikes_per_neuron_per_inference": spikes_per_neuron,
            "spike_rate": spike_rate,
            "timesteps": T,
            "per_layer_spikes": per_layer_spikes,
            "neuron_counts": neuron_counts,
            "total_neurons": total_neurons,
        }

# ------------------------------------------------------------------
# Level 2 – Analytical energy
# ------------------------------------------------------------------
class Level2:
    @staticmethod
    def estimate_ops_from_spikes(
        per_layer_spikes: Dict[str, float],
        fan_ins: Optional[Dict[str, int]] = None
    ) -> Dict[str, float]:
        total_spikes = sum(per_layer_spikes.values())
        accs = total_spikes
        addressing = total_spikes
        
        if fan_ins:
            accs = sum(per_layer_spikes.get(n, 0.0) * fan_ins.get(n, 1)
                        for n in set(per_layer_spikes) | set(fan_ins))
            
        return {"accs": accs, "addressing_ops": addressing, "macs": 0.0}
    
    @classmethod
    def analytical_energy(
        macs: float = 0.0,
        accs: float = 0.0,
        mem_reads: float = 0.0,
        mem_writes: float = 0.0,
        addressing_ops: float = 0.0
    ) -> Dict[str, float]:
        E_ops  = macs * E_MAC + accs * E_ACC
        E_mem  = mem_reads * E_SRAM_READ + mem_writes * E_SRAM_WRITE
        E_addr = addressing_ops * E_ADDRESS
        return {
            "E_ops": E_ops,
            "E_mem": E_mem,
            "E_addr": E_addr,
            "E_total": E_ops + E_mem + E_addr,
            "macs": macs,
            "accs": accs,
            "mem_reads": mem_reads,
            "mem_writes": mem_writes,
            "addressing_ops": addressing_ops,
        }
    
# ------------------------------------------------------------------
# Level 3 – Empirical hardware measurement
# ------------------------------------------------------------------
class Level3:
    @staticmethod
    def measure_empirical(snn: nn.Module, loader: DataLoader, T: int, device, n_samples: int = 1000) -> Dict:
        snn.eval()
        functional.reset_net(snn)

        # warm-up
        x0, _ = next(iter(loader))
        x0 = x0.to(device)
        for _ in range(5):
            functional.reset_net(snn)
            for _ in range(T):
                snn(x0)

        # start
        if _HAS_NVML:
            start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(_NVML_HANDLE)  # mJ
        else:
            start_energy = 0
        start_time = time.perf_counter()
        cpu_start  = psutil.cpu_percent(interval=None)
        mem_start  = psutil.Process().memory_info().rss

        seen = 0
        with torch.no_grad():
            for x, _ in loader:
                if seen >= n_samples:
                    break
                x = x.to(device)
                functional.reset_net(snn)
                for _ in range(T):
                    snn(x)
                seen += x.size(0)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        if _HAS_NVML:
            end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(_NVML_HANDLE)
        else:
            end_energy = 0
        cpu_end = psutil.cpu_percent(interval=None)
        mem_end = psutil.Process().memory_info().rss

        gpu_energy_J = (end_energy - start_energy) / 1000.0   # mJ → J
        wall_s = end_time - start_time

        return {
            "gpu_energy_J": gpu_energy_J,
            "wall_time_s": wall_s,
            "time_per_sample_ms": 1000.0 * wall_s / max(seen, 1),
            "avg_gpu_power_W": gpu_energy_J / wall_s if wall_s > 0 else 0.0,
            "cpu_util_%": (cpu_start + cpu_end) / 2,
            "ram_rss_MB": mem_end / (1024 ** 2),
            "ram_delta_MB": (mem_end - mem_start) / (1024 ** 2),
            "n_samples": seen,
            "T": T,
        }

class Plotter:
    @staticmethod
    def plot_all(df: pd.DataFrame, title_prefix: str = ""):
        """Produces the five recommended figures."""
        Ts = df["T"].values

        # 1. Total spikes vs T
        plt.figure(figsize=(7, 4))
        plt.plot(Ts, df["total_spikes"], "o-", lw=2, ms=7)
        plt.xlabel("Timesteps (T)")
        plt.ylabel("Total spikes per sample")
        plt.title(f"{title_prefix}Spike Activity vs Timesteps")
        plt.xscale("log", base=2)
        plt.xticks(Ts, Ts)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 2. Spikes/neuron + 1.72 threshold
        plt.figure(figsize=(7, 4))
        plt.plot(Ts, df["spikes_per_neuron"], "s-", color="C1", lw=2, ms=7,
                    label="spikes / neuron / inference")
        plt.axhline(1.72, color="red", ls="--", lw=1.5,
                    label="Davidson & Furber (1.72)")
        plt.xlabel("Timesteps (T)")
        plt.ylabel("Spikes per neuron per inference")
        plt.title(f"{title_prefix}Spike Rate vs Timesteps")
        plt.legend()
        plt.xscale("log", base=2)
        plt.xticks(Ts, Ts)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 3. Analytical energy (log)
        plt.figure(figsize=(7, 4))
        plt.plot(Ts, df["E_analytical"], "o-", color="C2", lw=2, ms=7)
        plt.xlabel("Timesteps (T)")
        plt.ylabel("Analytical energy (J)")
        plt.title(f"{title_prefix}Analytical Energy vs Timesteps")
        plt.xscale("log", base=2)
        plt.yscale("log")
        plt.xticks(Ts, Ts)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 4. Energy breakdown (stacked)
        plt.figure(figsize=(8, 4.5))
        plt.stackplot(Ts,
                        df["E_ops"], df["E_mem"], df["E_addr"],
                        labels=["E_ops", "E_mem", "E_addr"], alpha=0.85)
        plt.xlabel("Timesteps (T)")
        plt.ylabel("Energy (J)")
        plt.title(f"{title_prefix}Analytical Energy Breakdown")
        plt.legend(loc="upper left")
        plt.xscale("log", base=2)
        plt.xticks(Ts, Ts)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 5. Analytical vs Empirical (Level 2 vs Level 3)
        if "E_gpu" in df.columns and df["E_gpu"].sum() > 0:
            fig, ax1 = plt.subplots(figsize=(8, 4.5))
            ax1.plot(Ts, df["E_analytical"], "o-", color="C2", lw=2,
                        label="E_analytical")
            ax1.set_xlabel("Timesteps (T)")
            ax1.set_ylabel("Analytical energy (J)", color="C2")
            ax1.set_xscale("log", base=2)
            ax1.set_yscale("log")
            ax1.tick_params(axis="y", labelcolor="C2")
            ax1.set_xticks(Ts)
            ax1.set_xticklabels(Ts)

            ax2 = ax1.twinx()
            ax2.plot(Ts, df["E_gpu"], "s--", color="C3", lw=2, label="E_gpu")
            ax2.set_ylabel("Empirical GPU energy (J)", color="C3")
            ax2.tick_params(axis="y", labelcolor="C3")

            fig.tight_layout()
            plt.title(f"{title_prefix}Analytical vs Empirical Energy")
            plt.show()

            # also latency
            plt.figure(figsize=(7, 4))
            plt.plot(Ts, df["time_per_sample_ms"], "D-", color="C4", lw=2, ms=7)
            plt.xlabel("Timesteps (T)")
            plt.ylabel("Time per sample (ms)")
            plt.title(f"{title_prefix}Empirical Latency vs Timesteps")
            plt.xscale("log", base=2)
            plt.xticks(Ts, Ts)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

class Benchmark:
    """Three-level energy / activity profiler for SpikingJelly models."""
    
    
    # ------------------------------------------------------------------
    # Level 1 – Handle Computational activity
    # ------------------------------------------------------------------
    @staticmethod
    def count_spikes(snn: nn.Module, loader: DataLoader, T: int, device, max_batches: Optional[int] = None) -> Dict[str, float]:
       return Level1.count_spikes(snn, loader, T, device, max_batches)

    @staticmethod
    def get_neuron_counts(snn, example_input: torch.Tensor) -> Dict[str, int]:
        return Level1.get_neuron_counts(snn, example_input)

    @staticmethod
    def level1_metrics(per_layer_spikes: Dict[str, float], neuron_counts: Dict[str, int], T: int) -> Dict:
        return Level1.metrics(per_layer_spikes, neuron_counts)

    # ------------------------------------------------------------------
    # Level 2 – Analytical energy
    # ------------------------------------------------------------------
    @staticmethod
    def estimate_ops_from_spikes(per_layer_spikes: Dict[str, float], fan_ins: Optional[Dict[str, int]] = None) -> Dict[str, float]:
        return Level2.estimate_ops_from_spikes(per_layer_spikes, fan_ins)

    @staticmethod
    def analytical_energy(
        macs: float = 0.0,
        accs: float = 0.0,
        mem_reads: float = 0.0,
        mem_writes: float = 0.0,
        addressing_ops: float = 0.0
    ) -> Dict[str, float]:
        return Level2.analytical_energy(macs, accs, mem_reads, mem_writes, addressing_ops)

    # ------------------------------------------------------------------
    # Level 3 – Empirical hardware measurement
    # ------------------------------------------------------------------
    @staticmethod
    def measure_empirical(snn: nn.Module, loader: DataLoader, T: int, device, n_samples: int = 1000) -> Dict:
        return Level3.measure_empirical(snn, loader, T, device,  n_samples) 

    # ------------------------------------------------------------------
    # Full sweep (Levels 1 + 2 + 3)
    # ------------------------------------------------------------------
    @classmethod
    def run_sweep(
        cls,
        snn: nn.Module, 
        loader: DataLoader, 
        device: torch.device,
        Ts: List[int] = (1, 2, 4, 8, 16, 32, 64),
        max_batches_l1: int = 20,
        n_samples_l3: int = 1000,
        neuron_counts: Optional[Dict[str, int]] = None
    ) -> pd.DataFrame:
        """
        Runs Level 1, 2 and 3 for every T and returns a tidy DataFrame.
        """
        snn = snn.to(device).eval()

        if neuron_counts is None:
            x0, _ = next(iter(loader))
            neuron_counts = cls.get_neuron_counts(snn, x0[:1].to(device))

        rows = []
        for T in Ts:
            print(f"→ T = {T} …")

            # Level 1
            per_layer = cls.count_spikes(snn, loader, T, device,
                                         max_batches=max_batches_l1)
            l1 = cls.level1_metrics(per_layer, neuron_counts, T)

            # Level 2
            ops = cls.estimate_ops_from_spikes(l1["per_layer_spikes"])
            ana = cls.analytical_energy(
                macs=ops["macs"],
                accs=ops["accs"],
                mem_reads=ops["accs"] * 2,               # placeholder
                mem_writes=l1["total_spikes_per_sample"],
                addressing_ops=ops["addressing_ops"],
            )

            # Level 3
            emp = cls.measure_empirical(snn, loader, T, device,
                                        n_samples=n_samples_l3)

            rows.append({
                "T": T,
                # Level 1
                "total_spikes": l1["total_spikes_per_sample"],
                "spikes_per_neuron": l1["spikes_per_neuron_per_inference"],
                "spike_rate": l1["spike_rate"],
                # Level 2
                "E_analytical": ana["E_total"],
                "E_ops": ana["E_ops"],
                "E_mem": ana["E_mem"],
                "E_addr": ana["E_addr"],
                # Level 3
                "E_gpu": emp["gpu_energy_J"],
                "time_per_sample_ms": emp["time_per_sample_ms"],
                "avg_gpu_power_W": emp["avg_gpu_power_W"],
                "cpu_util_%": emp["cpu_util_%"],
                "ram_rss_MB": emp["ram_rss_MB"],
                "ram_delta_MB": emp["ram_delta_MB"],
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # ANN-specific Level 1 + 2  (sine it has no spikes)
    # ------------------------------------------------------------------
    @staticmethod
    def count_macs_ann(model: nn.Module, example_input: torch.Tensor) -> Dict[str, float]:
        """
        Simple MAC / ACC counter for a standard ANN.
        Uses a forward hook on every Conv2d and Linear layer.
        """
        macs = 0.0
        accs = 0.0
        hooks = []

        def conv_hook(module, inp, out):
            # out: [B, C_out, H, W]
            # weight: [C_out, C_in, kH, kW]
            B, C_out, H, W = out.shape
            C_in, kH, kW = module.weight.shape[1:]
            # each output element needs C_in*kH*kW MACs
            nonlocal macs
            macs += B * C_out * H * W * C_in * kH * kW

        def linear_hook(module, inp, out):
            # out: [B, out_features]
            B, out_f = out.shape
            in_f = module.weight.shape[1]
            nonlocal macs
            macs += B * out_f * in_f

        for mod in model.modules():
            if isinstance(mod, nn.Conv2d):
                hooks.append(mod.register_forward_hook(conv_hook))
            elif isinstance(mod, nn.Linear):
                hooks.append(mod.register_forward_hook(linear_hook))

        model.eval()
        with torch.no_grad():
            _ = model(example_input)

        for h in hooks:
            h.remove()

        # For a pure ANN every MAC is treated as one multiply-accumulate 
        return {
            "macs": macs / example_input.size(0),   # per sample
            "accs": 0.0,
            "addressing_ops": 0.0
        }

    @classmethod
    def run_ann(cls, model: nn.Module, loader: DataLoader, device: torch.device, n_samples_l3: int = 1000) -> pd.DataFrame:
        """For a Level 1-3 for a non-spiking ANN (T is meaningless so T=1)."""
        model = model.to(device).eval()
        x0, _ = next(iter(loader))
        example = x0[:1].to(device)

        # Level 1 / 2
        ops = cls.count_macs_ann(model, example)
        # crude memory estimate: weight reads + activation writes
        # Todo: (can be refined later with a real activation-size calculator)
        n_params = sum(p.numel() for p in model.parameters())
        ana = cls.analytical_energy(
            macs=ops["macs"],
            accs=ops["accs"],
            mem_reads=ops["macs"] + n_params,   # rough
            mem_writes=ops["macs"] * 0.5,       # rough
            addressing_ops=0.0,
        )

        # Level 3
        emp = cls.measure_empirical_ann(model, loader, device, n_samples=n_samples_l3)

        row = {
            "T": 1, # ANN has no timesteps
            "total_spikes": 0.0,
            "spikes_per_neuron": 0.0,
            "spike_rate": 0.0,
            "E_analytical": ana["E_total"],
            "E_ops": ana["E_ops"],
            "E_mem": ana["E_mem"],
            "E_addr": ana["E_addr"],
            "E_gpu": emp["gpu_energy_J"],
            "time_per_sample_ms": emp["time_per_sample_ms"],
            "avg_gpu_power_W": emp["avg_gpu_power_W"],
            "cpu_util_%": emp["cpu_util_%"],
            "ram_rss_MB": emp["ram_rss_MB"],
            "ram_delta_MB": emp["ram_delta_MB"],
        }
        return pd.DataFrame([row])

    @staticmethod
    def measure_empirical_ann(model: nn.Module, loader: DataLoader, device: torch.device, n_samples: int = 1000) -> Dict:
        """Same as measure_empirical but without the inner T-loop."""
        model.eval()
        x0, _ = next(iter(loader))
        x0 = x0.to(device)
        for _ in range(5):                          # warm-up
            _ = model(x0)

        if _HAS_NVML:
            start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(_NVML_HANDLE)
        else:
            start_energy = 0
        start_time = time.perf_counter()
        cpu_start  = psutil.cpu_percent(interval=None)
        mem_start  = psutil.Process().memory_info().rss

        seen = 0
        with torch.no_grad():
            for x, _ in loader:
                if seen >= n_samples:
                    break
                x = x.to(device)
                _ = model(x)
                seen += x.size(0)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        if _HAS_NVML:
            end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(_NVML_HANDLE)
        else:
            end_energy = 0
        cpu_end = psutil.cpu_percent(interval=None)
        mem_end = psutil.Process().memory_info().rss

        gpu_energy_J = (end_energy - start_energy) / 1000.0
        wall_s = end_time - start_time
        return {
            "gpu_energy_J": gpu_energy_J,
            "wall_time_s": wall_s,
            "time_per_sample_ms": 1000.0 * wall_s / max(seen, 1),
            "avg_gpu_power_W": gpu_energy_J / wall_s if wall_s > 0 else 0.0,
            "cpu_util_%": (cpu_start + cpu_end) / 2,
            "ram_rss_MB": mem_end / (1024 ** 2),
            "ram_delta_MB": (mem_end - mem_start) / (1024 ** 2),
            "n_samples": seen,
            "T": 1,
        }

    # ------------------------------------------------------------------
    # Direct SNN (your LIF linear model) – works with LIFNode
    # ------------------------------------------------------------------
    @staticmethod
    def count_spikes_direct(snn: nn.Module, loader: DataLoader, T: int, device, encoder=None, max_batches: Optional[int] = None) -> Dict[str, float]:
        """
        Hooks LIFNode and optionally applies a PoissonEncoder.
        """
        snn.eval()
        # monitor both IF and LIF
        spike_mon = monitor.OutputMonitor(snn, (neuron.IFNode, neuron.LIFNode))
        per_layer = {name: 0.0 for name in spike_mon.monitored_layers}
        n = 0

        with torch.no_grad():
            for b_idx, (x, _) in enumerate(loader):
                if max_batches is not None and b_idx >= max_batches:
                    break
                x = x.to(device)
                functional.reset_net(snn)
                spike_mon.clear_recorded_data()
                for _ in range(T):
                    inp = encoder(x) if encoder is not None else x
                    snn(inp)
                for name in spike_mon.monitored_layers:
                    per_layer[name] += sum(r.sum().item() for r in spike_mon[name])
                n += x.size(0)

        spike_mon.remove_hooks()
        if n == 0:
            return per_layer
        return {k: v / n for k, v in per_layer.items()}

    @classmethod
    def run_direct_snn(
        cls, snn: nn.Module, 
        loader: DataLoader, 
        device: torch.device,
        Ts=(1, 2, 4, 8, 16, 32, 64),
        encoder=None,
        max_batches_l1: int = 20,
        n_samples_l3: int = 1000,
        neuron_counts: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Full Level 1-2-3 sweep for a directly-trained SNN."""
        snn = snn.to(device).eval()

        if neuron_counts is None:
            x0, _ = next(iter(loader))
            # temporary monitor just to get neuron counts
            neuron_counts = cls.get_neuron_counts_lif(snn, x0[:1].to(device))

        rows = []
        for T in Ts:
            print(f"[Direct SNN] T = {T} …")
            per_layer = cls.count_spikes_direct(snn, loader, T, device, encoder=encoder, max_batches=max_batches_l1)
            l1 = cls.level1_metrics(per_layer, neuron_counts, T)
            ops = cls.estimate_ops_from_spikes(l1["per_layer_spikes"])
            ana = cls.analytical_energy(
                macs=ops["macs"],
                accs=ops["accs"],
                mem_reads=ops["accs"] * 2,
                mem_writes=l1["total_spikes_per_sample"],
                addressing_ops=ops["addressing_ops"],
            )
            
            
            emp = cls.measure_empirical_direct(snn, loader, T, device, encoder=encoder, n_samples=n_samples_l3)
            rows.append({
                "T": T,
                "total_spikes": l1["total_spikes_per_sample"],
                "spikes_per_neuron": l1["spikes_per_neuron_per_inference"],
                "spike_rate": l1["spike_rate"],
                "E_analytical": ana["E_total"],
                "E_ops": ana["E_ops"],
                "E_mem": ana["E_mem"],
                "E_addr": ana["E_addr"],
                "E_gpu": emp["gpu_energy_J"],
                "time_per_sample_ms": emp["time_per_sample_ms"],
                "avg_gpu_power_W": emp["avg_gpu_power_W"],
                "cpu_util_%": emp["cpu_util_%"],
                "ram_rss_MB": emp["ram_rss_MB"],
                "ram_delta_MB": emp["ram_delta_MB"],
            })
        return pd.DataFrame(rows)

    @staticmethod
    def get_neuron_counts_lif(snn, example_input: torch.Tensor) -> Dict[str, int]:
        """Neuron counts for both IFNode and LIFNode."""
        snn.eval()
        counts = {}
        hooks = []

        def make_hook(name):
            def hook(module, inp, out):
                counts[name] = out[0].numel() if isinstance(out, (tuple, list)) else out.numel()
            return hook

        for name, mod in snn.named_modules():
            if isinstance(mod, (neuron.IFNode, neuron.LIFNode)):
                hooks.append(mod.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            functional.reset_net(snn)
            _ = snn(example_input.to(next(snn.parameters()).device))

        for h in hooks:
            h.remove()
        return counts

    @staticmethod
    def measure_empirical_direct(snn: nn.Module, loader: DataLoader, T: int, device: torch.device, encoder=None, n_samples: int =1000) -> Dict:
        """Like measure_empirical but supports an optional PoissonEncoder."""
        snn.eval()
        functional.reset_net(snn)

        x0, _ = next(iter(loader))
        x0 = x0.to(device)
        for _ in range(5):
            functional.reset_net(snn)
            for _ in range(T):
                inp = encoder(x0) if encoder is not None else x0
                snn(inp)

        if _HAS_NVML:
            start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(_NVML_HANDLE)
        else:
            start_energy = 0
        start_time = time.perf_counter()
        cpu_start  = psutil.cpu_percent(interval=None)
        mem_start  = psutil.Process().memory_info().rss

        seen = 0
        with torch.no_grad():
            for x, _ in loader:
                if seen >= n_samples:
                    break
                x = x.to(device)
                functional.reset_net(snn)
                for _ in range(T):
                    inp = encoder(x) if encoder is not None else x
                    snn(inp)
                seen += x.size(0)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        if _HAS_NVML:
            end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(_NVML_HANDLE)
        else:
            end_energy = 0
        cpu_end = psutil.cpu_percent(interval=None)
        mem_end = psutil.Process().memory_info().rss

        gpu_energy_J = (end_energy - start_energy) / 1000.0
        wall_s = end_time - start_time
        return {
            "gpu_energy_J": gpu_energy_J,
            "wall_time_s": wall_s,
            "time_per_sample_ms": 1000.0 * wall_s / max(seen, 1),
            "avg_gpu_power_W": gpu_energy_J / wall_s if wall_s > 0 else 0.0,
            "cpu_util_%": (cpu_start + cpu_end) / 2,
            "ram_rss_MB": mem_end / (1024 ** 2),
            "ram_delta_MB": (mem_end - mem_start) / (1024 ** 2),
            "n_samples": seen,
            "T": T,
        }
        
    @staticmethod
    def plot_all(df: pd.DataFrame, title_prefix: str = ""):
        return Plotter.plot_all(df, title_prefix)