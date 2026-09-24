from __future__ import annotations
 
import math
import time
import threading
import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
 
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

import spikingjelly 
from spikingjelly.activation_based import neuron, functional

from ..constants import _NVML, _HAS_NVML, _HAS_PSUTIL



from .types import (
    LayerSpec, EnergyModel, SYNAPTIC_TYPES, NEURON_TYPES,
    E_ADD_INT32, E_MULT_INT32, E_ADD_FP32, E_MULT_FP32,
    _SRAM_POINTS, BYTES_PER_WORD, INT32_MODEL, FP32_MODEL
)
 
def sram_access_energy(size_kB: float) -> float:
    """Piecewise-linear SRAM access energy vs memory size (Lemaire's method).
    Assumes E_read == E_write."""
    pts = _SRAM_POINTS
    if size_kB <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if size_kB <= x1:
            return y0 + (y1 - y0) * (size_kB - x0) / (x1 - x0)
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    return y1 + (y1 - y0) * (size_kB - x1) / (x1 - x0)
 





# runs one forward pass with hooks attached to every Conv2d, Linear, and spiking neuron.
# It records each layer's shapes, kernel size, stride, which neuron follows it,
# and how many neurons that is. for Lemaire's equations, expectd to be modular,
# it reads it off the live model.

# Records:
# - Networks shape (channels, kernel size, input and output dimensions)
# - Which spiking neuron sits immediately after it
# - How many neurons that is
# - Whether that neuron is LIF (has leak) or IF (no leak)
# - Whether the layer receives spikes or analog numbers
# - If spikes, what they've been multiplied by

def profile_topology(model: nn.Module, example_input: torch.Tensor, device: torch.device, probe_T: int = 8) -> List[LayerSpec]:
    """
    Recovers layer geometry, links each synaptic layer to the neuron that
    follows it, and decides analog-vs-spiking from the DISTINCT VALUES seen
    across `probe_T` timesteps.  One timestep is not enough: right after a
    reset most layers emit all zeros, which looks binary.
    
    Info:
    ```
    profile_topology(model, x0[:1], device, probe_T=8)
    ├─ syn_hook fires      -> name, kind, exec_order, has_bias, geometry
    ├─ syn_hook fires x8   -> distinct values collected into seen_values
    ├─ neu_hook fires      -> exec_order, neurons per sample, is-LIF flag
    ├─ after probe         -> analog_input, input_scale
    └─ linking step        -> neuron_name, neuron_count, neuron_is_lif
    ```
    """
    model.eval() # evaluation mode: disables Dropoff(p=0.4)
    specs: Dict[str, LayerSpec] = {}
    order: List[Tuple[int, str, str]] = [] # List of tuples Execution Order - Each entry is  (position, name, "syn" | "neu") -> helps to wokout which neuron follows which layer
    neuron_info: Dict[str, Tuple[int, int, bool]] = {}  # neuron_name -> (position, neuron count, is it LIF)
    seen_values: Dict[str, set] = {} # Layer name -> the set of distinct values ever seen at its input. This is how spiking is distinguished from analog.
    counter = {"i": 0} # A counter shared by both hooks, so every layer gets a unique execution position.
    hooks = [] # A counter shared by both hooks, so every layer gets a unique execution position.
 
    # A hook only receives (m, inp, out) so it never learns its own layer's name.
    # So syn_hook("features.0", mod) returns a hook (function closure) that has "features.0" baked into it.
    # syn_hook can be called once per layer to manufacture a custom hook for each.
    def syn_hook(name, mod):
        def hook(m, inp, out):
            """
            out - the result what came out
            inp - a tuple of what went in
            m - the layer itself
            """
            x = inp[0] # incoming tensor
            with torch.no_grad():
                u = torch.unique(x.detach()) # torch.unique returns the sorted distinct values, which is the whole spiking detector.
                # Binary spikes -> [0., 1.] have two values
                # Scaled spikes -> [0., 1.8827] have two values
                # An image -> thousands of values
                if u.numel() <= 8: # Eight or fewer distinct values so as to record them all.
                    seen_values.setdefault(name, set()).update(
                        round(float(v), 6) for v in u) # gets the set for this layer, creating an empty one if it's the first time
                else:
                    seen_values.setdefault(name, set()).add(float("nan")) # More than eight values means definitely analog. storing would be wasteful so Nan means 'analog'
            if name in specs: # The hook fires 8 times (once per probe timestep). Shape only needs recording once, so exits out on repeats.
                return
            idx = counter["i"]; counter["i"] += 1 # Stamp this layer with its execution position and bump the counter.
            if isinstance(m, nn.Conv2d):
                # For a conv layer. Tensor shapes are [batch, channels, height, width], so index 1 is channels, 2 is height, 3 is width.
                # Input shape gives Cin/Hin/Win, output shape gives Cout/Hout/Wout.
                # m.in_channels is ignored as its motr accurate to read from the tensor incase of LazyConv2d
                sp = LayerSpec(
                    name=name, kind="conv", exec_order=idx,
                    has_bias=m.bias is not None, # if model/converted model has biases, DirectSNN uses bias=False
                    Cin=x.shape[1], Hin=x.shape[2], Win=x.shape[3],
                    Cout=out.shape[1], Hout=out.shape[2], Wout=out.shape[3],
                    Hk=m.kernel_size[0], Wk=m.kernel_size[1], S=m.stride[0] # S=m.stride[0] takes only the vertical stride.
                )
            else:
                # Linear layers are simpler and aren't lazy by the time we get here, so in_features and out_features can be read directly
                sp = LayerSpec(
                    name=name, kind="fc", exec_order=idx,
                    has_bias=m.bias is not None,
                    Nin=m.in_features, Nout=m.out_features
                )
            specs[name] = sp
            order.append((idx, name, "syn")) # Stores the linear layer it, and logs it as a synaptic layer executed at position idx
        return hook
 
    def neu_hook(name, mod): # the spy on spiking neurons
        def hook(m, inp, out):
            if name in neuron_info: # only once
                return
            idx = counter["i"]; counter["i"] += 1
            t = out[0] if isinstance(out, (tuple, list)) else out
            n = t.numel() // max(t.shape[0], 1) # most neurons return a plain tensor, but some variants return a tuple, this handles both cases
            # numel() is the total element count across the whole batch. Dividing by shape[0] (the batch size) gives neurons per sample.
            # max(..., 1) guards against a zero-length batch causing a division by zero.
            neuron_info[name] = (idx, n, isinstance(m, neuron.LIFNode))
            # The third element is the leak flag. LIF neurons decay their membrane potential
            # every timestep, which is a multiply, so they pay T * N MACs. IF neurons don't. Converted is all IF, also DirectSNN is all LIF
            order.append((idx, name, "neu"))
        return hook
 
    # attaching the spies(registering the hooks)
    for name, mod in model.named_modules():
        if isinstance(mod, SYNAPTIC_TYPES): # (nn.Conv2d, nn.Linear)
            hooks.append(mod.register_forward_hook(syn_hook(name, mod)))
        elif isinstance(mod, NEURON_TYPES): # neuron.BaseNode is the parent class of IFNode, LIFNode and the rest, so isinstance matches all of them.
            hooks.append(mod.register_forward_hook(neu_hook(name, mod)))
 
    with torch.no_grad():
        functional.reset_net(model) # reset_net clears every membrane potential to its starting value.
        for _ in range(probe_T):
            model(example_input.to(device))
    for h in hooks:
        h.remove() # Detach the spies.
    functional.reset_net(model) # Reset again, so the model is left exactly as found.
 
    for name, sp in specs.items():
        raw = seen_values.get(name, set())
        has_many = any(math.isnan(v) for v in raw)
        nonzero = sorted(v for v in raw if not math.isnan(v) and v != 0.0) # both spiking and analog tensors contain zeros.
        if has_many or len(nonzero) > 1:
            sp.analog_input = True # Many values, or more than one distinct non-zero value, means analog.
            sp.input_scale = 1.0
        else:
            sp.analog_input = False  # Exactly one distinct non-zero value means binary-scaled spikes,
            sp.input_scale = nonzero[0] if nonzero else 1.0
 
    order.sort() # Sorts by the first tuple element, idx, putting everything back into execution order.
    for i, (idx, name, kind) in enumerate(order):
        if kind != "syn":  # For each synaptic layer, scan forward until the first spiking neuron is found. That's its neuron. break stops at the first match.
            continue
        for j in range(i + 1, len(order)):
            if order[j][2] == "neu":
                nname = order[j][1]
                _, cnt, is_lif = neuron_info[nname]
                specs[name].neuron_name = nname
                specs[name].neuron_count = cnt
                specs[name].neuron_is_lif = is_lif
                break
 
    # Hand back the layer specs in execution order, such that the per-layer table reads top to bottom like the network.
    return sorted(specs.values(), key=lambda s: s.exec_order)
 
 
@torch.no_grad()
def accuracy_vs_T(model, loader, Ts: List[int], device,  spiking: bool = True, max_batches: Optional[int] = None) -> pd.DataFrame:
    """SNN: accumulate output over T timesteps, then argmax.  ANN: single pass."""
    model.eval()
    rows = []
    for T in (list(Ts) if spiking else [1]):
        correct = total = 0
        for b, (x, y) in enumerate(loader):
            if max_batches is not None and b >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            if spiking:
                functional.reset_net(model)
                out = None
                for _ in range(T):
                    o = model(x)
                    out = o if out is None else out + o
            else:
                out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.numel()
        functional.reset_net(model)
        rows.append({"T": T, "accuracy": correct / max(total, 1), "n_eval": total})
    return pd.DataFrame(rows)
 
 

 

def run_benchmark(
    snn: nn.Module, loader: DataLoader, device: Union[torch.device, str],
    Ts: List[int] = (1, 2, 4, 8, 16, 32, 64),
    max_batches_l1: Optional[int] = 20,
    energy_model: EnergyModel = INT32_MODEL,
    run_level3: bool = True,
    l3_min_duration_s: float = 20.0,
    l3_repeats: int = 3,
    measure_accuracy: bool = True,
    acc_max_batches: Optional[int] = None,
    fc_acc_literal: bool = False,
    strict_validation: bool = True,
    verbose: bool = True
):
    snn = snn.to(device).eval()
 
    x0, _ = next(iter(loader))
    base_specs = profile_topology(snn, x0[:1].to(device), device)
 
    if not any(s.neuron_count for s in base_specs):
        raise ValueError(
            "No spiking neurons found. If this is a plain ANN, use "
            "ann_baseline() instead -- run_benchmark() would treat every layer "
            "as an encoder and merely report the ANN executed T times.")
 
    if verbose:
        print(f"{len(base_specs)} synaptic layers, "
              f"{sum(s.neuron_count for s in base_specs)} spiking neurons/sample")
        for s in base_specs:
            tag = "ANALOG-IN" if s.analog_input else f"spike-in (x{s.input_scale:.4g})"
            print(f"  {s.name:<26} {s.kind:<5} {tag:<24} "
                  f"neurons={s.neuron_count:<7} -> {s.neuron_name}")
 
    rows, frames = [], {}
    for T in Ts:
        if verbose:
            print(f"\n--- T = {T} ---")
        specs = count_activity(snn, base_specs, loader, T, device,
                               max_batches=max_batches_l1)
        validate_activity(specs, T, strict=strict_validation)
        l1 = level1_summary(specs, T)
        snn_e, ann_e, per_layer = analytical_energy(
            specs, T, em=energy_model, fc_acc_literal=fc_acc_literal)
        frames[T] = per_layer
 
        row = {
            "T": T,
            "total_spikes": l1["total_spikes_per_sample"],
            "total_neurons": l1["total_neurons"],
            "spikes_per_neuron": l1["spikes_per_neuron_per_inference"],
            "spikes_per_neuron_per_step": l1["spikes_per_neuron_per_timestep"],
            "snn_mac": snn_e["mac"], "snn_acc": snn_e["acc"],
            "snn_mem_ops": snn_e["rd"] + snn_e["wr"],
            "ann_mac": ann_e["mac"], "ann_mem_ops": ann_e["rd"] + ann_e["wr"],
            "E_snn": snn_e["E_total"], "E_snn_mem": snn_e["E_mem"],
            "E_snn_ops": snn_e["E_ops"], "E_snn_addr": snn_e["E_addr"],
            "E_ann": ann_e["E_total"],
            "E_ratio_ann_over_snn": snn_e["ratio_ann_over_snn"],
            
            # "E_proxy": simple_proxy_energy(l1["total_spikes_per_sample"],snn_e["acc"] + snn_e["mac"]),
            # "proxy_underestimate": snn_e["E_total"] / simple_proxy_energy(l1["total_spikes_per_sample"], snn_e["acc"] + snn_e["mac"]),
        }
        if run_level3:
            row.update({k: v for k, v in measure_empirical(
                snn, loader, T, device, min_duration_s=l3_min_duration_s,
                repeats=l3_repeats).items() if k != "T"})
        rows.append(row)
 
        if verbose:
            print(f"  spikes/sample     : {row['total_spikes']:.1f}")
            print(f"  spikes/neuron/inf : {row['spikes_per_neuron']:.4f} "
                  f"({'BELOW' if row['spikes_per_neuron'] < 1.72 else 'ABOVE'} 1.72)")
            print(f"  E_ann / E_snn     : {row['E_ratio_ann_over_snn']:.3f}x")
 
    df = pd.DataFrame(rows)
        
    if measure_accuracy:
        acc = accuracy_vs_T(snn, loader, list(Ts), device, spiking=True, max_batches=acc_max_batches)
        df = df.merge(acc[["T", "accuracy"]], on="T", how="left")

        # Eq. 5: Accuracy / Energy (Sales2025)
        n_classes = base_specs[-1].n_out_positions
        chance = 1.0 / n_classes

        df["acc_per_joule_raw"] = df["accuracy"] / df["E_snn"]
        df["acc_per_joule"] = ((df["accuracy"] - chance).clip(lower=0) / df["E_snn"])
        df.loc[df["accuracy"] < 0.5, "acc_per_joule"] = np.nan
 
    return df, frames
 
 
def plot_all(df: pd.DataFrame, prefix: str = ""):
    Ts = df["T"].values
 
    def _x(ax):
        ax.set_xscale("log", base=2); ax.set_xticks(Ts)
        ax.set_xticklabels(Ts); ax.grid(True, alpha=0.3)
 
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(Ts, df["spikes_per_neuron"], "s-", color="C1", lw=2, ms=7,
            label="spikes / neuron / inference")
    ax.axhline(1.72, color="red", ls="--", lw=1.5, label="Davidson & Furber (1.72)")
    ax.set_xlabel("Timesteps (T)"); ax.set_ylabel("Spikes / neuron / inference")
    ax.set_title(f"{prefix}Spike rate"); ax.legend(); _x(ax)
    fig.tight_layout(); plt.show()
 
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(Ts, df["E_snn"], "o-", color="C2", lw=2, ms=7, label="SNN")
    ax.plot(Ts, df["E_ann"], "--", color="C3", lw=2, label="ANN baseline")
    ax.set_yscale("log"); ax.set_xlabel("Timesteps (T)")
    ax.set_ylabel("Analytical energy per inference (J)")
    ax.set_title(f"{prefix}Analytical energy"); ax.legend(); _x(ax)
    fig.tight_layout(); plt.show()
 
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(Ts, df["E_ratio_ann_over_snn"], "D-", color="C4", lw=2, ms=7)
    ax.axhline(1.0, color="k", ls=":", lw=1.5, label="break-even")
    ax.set_yscale("log"); ax.set_xlabel("Timesteps (T)")
    ax.set_ylabel(r"$E_{ANN}/E_{SNN}$")
    ax.set_title(f"{prefix}Energy advantage"); ax.legend(); _x(ax)
    fig.tight_layout(); plt.show()
 
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.stackplot(Ts, df["E_snn_mem"], df["E_snn_ops"], df["E_snn_addr"],
                 labels=[r"$E_{mem}$", r"$E_{ops}$", r"$E_{addr}$"], alpha=0.85)
    ax.set_xlabel("Timesteps (T)"); ax.set_ylabel("Energy (J)")
    ax.set_title(f"{prefix}SNN energy breakdown"); ax.legend(loc="upper left"); _x(ax)
    fig.tight_layout(); plt.show()
 
    if "accuracy" in df.columns and df["accuracy"].notna().any():
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        ax1.plot(Ts, 100 * df["accuracy"], "o-", color="C0", lw=2, ms=7)
        ax1.set_xlabel("Timesteps (T)"); ax1.set_ylabel("Accuracy (%)", color="C0")
        ax1.tick_params(axis="y", labelcolor="C0"); _x(ax1)
        ax2 = ax1.twinx()
        ax2.plot(Ts, df["E_ratio_ann_over_snn"], "D--", color="C4", lw=2)
        ax2.axhline(1.0, color="k", ls=":", lw=1.2)
        ax2.set_ylabel(r"$E_{ANN}/E_{SNN}$", color="C4"); ax2.set_yscale("log")
        ax2.tick_params(axis="y", labelcolor="C4")
        ax1.set_title(f"{prefix}Accuracy vs energy advantage")
        fig.tight_layout(); plt.show()
 
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(df["E_snn"], 100 * df["accuracy"], "o-", lw=2, ms=7)
        for _, r in df.iterrows():
            ax.annotate(f"T={int(r['T'])}", (r["E_snn"], 100 * r["accuracy"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.axvline(df["E_ann"].iloc[0], color="C3", ls="--", label="ANN energy")
        ax.set_xscale("log"); ax.set_xlabel("Analytical energy per inference (J)")
        ax.set_ylabel("Accuracy (%)"); ax.grid(True, alpha=0.3); ax.legend()
        ax.set_title(f"{prefix}Accuracy-energy Pareto front")
        fig.tight_layout(); plt.show()
 
    if "time_per_sample_ms" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(Ts, df["time_per_sample_ms"], "D-", color="C5", lw=2, ms=7)
        ax.set_xlabel("Timesteps (T)"); ax.set_ylabel("Time per sample (ms)")
        ax.set_title(f"{prefix}Measured latency"); _x(ax)
        fig.tight_layout(); plt.show()
        
    if "acc_per_joule" in df.columns and df["acc_per_joule"].notna().any():
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(Ts, df["acc_per_joule"], "o-", color="C6", lw=2, ms=7, label="floored (accuracy above chance)")
        ax.plot(Ts, df["acc_per_joule_raw"], "s--", color="C7", lw=1.5,alpha=0.6, label="raw Acc / E")
        ax.set_yscale("log"); ax.set_xlabel("Timesteps (T)")
        ax.set_ylabel("Accuracy per joule")
        ax.set_title(f"{prefix}Eq. 5 efficiency (raw form peaks where the network fails)[Sales2025]")
        ax.legend(fontsize=8); _x(ax)
        fig.tight_layout(); plt.show()    
        

from .level1 import (
    count_activity,
    validate_activity,
    level1_summary
)

from .level2 import (
    snn_layer_counts,
    _dense_counts,
    ann_layer_counts,
    layer_memory_kB,
    counts_to_energy,
    analytical_energy,
    ann_baseline,
    simple_proxy_energy
)

from .level3 import (
    _PowerSampler,
    _energy_J,
    measure_idle_power,
    measure_empirical
)


from .accuracy_retention import (
    Source,
    
    _label,
    accuracy_retention,
    best_retention,
    plot_accuracy_retention,
    plot_retention_vs_energy
)

from .ann_compare import (
    profile_ann,
    plot_ann,
    plot_comparison,
    comparison_table,
    crossing_points
)


from .totals_report import (
    build_totals,
    totals_at,
    plot_totals,
    plot_memory_accesses,
    memory_summary
)



__all__ = [
    
    "accuracy_vs_T",
    "run_benchmark",
    "plot_all",
    
    "count_activity",
    "validate_activity",
    "level1_summary",
    
    "snn_layer_counts",
    "_dense_counts",
    "ann_layer_counts",
    "layer_memory_kB",
    "counts_to_energy",
    "analytical_energy",
    "ann_baseline",
    
    "_PowerSampler",
    "_energy_J",
    "measure_idle_power",
    "measure_empirical",
    
    "Source",
    "_label",
    "accuracy_retention",
    "best_retention",
    "plot_accuracy_retention",
    "plot_retention_vs_energy",
    
    "profile_ann",
    "plot_ann",
    "plot_comparison",
    "comparison_table",
    "crossing_points"
    
    "build_totals",
    "totals_at",
    "plot_totals",
    "plot_memory_accesses",
    "memory_summary"
]
 