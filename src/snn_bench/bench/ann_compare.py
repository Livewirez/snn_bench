"""
ann_compare.py
==============
Companion to snn_energy.py.

The ANN is a single operating point, not a curve: it has no timestep axis.
So it is profiled once and drawn as a horizontal reference line on every
T-sweep plot, plus its own per-layer breakdown.

    from snn_energy import *
    from ann_compare import profile_ann, plot_ann, plot_comparison, comparison_table

    ann = profile_ann(best_model, a_test_dataloader, DEVICE)
    plot_ann(ann)

    plot_comparison(ann, {
        "Converted (max-norm)": conv_summary,
        "Direct (surrogate)":   direct_summary,
    })
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from typing import Dict, List, Optional, Tuple, Union

from . import accuracy_vs_T
from .types import LayerSpec, EnergyModel, SYNAPTIC_TYPES, NEURON_TYPES, INT32_MODEL,BYTES_PER_WORD

from .level2 import (
    ann_baseline
)

from .level3 import (
    measure_empirical
)

def profile_ann(
    ann : nn.Module,
    loader: DataLoader,
    device: Union[torch.device, str],
    energy_model: EnergyModel = INT32_MODEL,
    run_level3: bool = True,
    l3_min_duration_s: float = 20.0,
    l3_repeats: int = 3,
    acc_max_batches: Optional[int] = None,
    verbose: bool = True
) -> Dict:
    """
    Returns a dict with:
      totals      -- analytical energy totals (Level 2)
      per_layer   -- DataFrame, one row per synaptic layer
      accuracy    -- test accuracy (single forward pass)
      empirical   -- Level 3 measurements, or None
      point       -- one-row DataFrame, handy for concatenating with SNN sweeps
    """
    ann = ann.to(device).eval()
    x0, _ = next(iter(loader))

    totals, per_layer = ann_baseline(ann, x0[:1].to(device), device, em=energy_model)

    acc_df = accuracy_vs_T(ann, loader, [1], device, spiking=False,
                           max_batches=acc_max_batches)
    accuracy = float(acc_df["accuracy"].iloc[0])

    emp = None
    if run_level3:
        emp = measure_empirical(
            ann, loader, T=1, device=device,
            min_duration_s=l3_min_duration_s,
            repeats=l3_repeats, spiking=False
        )

    point = pd.DataFrame([{
        "model": "ANN",
        "T": np.nan,
        "accuracy": accuracy,
        "E_analytical": totals["E_total"],
        "E_mem": totals["E_mem"],
        "E_ops": totals["E_ops"],
        "E_addr": totals["E_addr"],
        "mac": totals["mac"],
        "mem_ops": totals["rd"] + totals["wr"],
        "time_per_sample_ms": emp["time_per_sample_ms"] if emp else np.nan,
        "E_gpu_marginal_per_sample": emp["gpu_energy_marginal_per_sample_J"] if emp else np.nan,
        "avg_gpu_power_W": emp["avg_gpu_power_W"] if emp else np.nan,
    }])

    if verbose:
        print(f"ANN baseline ({energy_model.label})")
        print(f"  accuracy        : {accuracy:.4f}")
        print(f"  MACs            : {totals['mac']:,.0f}")
        print(f"  memory ops      : {totals['rd'] + totals['wr']:,.0f}")
        print(f"  E_analytical    : {totals['E_total']:.4e} J/sample")
        print(f"    E_mem         : {totals['E_mem']:.4e} J "
              f"({100 * totals['E_mem'] / totals['E_total']:.1f}%)")
        print(f"    E_ops         : {totals['E_ops']:.4e} J "
              f"({100 * totals['E_ops'] / totals['E_total']:.1f}%)")
        print(f"    E_addr        : {totals['E_addr']:.4e} J "
              f"({100 * totals['E_addr'] / totals['E_total']:.1f}%)")
        if emp:
            print(f"  latency         : {emp['time_per_sample_ms']:.4f} ms/sample")
            print(f"  GPU (marginal)  : {emp['gpu_energy_marginal_per_sample_J']:.4e} J/sample")
            print(f"  power spread    : {emp['power_spread_W']:.1f} W across "
                  f"{emp['repeats']} trials")
        print()
        print(per_layer.to_string(index=False))

    return {
        "totals": totals, "per_layer": per_layer,
        "accuracy": accuracy, "empirical": emp, "point": point
    }


# ANN-ONLY PLOTS
def plot_ann(ann_result: Dict, prefix: str = "ANN - "):
    """Where the ANN's energy goes. No T axis exists, so these are per-layer."""
    df = ann_result["per_layer"]
    tot = ann_result["totals"]
    layers = df["layer"].values
    idx = np.arange(len(layers))

    # 1. per-layer energy plots
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(idx, df["E_mem"], label=r"$E_{mem}$")
    ax.bar(idx, df["E_ops"], bottom=df["E_mem"], label=r"$E_{ops}$")
    ax.bar(idx, df["E_total"] - df["E_mem"] - df["E_ops"],
           bottom=df["E_mem"] + df["E_ops"], label=r"$E_{addr}$")
    ax.set_yscale("log")
    ax.set_xticks(idx); ax.set_xticklabels(layers, rotation=30, ha="right")
    ax.set_ylabel("Energy per inference (J)")
    ax.set_title(f"{prefix}Energy by layer (total {tot['E_total']:.3e} J)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); plt.show()

    # 2. share of total: shows which layer owns the result
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(idx, df["pct_of_total"], color="C3")
    ax.bar_label(bars, fmt="%.1f%%", fontsize=9)
    ax.set_xticks(idx); ax.set_xticklabels(layers, rotation=30, ha="right")
    ax.set_ylabel("Share of total energy (%)")
    ax.set_title(f"{prefix}Which layer dominates")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); plt.show()

    # 3. memory footprint vs per-access cost
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.bar(idx - 0.2, df["mem_kB"] / 1024, width=0.4, color="C0", label="footprint")
    ax1.set_ylabel("Layer memory (MB)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.set_xticks(idx); ax1.set_xticklabels(layers, rotation=30, ha="right")
    ax2 = ax1.twinx()
    ax2.bar(idx + 0.2, df["pJ_per_access"], width=0.4, color="C1", label="cost/access")
    ax2.set_ylabel("Energy per SRAM access (pJ)", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax1.set_title(f"{prefix}Memory size drives per-access cost")
    fig.tight_layout(); plt.show()

    # 4. MACs vs memory operations
    fig, ax = plt.subplots(figsize=(7.5, 4))
    w = 0.4
    ax.bar(idx - w / 2, df["mac"], w, label="MACs")
    ax.bar(idx + w / 2, df["rd"] + df["wr"], w, label="memory accesses")
    ax.set_yscale("log")
    ax.set_xticks(idx); ax.set_xticklabels(layers, rotation=30, ha="right")
    ax.set_ylabel("Operations per inference")
    ax.set_title(f"{prefix}Compute vs memory traffic")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); plt.show()


# ANN vs SNN(s)
def plot_comparison(
    ann_result: Dict,
    snn_summaries: Dict[str, pd.DataFrame],
    show_empirical: bool = True
):
    """
    Overlays any number of SNN T-sweeps with the ANN drawn as a horizontal
    reference line. `snn_summaries` maps a label to the DataFrame returned by
    run_benchmark().
    """
    E_ann = ann_result["totals"]["E_total"]
    acc_ann = ann_result["accuracy"]
    emp = ann_result["empirical"]
    all_T = sorted({int(t) for d in snn_summaries.values() for t in d["T"]})

    def _x(ax):
        ax.set_xscale("log", base=2); ax.set_xticks(all_T)
        ax.set_xticklabels(all_T); ax.grid(True, alpha=0.3)

    # 1. analytical energy
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, d) in enumerate(snn_summaries.items()):
        ax.plot(d["T"], d["E_snn"], "o-", lw=2, ms=6, color=f"C{i}", label=name)
    ax.axhline(E_ann, color="k", ls="--", lw=2, label=f"ANN ({E_ann:.2e} J)")
    ax.set_yscale("log"); ax.set_xlabel("Timesteps (T)")
    ax.set_ylabel("Analytical energy per inference (J)")
    ax.set_title("Analytical energy: SNNs vs ANN baseline")
    ax.legend(); _x(ax)
    fig.tight_layout(); plt.show()

    # 2. energy ratio
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, d) in enumerate(snn_summaries.items()):
        ax.plot(d["T"], d["E_ratio_ann_over_snn"], "D-", lw=2, ms=6,
                color=f"C{i}", label=name)
    ax.axhline(1.0, color="k", ls=":", lw=2, label="break-even (= ANN)")
    ax.set_yscale("log"); ax.set_xlabel("Timesteps (T)")
    ax.set_ylabel(r"$E_{ANN}\,/\,E_{SNN}$")
    ax.set_title("Energy advantage over the ANN (above 1 = SNN wins)")
    ax.legend(); _x(ax)
    fig.tight_layout(); plt.show()

    # 3. accuracy
    if any("accuracy" in d.columns and d["accuracy"].notna().any()
           for d in snn_summaries.values()):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, (name, d) in enumerate(snn_summaries.items()):
            if "accuracy" in d.columns:
                ax.plot(d["T"], 100 * d["accuracy"], "o-", lw=2, ms=6,
                        color=f"C{i}", label=name)
        ax.axhline(100 * acc_ann, color="k", ls="--", lw=2,
                   label=f"ANN ({100 * acc_ann:.2f}%)")
        ax.set_xlabel("Timesteps (T)"); ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy retention vs timesteps")
        ax.legend(); _x(ax)
        fig.tight_layout(); plt.show()

        # 4. the money plot: accuracy vs energy, ANN as a single marker
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (name, d) in enumerate(snn_summaries.items()):
            if "accuracy" not in d.columns:
                continue
            ax.plot(d["E_snn"], 100 * d["accuracy"], "o-", lw=2, ms=6,
                    color=f"C{i}", label=name)
            for _, r in d.iterrows():
                ax.annotate(f"T={int(r['T'])}",
                            (r["E_snn"], 100 * r["accuracy"]),
                            textcoords="offset points", xytext=(5, 4), fontsize=8)
        ax.plot([E_ann], [100 * acc_ann], "k*", ms=18, label="ANN", zorder=5)
        ax.set_xscale("log")
        ax.set_xlabel("Analytical energy per inference (J)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy-energy trade-off (up and left is better)")
        ax.grid(True, alpha=0.3); ax.legend()
        fig.tight_layout(); plt.show()

    # 5. spike rate against the 1.72 bound (ANN has none, so no line)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, d) in enumerate(snn_summaries.items()):
        ax.plot(d["T"], d["spikes_per_neuron"], "s-", lw=2, ms=6,
                color=f"C{i}", label=name)
    ax.axhline(1.72, color="red", ls="--", lw=2, label="Davidson & Furber (1.72)")
    ax.set_yscale("log"); ax.set_xlabel("Timesteps (T)")
    ax.set_ylabel("Spikes / neuron / inference")
    ax.set_title("Spike activity vs the energy-saving bound")
    ax.legend(); _x(ax)
    fig.tight_layout(); plt.show()

    # 6 & 7. empirical latency and GPU energy
    if show_empirical and emp is not None:
        if any("time_per_sample_ms" in d.columns for d in snn_summaries.values()):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for i, (name, d) in enumerate(snn_summaries.items()):
                if "time_per_sample_ms" in d.columns:
                    ax.plot(d["T"], d["time_per_sample_ms"], "D-", lw=2, ms=6,
                            color=f"C{i}", label=name)
            ax.axhline(emp["time_per_sample_ms"], color="k", ls="--", lw=2,
                       label=f"ANN ({emp['time_per_sample_ms']:.3f} ms)")
            ax.set_xlabel("Timesteps (T)"); ax.set_ylabel("Time per sample (ms)")
            ax.set_title("Measured GPU latency")
            ax.legend(); _x(ax)
            fig.tight_layout(); plt.show()

        key = "gpu_energy_marginal_per_sample_J"
        if any(key in d.columns for d in snn_summaries.values()):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for i, (name, d) in enumerate(snn_summaries.items()):
                if key in d.columns:
                    ax.plot(d["T"], d[key], "^-", lw=2, ms=6,
                            color=f"C{i}", label=name)
            ax.axhline(emp["gpu_energy_marginal_per_sample_J"], color="k",
                       ls="--", lw=2, label="ANN")
            ax.set_xlabel("Timesteps (T)")
            ax.set_ylabel("Measured GPU energy per sample (J, idle-subtracted)")
            ax.set_title("Level 3: measured GPU energy (note: dense kernels, "
                         "no sparsity benefit)")
            ax.legend(); _x(ax)
            fig.tight_layout(); plt.show()


def comparison_table(ann_result: Dict, snn_summaries: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One tidy frame: the ANN row plus every SNN row, ready for the write-up."""
    cols = ["model", "T", "accuracy", "E_analytical", "E_mem", "E_ops", "E_addr",
            "mac", "mem_ops", "spikes_per_neuron", "E_ratio_ann_over_snn",
            "time_per_sample_ms", "E_gpu_marginal_per_sample"]

    rows = [ann_result["point"].assign(
        spikes_per_neuron=np.nan, E_ratio_ann_over_snn=1.0)]

    for name, d in snn_summaries.items():
        r = pd.DataFrame({
            "model": name,
            "T": d["T"],
            "accuracy": d.get("accuracy", np.nan),
            "E_analytical": d["E_snn"],
            "E_mem": d["E_snn_mem"],
            "E_ops": d["E_snn_ops"],
            "E_addr": d["E_snn_addr"],
            "mac": d["snn_mac"],
            "mem_ops": d["snn_mem_ops"],
            "spikes_per_neuron": d["spikes_per_neuron"],
            "E_ratio_ann_over_snn": d["E_ratio_ann_over_snn"],
            "time_per_sample_ms": d.get("time_per_sample_ms", np.nan),
            "E_gpu_marginal_per_sample": d.get("gpu_energy_marginal_per_sample_J", np.nan),
        })
        rows.append(r)

    out = pd.concat(rows, ignore_index=True)
    return out.reindex(columns=cols)


def crossing_points(ann_result: Dict, summary: pd.DataFrame, acc_tolerance: float = 0.01) -> Dict:
    """
    The three numbers your results chapter needs:
      T_energy   -- smallest T where the SNN stops beating the ANN on energy
      T_1p72     -- smallest T where spikes/neuron exceeds 1.72
      T_accuracy -- smallest T reaching within `acc_tolerance` of ANN accuracy

    If T_accuracy > T_energy, rate-coded conversion cannot win on this model.
    """
    d = summary.sort_values("T")
    out = {"T_energy_breakeven": np.nan, "T_1p72": np.nan, "T_accuracy": np.nan}

    lost = d[d["E_ratio_ann_over_snn"] < 1.0]
    if len(lost):
        out["T_energy_breakeven"] = int(lost["T"].iloc[0])

    over = d[d["spikes_per_neuron"] > 1.72]
    if len(over):
        out["T_1p72"] = int(over["T"].iloc[0])

    if "accuracy" in d.columns and d["accuracy"].notna().any():
        target = ann_result["accuracy"] - acc_tolerance
        ok = d[d["accuracy"] >= target]
        if len(ok):
            out["T_accuracy"] = int(ok["T"].iloc[0])

    out["verdict"] = (
        "SNN never reaches ANN accuracy in the swept range"
        if np.isnan(out["T_accuracy"]) else
        "usable accuracy costs more energy than the ANN"
        if (not np.isnan(out["T_energy_breakeven"])
            and out["T_accuracy"] >= out["T_energy_breakeven"]) else
        "SNN wins on both accuracy and energy"
    )
    return out