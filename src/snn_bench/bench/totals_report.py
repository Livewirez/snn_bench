from __future__ import annotations

from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_totals(ann_result: Dict, models: Dict[str, Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]]) -> pd.DataFrame:
    """
    One row per (model, T), plus a single ANN row with T = NaN.

    `models` maps a label to the (summary_df, per_layer_frames) pair returned
    by run_benchmark().  Reads and writes are taken from the per-layer frames,
    since the summary only carries their sum.
    """
    rows = []

    a_layers = ann_result["per_layer"]
    a_tot = ann_result["totals"]
    a_rd, a_wr = float(a_layers["rd"].sum()), float(a_layers["wr"].sum())
    rows.append({
        "model": "ANN", "T": np.nan,
        "accuracy": ann_result["accuracy"],
        "spikes": 0.0, "neurons": 0,
        "spikes_per_neuron": np.nan,
        "mac": a_tot["mac"], "acc": a_tot["acc"],
        "mem_reads": a_rd, "mem_writes": a_wr, "mem_accesses": a_rd + a_wr,
        "accesses_per_mac": (a_rd + a_wr) / a_tot["mac"] if a_tot["mac"] else np.nan,
        "accesses_per_spike": np.nan,
        "E_mem": a_tot["E_mem"], "E_ops": a_tot["E_ops"],
        "E_addr": a_tot["E_addr"], "E_total": a_tot["E_total"],
        "pct_E_mem": 100 * a_tot["E_mem"] / a_tot["E_total"],
        "mean_pJ_per_access": 1e12 * a_tot["E_mem"] / (a_rd + a_wr) if (a_rd + a_wr) else np.nan,
        "E_ratio_ann_over_model": 1.0,
        "mem_amplification": 1.0,
    })

    for name, (summary, frames) in models.items():
        for _, r in summary.sort_values("T").iterrows():
            T = int(r["T"])
            pl = frames[T]
            rd, wr = float(pl["snn_rd"].sum()), float(pl["snn_wr"].sum())
            spikes = float(r["total_spikes"])
            rows.append({
                "model": name, "T": T,
                "accuracy": r.get("accuracy", np.nan),
                "spikes": spikes, "neurons": int(r["total_neurons"]),
                "spikes_per_neuron": r["spikes_per_neuron"],
                "mac": r["snn_mac"], "acc": r["snn_acc"],
                "mem_reads": rd, "mem_writes": wr, "mem_accesses": rd + wr,
                "accesses_per_mac": (rd + wr) / r["snn_mac"] if r["snn_mac"] else np.inf,
                "accesses_per_spike": (rd + wr) / spikes if spikes else np.nan,
                "E_mem": r["E_snn_mem"], "E_ops": r["E_snn_ops"],
                "E_addr": r["E_snn_addr"], "E_total": r["E_snn"],
                "pct_E_mem": 100 * r["E_snn_mem"] / r["E_snn"],
                "mean_pJ_per_access": 1e12 * r["E_snn_mem"] / (rd + wr) if (rd + wr) else np.nan,
                "E_ratio_ann_over_model": r["E_ratio_ann_over_snn"],
                "mem_amplification": (rd + wr) / (a_rd + a_wr),
            })

    return pd.DataFrame(rows)


def totals_at(totals: pd.DataFrame, at_T: int) -> pd.DataFrame:
    """The ANN row plus every model's row at one chosen T."""
    return pd.concat([totals[totals["T"].isna()], totals[totals["T"] == at_T]], ignore_index=True)


def plot_totals(ann_result: Dict, totals: pd.DataFrame, at_T: int = 32):
    """Side-by-side totals for the ANN and every SNN at one operating point,
    plus the component curves across T."""
    snap = totals_at(totals, at_T)
    labels = [f"{m}" if np.isnan(t) else f"{m}\nT={int(t)}" for m, t in zip(snap["model"], snap["T"])]
    idx = np.arange(len(snap))

    # 1. total energy, split into Lemaire's three components
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(idx, snap["E_mem"], label=r"$E_{mem}$", color="C0")
    ax.bar(idx, snap["E_ops"], bottom=snap["E_mem"], label=r"$E_{ops}$", color="C1")
    ax.bar(idx, snap["E_addr"], bottom=snap["E_mem"] + snap["E_ops"], label=r"$E_{addr}$", color="C2")
    for i, v in enumerate(snap["E_total"]):
        ax.annotate(f"{v:.2e} J", (i, v), ha="center", textcoords="offset points", xytext=(0, 4), fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(idx); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Energy per inference (J)")
    ax.set_title(f"Total analytical energy (T = {at_T})")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); plt.show()

    # 2. where each model's energy sits, as shares
    fig, ax = plt.subplots(figsize=(8, 4))
    share = snap[["E_mem", "E_ops", "E_addr"]].div(snap["E_total"], axis=0) * 100
    left = np.zeros(len(snap))
    for col, c in zip(["E_mem", "E_ops", "E_addr"], ["C0", "C1", "C2"]):
        ax.barh(idx, share[col], left=left, color=c, label=col)
        for i, (v, l) in enumerate(zip(share[col], left)):
            if v > 6:
                ax.annotate(f"{v:.0f}%", (l + v / 2, i), ha="center", va="center", fontsize=9, color="white")
        left += share[col].values
    ax.set_yticks(idx); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Share of total energy (%)"); ax.set_xlim(0, 100)
    ax.set_title(f"Energy composition (T = {at_T})")
    ax.legend(loc="lower right"); ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout(); plt.show()

    # 3. operation counts, all four kinds
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.2
    for k, (col, lab) in enumerate([("mac", "MACs"), ("acc", "ACCs"), ("mem_reads", "reads"), ("mem_writes", "writes")]):
        vals = snap[col].replace(0, np.nan)
        ax.bar(idx + (k - 1.5) * w, vals, w, label=lab, color=f"C{k}")
    ax.set_yscale("log")
    ax.set_xticks(idx); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Operations per inference")
    ax.set_title(f"Operation totals (T = {at_T}; zero bars omitted)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); plt.show()

    # 4. component energies across the whole sweep
    E_ann = ann_result["totals"]
    sweeps = totals[totals["T"].notna()]
    all_T = sorted(sweeps["T"].unique().astype(int))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    for ax, comp, ann_v in zip(axes, ["E_mem", "E_ops", "E_total"], [E_ann["E_mem"], E_ann["E_ops"], E_ann["E_total"]]):
        for i, (name, g) in enumerate(sweeps.groupby("model")):
            g = g.sort_values("T")
            ax.plot(g["T"], g[comp], "o-", lw=2, ms=5, color=f"C{i}", label=name)
        ax.axhline(ann_v, color="k", ls="--", lw=2, label="ANN")
        ax.set_yscale("log"); ax.set_xscale("log", base=2)
        ax.set_xticks(all_T); ax.set_xticklabels(all_T)
        ax.set_xlabel("Timesteps (T)"); ax.set_title(comp)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Energy per inference (J)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Energy components vs timesteps")
    fig.tight_layout(); plt.show()


# MEMORY ACCESS PLOTS
def plot_memory_accesses(ann_result: Dict, models: Dict[str, Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]], at_T: int = 32):
    """
    Memory Traffic.
    """
    a_layers = ann_result["per_layer"]
    a_rd, a_wr = float(a_layers["rd"].sum()), float(a_layers["wr"].sum())
    a_total = a_rd + a_wr
    all_T = sorted({int(t) for s, _ in models.values() for t in s["T"]})

    def _x(ax):
        ax.set_xscale("log", base=2); ax.set_xticks(all_T)
        ax.set_xticklabels(all_T); ax.grid(True, alpha=0.3)

    # 1. total memory accesses vs T
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, (summary, frames)) in enumerate(models.items()):
        Ts = sorted(frames)
        acc = [frames[T]["snn_rd"].sum() + frames[T]["snn_wr"].sum() for T in Ts]
        ax.plot(Ts, acc, "o-", lw=2, ms=6, color=f"C{i}", label=name)
    ax.axhline(a_total, color="k", ls="--", lw=2, label=f"ANN ({a_total:,.0f})")
    ax.set_yscale("log"); ax.set_xlabel("Timesteps (T)")
    ax.set_ylabel("Memory accesses per inference")
    ax.set_title("Total memory traffic"); ax.legend(); _x(ax)
    fig.tight_layout(); plt.show()

    # 2. how many times more traffic than the ANN
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, (summary, frames)) in enumerate(models.items()):
        Ts = sorted(frames)
        amp = [(frames[T]["snn_rd"].sum() + frames[T]["snn_wr"].sum()) / a_total
               for T in Ts]
        ax.plot(Ts, amp, "D-", lw=2, ms=6, color=f"C{i}", label=name)
        for T, v in zip(Ts, amp):
            if T in (all_T[-1], all_T[len(all_T) // 2]):
                ax.annotate(f"{v:.1f}x", (T, v), textcoords="offset points",
                            xytext=(4, 4), fontsize=9)
    ax.axhline(1.0, color="k", ls=":", lw=2, label="ANN parity")
    ax.set_yscale("log"); ax.set_xlabel("Timesteps (T)")
    ax.set_ylabel("SNN accesses / ANN accesses")
    ax.set_title("Memory amplification over the ANN")
    ax.legend(); _x(ax)
    fig.tight_layout(); plt.show()

    # 3. reads vs writes, stacked per model across T
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.2), squeeze=False)
    for ax, (name, (summary, frames)) in zip(axes[0], models.items()):
        Ts = sorted(frames)
        rd = [frames[T]["snn_rd"].sum() for T in Ts]
        wr = [frames[T]["snn_wr"].sum() for T in Ts]
        x = np.arange(len(Ts))
        ax.bar(x, rd, label="reads", color="C0")
        ax.bar(x, wr, bottom=rd, label="writes", color="C1")
        ax.axhline(a_total, color="k", ls="--", lw=1.8, label="ANN total")
        ax.set_yscale("log")
        ax.set_xticks(x); ax.set_xticklabels(Ts)
        ax.set_xlabel("Timesteps (T)"); ax.set_title(name)
        ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)
    axes[0][0].set_ylabel("Memory accesses per inference")
    fig.suptitle("Reads vs writes")
    fig.tight_layout(); plt.show()

    # 4. accesses triggered per spike -- a structural constant of the topology
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, (summary, frames)) in enumerate(models.items()):
        s = summary.sort_values("T")
        Ts = s["T"].astype(int).tolist()
        per_spike = []
        for T in Ts:
            sp = float(s.loc[s["T"] == T, "total_spikes"].iloc[0])
            tot = frames[T]["snn_rd"].sum() + frames[T]["snn_wr"].sum()
            per_spike.append(tot / sp if sp > 0 else np.nan)
        ax.plot(Ts, per_spike, "s-", lw=2, ms=6, color=f"C{i}", label=name)
    ax.set_yscale("log"); ax.set_xlabel("Timesteps (T)")
    ax.set_ylabel("Memory accesses per spike")
    ax.set_title("Cost of one spike in memory traffic")
    ax.legend(); _x(ax)
    fig.tight_layout(); plt.show()

    # 5. per-layer memory accesses at one T, against the ANN's per-layer counts
    ann_by_layer = dict(zip(a_layers["layer"], a_layers["rd"] + a_layers["wr"]))
    for name, (summary, frames) in models.items():
        pl = frames[at_T]
        layers = pl["layer"].tolist()
        x = np.arange(len(layers))
        w = 0.38
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(x - w / 2, [ann_by_layer.get(l, 0) for l in layers], w,
               label="ANN", color="C7")
        ax.bar(x + w / 2, pl["snn_rd"] + pl["snn_wr"], w, label=name, color="C0")
        ax.set_yscale("log")
        ax.set_xticks(x); ax.set_xticklabels(layers, rotation=30, ha="right")
        ax.set_ylabel("Memory accesses per inference")
        ax.set_title(f"Per-layer memory traffic, {name} (T = {at_T})")
        ax.legend(); ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout(); plt.show()

    # 6. per-layer access cost vs footprint
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(a_layers["mem_kB"] / 1024, a_layers["pJ_per_access"],  s=90, marker="s", color="C7", label="ANN layers", zorder=3)
    for _, r in a_layers.iterrows():
        ax.annotate(r["layer"], (r["mem_kB"] / 1024, r["pJ_per_access"]), textcoords="offset points", xytext=(6, -3), fontsize=8)
    for i, (name, (summary, frames)) in enumerate(models.items()):
        pl = frames[at_T]
        ax.scatter(pl["snn_mem_kB"] / 1024, pl["snn_pJ_per_access"], s=70, color=f"C{i}", label=name, zorder=3)
    for kb, pj in [(8, 10), (32, 20), (1024, 100)]:
        ax.plot(kb / 1024, pj, "kx", ms=9)
    ax.annotate("interpolation anchors\n(8 kB, 32 kB, 1 MB)", (1.0, 100), textcoords="offset points", xytext=(-110, 18), fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("Layer memory footprint (MB)")
    ax.set_ylabel("Energy per SRAM access (pJ)")
    ax.set_title("Per-access cost (set by layer size, not by spiking)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); plt.show()


def memory_summary(ann_result: Dict, totals: pd.DataFrame, at_T: int = 32) -> pd.DataFrame:
    snap = totals_at(totals, at_T)
    return snap[
        [
            "model", "T", "mem_reads", "mem_writes", "mem_accesses",
            "mem_amplification", "accesses_per_spike", "accesses_per_mac",
            "mean_pJ_per_access", "E_mem", "pct_E_mem"
        ]
    ].copy()