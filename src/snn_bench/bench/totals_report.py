from __future__ import annotations

from typing import Dict, Tuple, Optional, List, Union, Iterable

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


def totals_at(totals: pd.DataFrame, at_T: Union[int, Iterable[int], None] = None) -> pd.DataFrame:
    """ANN row plus model rows at the requested T values. at_T=None means every T in the sweep."""
    ann_row = totals[totals["T"].isna()]
    swept = totals[totals["T"].notna()]
    if at_T is not None:
        value = [at_T] if isinstance(at_T, (int, float)) else list(at_T)
        swept = swept[swept["T"].isin(value)]
    return pd.concat([ann_row, swept.sort_values(["model", "T"])], ignore_index=True)


def plot_totals(ann_result: Dict, totals: pd.DataFrame, at_T: Union[int, Iterable[int], None] = None):
    """Side-by-side totals for the ANN and every SNN, Totals across the whole sweep. at_T=None uses every T."""
    snap = totals_at(totals, at_T)
    swept = totals[totals["T"].notna()]
    all_T = sorted(swept["T"].unique().astype(int))
    E_ann = ann_result["totals"]

    def _x(ax):
        ax.set_xscale("log", base=2); ax.set_xticks(all_T)
        ax.set_xticklabels(all_T); ax.grid(True, alpha=0.3)

    # 1. stacked energy, every model at every T, ANN as a leading group
    labels = [m if np.isnan(t) else f"{m.split(' (')[0]}\nT={int(t)}"
              for m, t in zip(snap["model"], snap["T"])]
    idx = np.arange(len(snap))
    fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(snap)), 5))
    ax.bar(idx, snap["E_mem"], label=r"$E_{mem}$", color="C0")
    ax.bar(idx, snap["E_ops"], bottom=snap["E_mem"], label=r"$E_{ops}$", color="C1")
    ax.bar(idx, snap["E_addr"], bottom=snap["E_mem"] + snap["E_ops"],
           label=r"$E_{addr}$", color="C2")
    ax.axhline(E_ann["E_total"], color="k", ls="--", lw=1.5, label="ANN total")
    ax.set_yscale("log")
    ax.set_xticks(idx); ax.set_xticklabels(labels, fontsize=7, rotation=90)
    ax.set_ylabel("Energy per inference (J)")
    ax.set_title("Total analytical energy, full run")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); plt.show()

    # 2. memory share vs T (replaces the single-T composition bar)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, g) in enumerate(swept.groupby("model")):
        g = g.sort_values("T")
        ax.plot(g["T"], g["pct_E_mem"], "o-", lw=2, ms=6, color=f"C{i}", label=name)
    ax.axhline(100 * E_ann["E_mem"] / E_ann["E_total"], color="k", ls="--",
               lw=2, label="ANN")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Timesteps (T)")
    ax.set_ylabel(r"$E_{mem}$ share of total (%)")
    ax.set_title("Memory's share of energy"); ax.legend(fontsize=8); _x(ax)
    fig.tight_layout(); plt.show()

    # 3. operation counts vs T, one panel each
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    panels = [
        ("mac", "MACs", E_ann["mac"]), 
        ("acc", "ACCs", E_ann["acc"]),
        ("mem_reads", "Memory reads", E_ann["rd"]),
        ("mem_writes", "Memory writes", E_ann["wr"])
    ]
    for ax, (col, title, ann_v) in zip(axes.ravel(), panels):
        for i, (name, g) in enumerate(swept.groupby("model")):
            g = g.sort_values("T")
            ax.plot(g["T"], g[col].replace(0, np.nan), "o-", lw=2, ms=5, color=f"C{i}", label=name)
        if ann_v > 0:
            ax.axhline(ann_v, color="k", ls="--", lw=2, label="ANN")
        ax.set_yscale("log"); ax.set_title(title); _x(ax)
    axes[1, 0].set_xlabel("Timesteps (T)"); axes[1, 1].set_xlabel("Timesteps (T)")
    axes[0, 0].set_ylabel("Operations per inference")
    axes[1, 0].set_ylabel("Operations per inference")
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("Operation totals vs timesteps")
    fig.tight_layout(); plt.show()

    # 4. component energies vs T
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    for ax, comp, ann_v in zip(axes, ["E_mem", "E_ops", "E_total"], [E_ann["E_mem"], E_ann["E_ops"], E_ann["E_total"]]):
        for i, (name, g) in enumerate(swept.groupby("model")):
            g = g.sort_values("T")
            ax.plot(g["T"], g[comp], "o-", lw=2, ms=5, color=f"C{i}", label=name)
        ax.axhline(ann_v, color="k", ls="--", lw=2, label="ANN")
        ax.set_yscale("log"); ax.set_xlabel("Timesteps (T)")
        ax.set_title(comp); _x(ax)
    axes[0].set_ylabel("Energy per inference (J)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Energy components vs timesteps")
    fig.tight_layout(); plt.show()

# MEMORY ACCESS PLOTS
def plot_memory_accesses(ann_result: Dict, models: Dict[str, Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]],at_T: Union[int, Iterable[int], None] = None):
    """Memory traffic across the un. at_T=None uses every T."""
    a_layers = ann_result["per_layer"]
    a_rd, a_wr = float(a_layers["rd"].sum()), float(a_layers["wr"].sum())
    a_total = a_rd + a_wr
    all_T = sorted({int(t) for _, f in models.values() for t in f})
    if at_T is not None:
        want = [at_T] if isinstance(at_T, (int, float)) else list(at_T)
        all_T = [t for t in all_T if t in want]

    def _x(ax):
        ax.set_xscale("log", base=2); ax.set_xticks(all_T)
        ax.set_xticklabels(all_T); ax.grid(True, alpha=0.3)

    def _acc(frames, T):
        return float(frames[T]["snn_rd"].sum() + frames[T]["snn_wr"].sum())

    # 1. total memory accesses vs T
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, (_, frames)) in enumerate(models.items()):
        Ts = [t for t in sorted(frames) if t in all_T]
        ax.plot(Ts, [_acc(frames, t) for t in Ts], "o-", lw=2, ms=6, color=f"C{i}", label=name)
    ax.axhline(a_total, color="k", ls="--", lw=2, label=f"ANN ({a_total:,.0f})")
    ax.set_yscale("log"); ax.set_xlabel("Timesteps (T)")
    ax.set_ylabel("Memory accesses per inference")
    ax.set_title("Total memory traffic"); ax.legend(); _x(ax)
    fig.tight_layout(); plt.show()

    # 2. how many times more traffic than the ANN
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, (_, frames)) in enumerate(models.items()):
        Ts = [t for t in sorted(frames) if t in all_T]
        amp = [_acc(frames, t) / a_total for t in Ts]
        ax.plot(Ts, amp, "D-", lw=2, ms=6, color=f"C{i}", label=name)
        ax.annotate(f"{amp[-1]:.1f}x", (Ts[-1], amp[-1]),
                    textcoords="offset points", xytext=(5, 4), fontsize=9)
    ax.axhline(1.0, color="k", ls=":", lw=2, label="ANN parity")
    ax.set_yscale("log"); ax.set_xlabel("Timesteps (T)")
    ax.set_ylabel("SNN accesses / ANN accesses")
    ax.set_title("Memory amplification over the ANN"); ax.legend(); _x(ax)
    fig.tight_layout(); plt.show()

    # 3. reads vs writes
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.2), squeeze=False)
    for ax, (name, (_, frames)) in zip(axes[0], models.items()):
        Ts = [t for t in sorted(frames) if t in all_T]
        rd = [float(frames[t]["snn_rd"].sum()) for t in Ts]
        wr = [float(frames[t]["snn_wr"].sum()) for t in Ts]
        x = np.arange(len(Ts))
        ax.bar(x, rd, label="reads", color="C0")
        ax.bar(x, wr, bottom=rd, label="writes", color="C1")
        ax.axhline(a_total, color="k", ls="--", lw=1.8, label="ANN total")
        ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(Ts)
        ax.set_xlabel("Timesteps (T)"); ax.set_title(name)
        ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)
    axes[0][0].set_ylabel("Memory accesses per inference")
    fig.suptitle("Reads vs writes"); fig.tight_layout(); plt.show()

    # 4. accesses per spike
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, (summary, frames)) in enumerate(models.items()):
        s = summary.sort_values("T")
        Ts = [int(t) for t in s["T"] if int(t) in all_T]
        vals = []
        for t in Ts:
            sp = float(s.loc[s["T"] == t, "total_spikes"].iloc[0])
            vals.append(_acc(frames, t) / sp if sp > 0 else np.nan)
        ax.plot(Ts, vals, "s-", lw=2, ms=6, color=f"C{i}", label=name)
    ax.set_yscale("log"); ax.set_xlabel("Timesteps (T)")
    ax.set_ylabel("Memory accesses per spike")
    ax.set_title("Cost of one spike in memory traffic"); ax.legend(); _x(ax)
    fig.tight_layout(); plt.show()

    # 5.  per-layer memory accesses at one T, against the ANN's per-layer counts
    ann_by_layer = dict(zip(a_layers["layer"], a_layers["rd"] + a_layers["wr"]))
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 4.6), squeeze=False)
    for ax, (name, (_, frames)) in zip(axes[0], models.items()):
        Ts = [t for t in sorted(frames) if t in all_T]
        layers = frames[Ts[-1]]["layer"].tolist()
        x = np.arange(len(Ts))
        bottom = np.zeros(len(Ts))
        for j, lyr in enumerate(layers):
            vals = np.array([
                float(frames[t].loc[frames[t]["layer"] == lyr, "snn_rd"].iloc[0]
                      + frames[t].loc[frames[t]["layer"] == lyr, "snn_wr"].iloc[0])
                for t in Ts])
            ax.bar(x, vals, bottom=bottom, label=lyr, color=f"C{j}")
            bottom += vals
        ax.axhline(a_total, color="k", ls="--", lw=1.8, label="ANN total")
        ax.set_xticks(x); ax.set_xticklabels(Ts)
        ax.set_xlabel("Timesteps (T)"); ax.set_title(name)
        ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=7)
    axes[0][0].set_ylabel("Memory accesses per inference")
    fig.suptitle("Per-layer memory traffic (linear scale shows which layer dominates)")
    fig.tight_layout(); plt.show()

    # 6. per-layer access cost vs footprint
    ref_T = all_T[-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(a_layers["mem_kB"] / 1024, a_layers["pJ_per_access"],
               s=90, marker="s", color="C7", label="ANN layers", zorder=3)
    for _, r in a_layers.iterrows():
        ax.annotate(r["layer"], (r["mem_kB"] / 1024, r["pJ_per_access"]),
                    textcoords="offset points", xytext=(6, -3), fontsize=8)
    for i, (name, (_, frames)) in enumerate(models.items()):
        pl = frames[ref_T]
        ax.scatter(pl["snn_mem_kB"] / 1024, pl["snn_pJ_per_access"],
                   s=70, color=f"C{i}", label=name, zorder=3)
    for kb, pj in [(8, 10), (32, 20), (1024, 100)]:
        ax.plot(kb / 1024, pj, "kx", ms=9)
    ax.annotate("interpolation anchors\n(8 kB, 32 kB, 1 MB)", (1.0, 100),
                textcoords="offset points", xytext=(-110, 18), fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("Layer memory footprint (MB)")
    ax.set_ylabel("Energy per SRAM access (pJ)")
    ax.set_title("Per-access cost is set by layer size, not by T or spiking")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); plt.show()


def memory_summary(ann_result: Dict, totals: pd.DataFrame,
                   at_T: Union[int, Iterable[int], None] = None) -> pd.DataFrame:
    """Compact memory table. Defaults to every T."""
    snap = totals_at(totals, at_T)
    return snap[
        [
            "model", "T", "mem_reads", "mem_writes", "mem_accesses",
            "mem_amplification", "accesses_per_spike", "accesses_per_mac",
            "mean_pJ_per_access", "E_mem", "pct_E_mem"
        ]
    ].copy()