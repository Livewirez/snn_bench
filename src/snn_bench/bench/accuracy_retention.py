"""
accuracy_retention.py
=====================
Accuracy retention of each SNN relative to the ANN baseline:

    retention = Acc_SNN / Acc_ANN

e.g. Acc_ANN = 95%, Acc_SNN = 91%  ->  91/95 = 0.957  =  95.7% retention
                                       labelled  0.96x  (-4.2%)

Works from models (evaluates them) or from the summary DataFrames that
run_benchmark() already produced.

    # ann is the dict returned by profile_ann
    ann = profile_ann(best_model, a_test_dataloader, DEVICE)
    #   ann["accuracy"]   -> float, e.g. 0.9864
    #   ann["totals"]     -> dict of analytical energy totals
    #   ann["per_layer"]  -> DataFrame, one row per layer
    #   ann["empirical"]  -> Level 3 dict
    #   ann["point"]      -> one-row DataFrame

    # conv_sum and dir_sum are the first return value of run_benchmark
    conv_sum, conv_layers = run_benchmark(conv_snn_max, a_test_dataloader, DEVICE)
    dir_sum,  dir_layers  = run_benchmark(snn,          a_test_dataloader, DEVICE)
    #   *_sum    -> summary DataFrame, one row per T, with an 'accuracy' column
    #   *_layers -> dict mapping T to a per-layer DataFrame

    ret = accuracy_retention(ann["accuracy"], {
        "Converted (max-norm)": conv_sum,
        "Direct (surrogate)":   dir_sum,
    })
    print(ret.to_string(index=False))
    print(best_retention(ret).to_string(index=False))

    plot_accuracy_retention(ret)                 # picks the best T for the bar chart
    plot_accuracy_retention(ret, at_T=8)         # or force one
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split
from . import accuracy_vs_T


Source = Union[nn.Module, pd.DataFrame, float]


def _label(ratio: float) -> str:
    """'1.02x (+2.1%)' / '0.96x (-4.2%)'. A ratio below 1 is 0.96x, never
    -0.96x."""
    if np.isnan(ratio):
        return "n/a"
    return f"{ratio:.2f}x ({100 * (ratio - 1):+.1f}%)"


def accuracy_retention(
    baseline: Source,
    models: Dict[str, Source],
    loader: Optional[DataLoader] = None,
    device: Optional[Union[torch.device, str]] = None,
    Ts: Optional[List[int]] = None,
    max_batches: Optional[int] = None
) -> pd.DataFrame:
    """
    baseline : the ANN. An nn.Module (evaluated with a single forward pass),
               a float accuracy, or a DataFrame with an 'accuracy' column.
    models   : label -> nn.Module (swept over Ts) or a run_benchmark summary
               DataFrame that already carries 'T' and 'accuracy'.

    Returns one row per (model, T) with:
      accuracy, baseline_accuracy, retention, retention_pct,
      delta_pp (percentage points), delta_rel_pct, label
    """
    # ---- baseline accuracy 
    if isinstance(baseline, nn.Module):
        if loader is None or device is None:
            raise ValueError("loader and device are required to evaluate an "
                             "nn.Module baseline")
        base_acc = float(accuracy_vs_T(baseline, loader, [1], device,
                                       spiking=False,
                                       max_batches=max_batches)["accuracy"].iloc[0])
    elif isinstance(baseline, pd.DataFrame):
        base_acc = float(baseline["accuracy"].iloc[0])
    else:
        base_acc = float(baseline)

    if not 0 < base_acc <= 1:
        raise ValueError(f"baseline accuracy {base_acc} is not a fraction in "
                         "(0, 1] -- pass 0.95, not 95")

    # each model 
    rows = []
    for name, src in models.items():
        if isinstance(src, nn.Module):
            if loader is None or device is None or Ts is None:
                raise ValueError(f"loader, device and Ts are required to "
                                 f"evaluate '{name}'")
            got = accuracy_vs_T(src, loader, list(Ts), device,
                                spiking=True, max_batches=max_batches)
        elif isinstance(src, pd.DataFrame):
            if "accuracy" not in src.columns:
                raise ValueError(f"'{name}' has no 'accuracy' column -- run "
                                 "run_benchmark with measure_accuracy=True")
            got = src[["T", "accuracy"]].copy()
        else:
            got = pd.DataFrame({"T": [np.nan], "accuracy": [float(src)]})

        for _, r in got.sort_values("T").iterrows():
            acc = float(r["accuracy"])
            ratio = acc / base_acc
            rows.append({
                "model": name,
                "T": r["T"],
                "accuracy": acc,
                "baseline_accuracy": base_acc,
                "retention": ratio,
                "retention_pct": 100 * ratio,
                "delta_pp": 100 * (acc - base_acc),
                "delta_rel_pct": 100 * (ratio - 1),
                "label": _label(ratio),
            })

    df = pd.DataFrame(rows)
    df.attrs["baseline_accuracy"] = base_acc
    return df


def best_retention(df: pd.DataFrame) -> pd.DataFrame:
    """The strongest operating point for each model."""
    return (
        df.sort_values("retention", ascending=False)
              .groupby("model", as_index=False).first()
              .sort_values("retention", ascending=False)
    )


def plot_accuracy_retention(
    df: pd.DataFrame,
    annotate: bool = True,
    at_T: Optional[int] = None,
    target: float = 0.95
):
    """
    1. retention vs T'
    2. horizontal bars at one T, coloured by direction
    """
    base = df.attrs.get("baseline_accuracy", df["baseline_accuracy"].iloc[0])
    swept = df[df["T"].notna()]

    # 1. retention vs T
    if len(swept):
        all_T = sorted(swept["T"].unique().astype(int))
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, (name, g) in enumerate(swept.groupby("model")):
            g = g.sort_values("T")
            ax.plot(g["T"], g["retention"], "o-", lw=2, ms=7,
                    color=f"C{i}", label=name)
            if annotate:
                for _, r in g.iterrows():
                    ax.annotate(r["label"], (r["T"], r["retention"]),
                                textcoords="offset points", xytext=(0, 9),
                                ha="center", fontsize=7.5, color=f"C{i}")

        ax.axhline(1.0, color="k", ls="--", lw=2,
                   label=f"ANN baseline ({100 * base:.2f}%)")
        ax.axhline(target, color="red", ls=":", lw=1.5,
                   label=f"{100 * target:.0f}% retention")
        ax.fill_between([min(all_T), max(all_T)], 1.0, ax.get_ylim()[1],
                        color="green", alpha=0.06)

        ax.set_xscale("log", base=2)
        ax.set_xticks(all_T); ax.set_xticklabels(all_T)
        ax.set_xlabel("Timesteps (T)")
        ax.set_ylabel(r"Accuracy retention  ($Acc_{SNN}/Acc_{ANN}$)")
        ax.set_title("Accuracy retention vs timesteps")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout(); plt.show()

    # 2. bars at one operating point
    if at_T is None and len(swept):
        at_T = int(swept.loc[swept["retention"].idxmax(), "T"])
    snap = df[df["T"] == at_T] if at_T is not None else df
    if not len(snap):
        return

    snap = snap.sort_values("retention")
    idx = np.arange(len(snap))
    colours = ["#2a9d5c" if v >= 1 else "#c0392b" for v in snap["retention"]]

    fig, ax = plt.subplots(figsize=(8.5, 0.8 * len(snap) + 2.4))
    ax.barh(idx, snap["retention"], color=colours, alpha=0.85)
    ax.axvline(1.0, color="k", ls="--", lw=2)
    for i, (_, r) in enumerate(snap.iterrows()):
        ax.annotate(f"  {r['label']}   acc {100 * r['accuracy']:.2f}%",
                    (r["retention"], i), va="center", fontsize=9)
    ax.set_yticks(idx); ax.set_yticklabels(snap["model"], fontsize=9)
    ax.set_xlabel(r"Accuracy retention  ($Acc_{SNN}/Acc_{ANN}$)")
    ax.set_xlim(0, max(1.15, snap["retention"].max() * 1.35))
    ax.set_title(f"Retention against the ANN "
                 f"({100 * base:.2f}%)" + (f", T = {at_T}" if at_T else ""))
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout(); plt.show()


def plot_retention_vs_energy(
    df: pd.DataFrame,
    summaries: Dict[str, pd.DataFrame],
    E_ann: float
):
    """
    Retention against analytical energy. The useful region is upper-left:
    retention near 1 at energy below the ANN's.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i, (name, s) in enumerate(summaries.items()):
        g = df[df["model"] == name].merge(
            s[["T", "E_snn"]], on="T").sort_values("T")
        ax.plot(g["E_snn"], g["retention"], "o-", lw=2, ms=7,
                color=f"C{i}", label=name)
        for _, r in g.iterrows():
            ax.annotate(f"T={int(r['T'])}", (r["E_snn"], r["retention"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.axhline(1.0, color="k", ls="--", lw=2, label="ANN accuracy")
    ax.axvline(E_ann, color="C3", ls="--", lw=2, label="ANN energy")
    ax.set_xscale("log")
    ax.set_xlabel("Analytical energy per inference (J)")
    ax.set_ylabel(r"Accuracy retention  ($Acc_{SNN}/Acc_{ANN}$)")
    ax.set_title("Retention vs energy (upper-left quadrant is the win)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); plt.show()