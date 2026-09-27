from __future__ import annotations

import math
import time
import threading
import copy
 
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..config import DEVICE
from typing import Dict, List, Optional, Tuple, Union
from . import sram_access_energy, profile_topology
from .types import LayerSpec, EnergyModel, SYNAPTIC_TYPES, NEURON_TYPES, INT32_MODEL,BYTES_PER_WORD, E_ADD_FP32

from spikingjelly.activation_based import neuron, functional

def snn_layer_counts(s: LayerSpec, T: int, fc_acc_literal: bool = False) -> Dict[str, float]:
    th_in, th_out = s.theta_in, s.theta_out
 
    if s.analog_input:
        # Static frame-based encoding: analog input repeated T times => dense MACs
        return _dense_counts(s, repeats=T, is_snn_encoder=True)
 
    if s.kind == "conv":
        mac = T * s.n_out_positions if s.neuron_is_lif else 0.0 # Eq. 1 (Lemaire)
        acc = (th_in * math.ceil(s.Hk / s.S) * math.ceil(s.Wk / s.S) * s.Cout
               + (T * s.n_out_positions if s.has_bias else 0.0)
               + th_out) # Eq. 1 (Lemaire)
        rd = (th_in
              + th_in * s.Cout * s.Wk * s.Hk # weights
              + (s.n_out_positions if s.has_bias else 0.0) # bias
              + th_in * s.Cout * s.Wk * s.Hk + s.n_out_positions) # Eq. 7, 8 — parameter reads  # Eq. 9
        wr = th_out + th_in * s.Cout * s.Wk * s.Hk + s.n_out_positions
        mac_addr = th_in * 2
        acc_addr = th_in * s.Cout * s.Hk * s.Wk
    else:
        mac = T * s.Nout if s.neuron_is_lif else 0.0 # Eq. 1 (Lemaire)
        fan = (s.Nin * s.Nout) if fc_acc_literal else s.Nout # Eq. 2 fc synaptic operations, SNN (Lemaire)
        acc = th_in * fan + (T * s.Nout if s.has_bias else 0.0) + th_out
        rd = (th_in
              + th_in * s.Nout + (s.Nout if s.has_bias else 0.0)
              + (th_in + 1) * s.Nout)  # Eq. 5, 6  input reads SNN (Lemaire)
        wr = th_out + th_in * s.Nout + s.Nout
        mac_addr = 0.0
        acc_addr = th_in * s.Nout
 
    return {
        "mac": mac, "acc": acc, "rd": rd, "wr": wr,
        "mac_addr": mac_addr, "acc_addr": acc_addr
    }
 
 
def _dense_counts(s: LayerSpec, repeats: int = 1, is_snn_encoder: bool = False) -> Dict[str, float]:
    if s.kind == "conv":
        mac = s.n_out_positions * s.Cin * s.Hk * s.Wk  # Eq. 3, 4 synaptic operations, FNN (Lemaire) # conv
        acc = s.n_out_positions if s.has_bias else 0.0
        rd = (s.Cin * s.Cout * s.Hout * s.Wout * s.Wk * s.Hk
              + (s.Cin * s.Wk * s.Hk + (1 if s.has_bias else 0))
                * s.Cout * s.Wout * s.Hout)
        wr = s.n_out_positions
        mac_addr = 0.0
        acc_addr = (s.Cin * s.Hin * s.Win + s.n_out_positions
                    + s.Cout * s.Hk * s.Wk)
    else:
        mac = s.Nin * s.Nout # Eq. 3, 4 synaptic operations, FNN (Lemaire) # fc
        acc = s.Nout if s.has_bias else 0.0
        rd = s.Nin + (s.Nin + (1 if s.has_bias else 0)) * s.Nout  # Eq. 5, 6  input reads SNN (Lemaire) FNN conv
        wr = s.Nout
        mac_addr = 0.0
        acc_addr = s.Nin + s.Nout
    out = {
        "mac": mac, "acc": acc, "rd": rd, "wr": wr,
        "mac_addr": mac_addr, "acc_addr": acc_addr
    }
    if repeats != 1:
        out = {k: v * repeats for k, v in out.items()}
        if is_snn_encoder:
            out["rd"] += repeats * s.n_out_positions
            out["wr"] += repeats * s.n_out_positions
    return out
 
 
def ann_layer_counts(s: LayerSpec) -> Dict[str, float]:
    return _dense_counts(s, repeats=1)
 
 
def layer_memory_kB(s: LayerSpec, spiking: bool) -> float:
    words = s.n_weights + s.n_bias
    if spiking:
        words += s.n_out_positions
        words += max(s.n_in_elements // 8, 64)
    else:
        words += s.n_in_elements + s.n_out_positions
    return words * BYTES_PER_WORD / 1024.0
 
 
def counts_to_energy(counts: Dict[str, float], mem_kB: float, em: EnergyModel) -> Dict[str, float]:
    e_access = sram_access_energy(mem_kB)
    E_mem = (counts["rd"] + counts["wr"]) * e_access
    E_ops = counts["mac"] * em.e_mac + counts["acc"] * em.e_acc
    E_addr = counts["mac_addr"] * em.e_mac + counts["acc_addr"] * em.e_acc
    return {
        "E_mem": E_mem, "E_ops": E_ops, "E_addr": E_addr,
        "E_total": E_mem + E_ops + E_addr,
        "mem_kB": mem_kB, "e_access_J": e_access, **counts
    }

def simple_proxy_energy(total_spikes: float, total_ops: float, e_spike: float = 0.0, e_synapse: float = E_ADD_FP32) -> float:
    """
    Eq. 4:  E_total = E_spike * S + E_synapse * C

    The two-term proxy common in SNN literature. Assumes a constant cost per
    operation, ignoring that SRAM access energy scales with memory size.
    Reported alongside the Lemaire model to quantify that simplification.
    """
    return e_spike * total_spikes + e_synapse * total_ops
 
def analytical_energy(specs: List[LayerSpec], T: int, em: EnergyModel = INT32_MODEL, fc_acc_literal: bool = False):
    rows = []
    keys = ["E_mem", "E_ops", "E_addr", "E_total", "mac", "acc", "rd", "wr"]
    snn_tot = {k: 0.0 for k in keys}
    ann_tot = {k: 0.0 for k in keys}
 
    for s in specs:
        sc = snn_layer_counts(s, T, fc_acc_literal=fc_acc_literal)
        ac = ann_layer_counts(s)
        se = counts_to_energy(sc, layer_memory_kB(s, True), em)
        ae = counts_to_energy(ac, layer_memory_kB(s, False), em)
        for k in keys:
            snn_tot[k] += se[k]; ann_tot[k] += ae[k]
        rows.append({
            "layer": s.name, "kind": s.kind, "analog_in": s.analog_input,
            "input_scale": s.input_scale,
            "theta_in": s.theta_in, "theta_out": s.theta_out,
            "neurons": s.neuron_count,
            "firing_rate": s.theta_out / (s.neuron_count * T) if s.neuron_count else float("nan"),
            "snn_mac": sc["mac"], "snn_acc": sc["acc"],
            "snn_rd": sc["rd"], "snn_wr": sc["wr"],
            "snn_E_mem": se["E_mem"], "snn_E_ops": se["E_ops"],
            "snn_E_addr": se["E_addr"], "snn_E_total": se["E_total"],
            "snn_mem_kB": se["mem_kB"], "snn_pJ_per_access": se["e_access_J"] * 1e12,
            "ann_mac": ac["mac"], "ann_E_total": ae["E_total"],
            "ann_mem_kB": ae["mem_kB"],
            "ratio_ann_over_snn": ae["E_total"] / se["E_total"] if se["E_total"] else float("nan"),
        })
 
    snn_tot["ratio_ann_over_snn"] = (
        ann_tot["E_total"] / snn_tot["E_total"] if snn_tot["E_total"] else float("nan")
    )
    df = pd.DataFrame(rows)
    if len(df):
        df["snn_pct_of_total"] = 100 * df["snn_E_total"] / snn_tot["E_total"]
    return snn_tot, ann_tot, df
 
 
def ann_baseline(
    ann: nn.Module, example_input: torch.Tensor, device: Union[torch.device, str] = DEVICE,
    em: EnergyModel = INT32_MODEL
) -> Tuple[Dict, pd.DataFrame]:
    """
    Analytical energy of the non-spiking ANN.  Use THIS for the baseline --
    never pass an ANN to run_benchmark(), which would treat every layer as an
    encoder and report the ANN executed T times.
    """
    specs = profile_topology(ann, example_input, device, probe_T=1)
    keys = ["E_mem", "E_ops", "E_addr", "E_total", "mac", "acc", "rd", "wr"]
    rows, tot = [], {k: 0.0 for k in keys}
    for s in specs:
        ac = ann_layer_counts(s)
        ae = counts_to_energy(ac, layer_memory_kB(s, False), em)
        for k in keys:
            tot[k] += ae[k]
        rows.append({
            "layer": s.name, "kind": s.kind, "mac": ac["mac"],
            "rd": ac["rd"], "wr": ac["wr"], "mem_kB": ae["mem_kB"],
            "pJ_per_access": ae["e_access_J"] * 1e12,
            "E_mem": ae["E_mem"], "E_ops": ae["E_ops"],
            "E_total": ae["E_total"]
        })
    df = pd.DataFrame(rows)
    df["pct_of_total"] = 100 * df["E_total"] / tot["E_total"]
    return tot, df
 