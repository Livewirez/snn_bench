"""For an SNN to be more energy-efficient than the corresponding ANN on digital hardware,    
the average number of spikes per neuron per inference must stay below approximately 1.72.

Lemaire: An Analytical Esimation "The energy consumption of single operations 
(addition, multiplication and memory accesses) are drawn from the literature [15] for 45nm CMOS technology. 
For addition and multiplication with 32-bit integers, we use respectively 0.1pJ and 3.1pJ. For SRAM memory accesses, 
we compute a linear interpolation function based on 3 particular values: 8 kB (10pJ), 32 kB (20pJ) and 1 MB (100pJ). 
This function enables to compute the energy cost of a memory access knowing the memory size (i.e. knowing the network hyper-parameters)."

"""




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

from typing import Dict, List, Optional, Tuple, Union
from .types import LayerSpec, HandlerRegistry, SYNAPTIC_TYPES, NEURON_TYPES, DEFAULT_REGISTRY
from ..config import DEVICE

from spikingjelly.activation_based import neuron, functional

def count_activity(
    model: nn.Module,
    specs: List[LayerSpec],
    loader,
    T: int,
    device: Union[torch.device, str] = DEVICE,
    max_batches: Optional[int] = None,
    registry: HandlerRegistry = DEFAULT_REGISTRY,
    strict: bool = True,
) -> List[LayerSpec]:
    """
    Level 1 measurement. Per sample, summed over all T timesteps:

      theta_in  = number of NONZERO input elements (spike events)
      theta_out = number of spikes emitted by the following neuron

    Counting events rather than summing tensor values is required because
    SpikingJelly's conversion inserts VoltageScaler modules, so spike tensors
    take values in {0, lambda} rather than {0, 1}. Summing inflates theta by
    lambda, measured between 2.18 and 3.04 across the networks evaluated.

    For the analogue encoder layer theta_in is the value sum instead. It is not
    a spike count and is not consumed by the dense path; it is retained only
    for reference.

    The returned list is a deep copy, so theta values from one value of T
    cannot leak into the next when sweeping.
    """
    specs = copy.deepcopy(specs)
    by_name = {s.name: s for s in specs}
    ev_in = {s.name: 0.0 for s in specs}
    ev_out: Dict[str, float] = {}
    hooks = []

    def syn_hook(name, analog):
        def hook(m, inp, out):
            x = inp[0].detach()
            ev_in[name] += float(x.sum().item() if analog
                                 else (x != 0).sum().item())
        return hook

    def neu_hook(name):
        def hook(m, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            ev_out[name] = ev_out.get(name, 0.0) + float(
                (t.detach() != 0).sum().item())
        return hook

    wanted_neurons = {s.neuron_name for s in specs if s.neuron_name}
    attached_syn, attached_neu = set(), set()

    for name, mod in model.named_modules():
        if name in by_name:
            if registry.for_module(mod) is None:
                raise RuntimeError(
                    f"'{name}' was profiled as a '{by_name[name].kind}' layer "
                    f"but no handler now claims {type(mod).__name__}. The "
                    f"registry passed to count_activity() differs from the one "
                    f"used by profile_topology().")
            hooks.append(mod.register_forward_hook(
                syn_hook(name, by_name[name].analog_input)))
            attached_syn.add(name)
        elif name in wanted_neurons and isinstance(mod, NEURON_TYPES):
            hooks.append(mod.register_forward_hook(neu_hook(name)))
            attached_neu.add(name)

    # Every profiled layer must be instrumented, or its theta stays at zero
    # and the resulting energy figure is silently wrong for that layer.
    missing_syn = set(by_name) - attached_syn
    missing_neu = wanted_neurons - attached_neu
    if (missing_syn or missing_neu) and strict:
        for h in hooks:
            h.remove()
        raise RuntimeError(
            f"topology and activity passes disagree about the model. "
            f"Profiled but not instrumented: layers {sorted(missing_syn)}, "
            f"neurons {sorted(missing_neu)}.")

    # ------------------------------------------------------------------
    # Measure
    # ------------------------------------------------------------------
    model.eval()
    n = 0
    try:
        with torch.no_grad():
            for b, (x, _) in enumerate(loader):
                if max_batches is not None and b >= max_batches:
                    break
                x = x.to(device)
                functional.reset_net(model)
                for _ in range(T):
                    model(x)
                n += x.size(0)
    finally:
        # Hooks must come off even if the loop raises; a forgotten hook slows
        # every subsequent forward pass for the rest of the session.
        for h in hooks:
            h.remove()
        functional.reset_net(model)

    if n == 0:
        raise RuntimeError("no samples were processed; check the dataloader "
                           "and max_batches")

    # Normalise to per sample. theta remains summed over all T timesteps,
    # matching Lemaire's definition; 
    for s in specs:
        s.theta_in = ev_in[s.name] / n
        s.theta_out = (ev_out.get(s.neuron_name, 0.0) / n
                       if s.neuron_name else 0.0)

    return specs
 
def validate_activity(specs: List[LayerSpec], T: int, strict: bool = True):
    """
    Validate Counted Activity
    """
    problems = []
    for s in specs:
        if s.neuron_count:
            rate = s.theta_out / (s.neuron_count * T)
            if rate > 1.0 + 1e-6:
                problems.append(
                    f"{s.name}: firing rate {rate:.3f} > 1 -- theta_out is not "
                    f"an event count")
        if not s.analog_input:
            cap = s.n_in_elements * T
            if s.theta_in > cap * (1 + 1e-6):
                problems.append(
                    f"{s.name}: theta_in {s.theta_in:.0f} exceeds capacity "
                    f"{cap} ({s.n_in_elements} inputs x T={T})")
    for prev, cur in zip(specs[:-1], specs[1:]):
        if cur.analog_input or prev.theta_out == 0:
            continue
        if cur.theta_in > prev.theta_out * (1 + 1e-6):
            problems.append(
                f"{cur.name}: theta_in {cur.theta_in:.0f} > theta_out of "
                f"{prev.name} ({prev.theta_out:.0f}) -- impossible through pooling")
    if problems:
        msg = "Activity validation failed:\n  " + "\n  ".join(problems)
        if strict:
            raise ValueError(msg)
        print("WARNING: " + msg)
    return problems
 
 
def level1_summary(specs: List[LayerSpec], T: int) -> Dict:
    total_spikes = sum(s.theta_out for s in specs)
    total_neurons = sum(s.neuron_count for s in specs)
    return {
        "T": T,
        "total_spikes_per_sample": total_spikes,
        "total_neurons": total_neurons,
        "spikes_per_neuron_per_inference": total_spikes / total_neurons if total_neurons else 0.0,
        "spikes_per_neuron_per_timestep": (total_spikes / total_neurons / T) if (total_neurons and T) else 0.0,
    }
 