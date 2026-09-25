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
from .types import LayerSpec, SYNAPTIC_TYPES, NEURON_TYPES
from ..config import DEVICE

from spikingjelly.activation_based import neuron, functional

def count_activity(
    model: nn.Module, specs: List[LayerSpec], loader, T: int,
    device: Union[torch.device, str] = DEVICE, max_batches: Optional[int] = None
) -> List[LayerSpec]:
    """
    Per sample, summed over all T timesteps:
      theta_in  = number of NONZERO input elements (spike events)
      theta_out = number of spikes emitted by the following neuron
 
    For the analog encoder layer theta_in is the value sum; it is not a spike
    count and is not consumed by the dense path, kept only for reference.
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
            ev_out[name] = ev_out.get(name, 0.0) + float((t.detach() != 0).sum().item())
        return hook
 
    wanted = {s.neuron_name for s in specs if s.neuron_name}
    for name, mod in model.named_modules():
        if name in by_name and isinstance(mod, SYNAPTIC_TYPES):
            hooks.append(mod.register_forward_hook(
                syn_hook(name, by_name[name].analog_input)))
        elif name in wanted and isinstance(mod, NEURON_TYPES):
            hooks.append(mod.register_forward_hook(neu_hook(name)))
 
    model.eval()
    n = 0
    with torch.no_grad():
        for b, (x, _) in enumerate(loader):
            if max_batches is not None and b >= max_batches:
                break
            x = x.to(device)
            functional.reset_net(model)
            for _ in range(T):
                model(x)
            n += x.size(0)
    for h in hooks:
        h.remove()
    functional.reset_net(model)
 
    n = max(n, 1)
    for s in specs:
        s.theta_in = ev_in[s.name] / n
        s.theta_out = ev_out.get(s.neuron_name, 0.0) / n if s.neuron_name else 0.0
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
 