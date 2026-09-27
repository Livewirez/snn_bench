from dataclasses import dataclass, field
from typing import Optional
from enum import Enum 
import pandas as pd

import math
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron

from enum import Enum
from typing import Iterable, List, Optional, Union, Protocol, runtime_checkable, Type, Any, Dict, Tuple

from .level2 import _standard_memory_words, _dense_conv_counts, _dense_fc_counts


# ===================================================================================================
# ENERGY CONSTANTS  -- Ten Lessons From Three Generations Shaped Google’s TPUv4i : Industrial Product
# ===================================================================================================
E_ADD_INT32  = 0.1e-12
E_MULT_INT32 = 3.1e-12
E_ADD_FP32   = 0.9e-12
E_MULT_FP32  = 3.7e-12
 
_SRAM_POINTS = [(8.0, 10e-12), (32.0, 20e-12), (1024.0, 100e-12)]  # (kB, J)
BYTES_PER_WORD = 4


# MAC = Multiply-Accumulate. Multiply two numbers, add the result to a running total:
# total += weight * input

# ACC = ACCumulate. only addition. no multiply:
# total += weight

@dataclass
class EnergyModel:
    e_add: float = E_ADD_INT32
    e_mult: float = E_MULT_INT32
    label: str = "int32"
 
    @property
    def e_acc(self) -> float: # A MAC is "multiply-accumulate": multiply two numbers, then add the result to a running total.
        return self.e_add
 
    @property
    def e_mac(self) -> float:  # A MAC is "multiply-accumulate": multiply two numbers, then add the result to a running total.
        return self.e_add + self.e_mult
 
 
INT32_MODEL = EnergyModel(E_ADD_INT32, E_MULT_INT32, "int32")
FP32_MODEL  = EnergyModel(E_ADD_FP32,  E_MULT_FP32,  "fp32")

 
# ======================================================================
# TOPOLOGY
# ======================================================================
SYNAPTIC_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)
NEURON_TYPES   = (neuron.BaseNode,)
 
# Cin, Hin, Win, Cout, Hout, Wout, Hk, Wk, S    # conv geometry
# Nin, Nout                                     # fc geometry

# Describes one weighted layer: one Conv2d or one Linear.
# For Lemaire's Equation: Eq. 1 needs Cout·Hk·Wk for the accumulation count;
# Eq. 7 needs the same for parameter reads; Eq. 16 needs it for addressing.
@dataclass
class LayerSpec:
    name: str
    kind: str # 'conv' | 'fc'
    exec_order: int
    has_bias: bool
 
    Cin: int = 0; Hin: int = 0; Win: int = 0
    Cout: int = 0; Hout: int = 0; Wout: int = 0
    Hk: int = 1; Wk: int = 1; S: int = 1
    Nin: int = 0
    Nout: int = 0
 
    neuron_name: Optional[str] = None
    neuron_count: int = 0 # neuron_count is neurons per sample, which is the denominator of the Davidson & Furber metric.
    neuron_is_lif: bool = False
 
    theta_in: float = 0.0           # spike EVENTS arriving, summed over T
    theta_out: float = 0.0          # spike EVENTS emitted by the linked neuron
    analog_input: bool = False      # True => encoder layer, dense MACs
    input_scale: float = 1.0        # VoltageScaler lambda, if detected
    
    extra: Dict[str, Any] = field(default_factory=dict)
 
    @property
    def n_out_positions(self) -> int:
        return self.Cout * self.Hout * self.Wout if self.kind == "conv" else self.Nout
 
    @property
    def n_weights(self) -> int:
        if self.kind == "conv":
            return self.Cout * self.Cin * self.Hk * self.Wk
        return self.Nin * self.Nout
 
    @property
    def n_bias(self) -> int:
        return (self.Cout if self.kind == "conv" else self.Nout) if self.has_bias else 0
 
    @property
    def n_in_elements(self) -> int:
        return self.Cin * self.Hin * self.Win if self.kind == "conv" else self.Nin


@runtime_checkable
class LayerHandler(Protocol):
    """
    A handler must supply all three methods,
    because geometry alone is not enough: example Conv1d and Conv3d need different
    fan-out and addressing expressions, not just different shape fields.
    """

    kind: str  # value written to LayerSpec.kind
    module_types: Tuple[type, ...]  # what this handler claims

    def build_spec(
        self, name: str, module: nn.Module, idx: int,
        x: torch.Tensor, out: torch.Tensor
    ) -> LayerSpec:
        """Called once, from the profiling hook."""

    def snn_counts(self, s: LayerSpec, T: int) -> Dict[str, float]:
        """Spiking operation, memory and addressing counts (Lemaire Eq. 1-17)."""

    def ann_counts(self, s: LayerSpec) -> Dict[str, float]:
        """Formal-network counterpart."""

    def memory_words(self, s: LayerSpec, spiking: bool) -> int:
        """Layer footprint in 32-bit words, for the SRAM interpolation."""
        
        
class HandlerRegistry:
    def __init__(self, handlers: Optional[Iterable[LayerHandler]] = None):
        self._by_type: Dict[type, LayerHandler] = {}
        self._by_kind: Dict[str, LayerHandler] = {}
        for h in (handlers or []):
            self.register(h)

    def register(self, handler: LayerHandler) -> HandlerRegistry:
        for t in handler.module_types:
            self._by_type[t] = handler
        self._by_kind[handler.kind] = handler
        return self

    def for_module(self, m: nn.Module) -> Optional[LayerHandler]:
        # exact type first, then MRO, so subclasses inherit a handler
        h = self._by_type.get(type(m))
        if h is not None:
            return h
        for base in type(m).__mro__[1:]:
            if base in self._by_type:
                return self._by_type[base]
        return None

    def for_spec(self, s: LayerSpec) -> LayerHandler:
        h = self._by_kind.get(s.kind)
        if h is None:
            raise KeyError(f"no handler registered for kind '{s.kind}'")
        return h

    @property
    def module_types(self) -> Tuple[type, ...]:
        return tuple(self._by_type)


class BenchLevel(Enum):
    LEVEL_ONE   = 1   # computational activity [measured on any hardware]
    LEVEL_TWO   = 2   # analytical energy [Lemaire equations + constants]
    LEVEL_THREE = 3   # empirical measurement [NVML / psutil on the host]
    LEVEL_ALL   = 4


# The columns present in every view, regardless of level.
IDENTITY_COLUMNS: List[str] = ["model", "T", "accuracy"]

# Level 1 holds only what was measured by running the network. Nothing here
# depends on a cost model, so these values are unchanged if the energy
# constants or the target process node are altered. (Hardware Independent)
LEVEL_ONE_COLUMNS: List[str] = [
    "spikes", "neurons", "spikes_per_neuron",
]

# Level 2 holds everything derived from the Lemaire equations: operation and
# memory-access counts as well as the energies computed from them. They follow from the equations applied
# to the measured activity, not from the measurement itself.
LEVEL_TWO_COLUMNS: List[str] = [
    "mac", "acc",
    "mem_reads", "mem_writes", "mem_accesses",
    "accesses_per_mac", "accesses_per_spike", "mem_amplification",
    "E_mem", "E_ops", "E_addr", "E_total",
    "pct_E_mem", "mean_pJ_per_access", "E_ratio_ann_over_model",
]

# Level 3 holds host measurements. These describe a general-purpose GPU running
# dense kernels and are not comparable with Level 2 on the same axis.
LEVEL_THREE_COLUMNS: List[str] = [
    "time_per_sample_ms",
    "gpu_energy_per_sample_J", "gpu_energy_marginal_per_sample_J",
    "avg_gpu_power_W", "power_spread_W", "idle_power_W",
    "gpu_peak_mem_MB", "cpu_util_pct", "energy_source",
]

_LEVEL_MAP = {
    BenchLevel.LEVEL_ONE:   LEVEL_ONE_COLUMNS,
    BenchLevel.LEVEL_TWO:   LEVEL_TWO_COLUMNS,
    BenchLevel.LEVEL_THREE: LEVEL_THREE_COLUMNS,
}

_SUMMARY_L3 = {
    "gpu_energy_per_sample_J":          "gpu_energy_per_sample_J",
    "gpu_energy_marginal_per_sample_J": "gpu_energy_marginal_per_sample_J",
}

@dataclass
class BenchColumMapper:
    id_cols: List[str] = field(default_factory=lambda: IDENTITY_COLUMNS)
    level_one_cols: List[str] = field(default_factory=lambda: LEVEL_ONE_COLUMNS)
    level_two_cols: List[str] = field(default_factory=lambda: LEVEL_TWO_COLUMNS)
    level_three_cols: List[str] = field(default_factory=lambda: LEVEL_THREE_COLUMNS)
    
    
class Conv2dHandler:
    kind = "conv"
    module_types = (nn.Conv2d,)

    def build_spec(self, name, m, idx, x, out):
        return LayerSpec(
            name=name, kind=self.kind, exec_order=idx,
            has_bias=m.bias is not None,
            Cin=x.shape[1], Hin=x.shape[2], Win=x.shape[3],
            Cout=out.shape[1], Hout=out.shape[2], Wout=out.shape[3],
            Hk=m.kernel_size[0], Wk=m.kernel_size[1],
            Sh=m.stride[0], Sw=m.stride[1]
        )

    def snn_counts(self, s, T):
        if s.analog_input:
            return _dense_conv_counts(s, repeats=T, is_snn_encoder=True)
        
        th_in, th_out = s.theta_in, s.theta_out
        fan = s.Cout * s.Hk * s.Wk
        return {
            "mac": T * s.n_out_positions if s.neuron_is_lif else 0.0,
            "acc": (th_in * math.ceil(s.Hk / s.Sh) * math.ceil(s.Wk / s.Sw)
                    * s.Cout
                    + (T * s.n_out_positions if s.has_bias else 0.0)
                    + th_out),
            "rd": (th_in + th_in * fan
                   + (s.n_out_positions if s.has_bias else 0.0)
                   + th_in * fan + s.n_out_positions),
            "wr": th_out + th_in * fan + s.n_out_positions,
            "mac_addr": th_in * 2,
            "acc_addr": th_in * fan,
        }

    def ann_counts(self, s):
        return _dense_conv_counts(s)

    def memory_words(self, s, spiking):
        return _standard_memory_words(s, spiking)
    
    
class Conv3dHandler:
    kind = "conv3d"
    module_types = (nn.Conv3d,)

    def build_spec(self, name, m, idx, x, out):
        s = LayerSpec(
            name=name, kind=self.kind, exec_order=idx,
            has_bias=m.bias is not None,
            Cin=x.shape[1], Hin=x.shape[3], Win=x.shape[4],
            Cout=out.shape[1], Hout=out.shape[3], Wout=out.shape[4],
            Hk=m.kernel_size[1], Wk=m.kernel_size[2],
            Sh=m.stride[1], Sw=m.stride[2]
        )
        s.extra.update(Din=x.shape[2], Dout=out.shape[2], Dk=m.kernel_size[0], Sd=m.stride[0])
        return s

    def snn_counts(self, s, T):
        e = s.extra
        n_out = s.Cout * e["Dout"] * s.Hout * s.Wout
        fan = s.Cout * e["Dk"] * s.Hk * s.Wk
        th_in, th_out = s.theta_in, s.theta_out
        return {
            "mac": T * n_out if s.neuron_is_lif else 0.0,
            "acc": (th_in * math.ceil(e["Dk"] / e["Sd"])
                    * math.ceil(s.Hk / s.Sh) * math.ceil(s.Wk / s.Sw) * s.Cout
                    + (T * n_out if s.has_bias else 0.0) + th_out),
            "rd": th_in + 2 * th_in * fan + n_out
                  + (n_out if s.has_bias else 0.0),
            "wr": th_out + th_in * fan + n_out,
            "mac_addr": th_in * 3,          # three coordinates
            "acc_addr": th_in * fan,
        }
        
    def ann_counts(self, s):
        return _dense_conv_counts(s)

    def memory_words(self, s, spiking):
        return _standard_memory_words(s, spiking)
    
class LinearHandler:
    """
    Fully connected layers. Lemaire Eq. 2, 5, 8, 10, 13, 15, 17 for the
    spiking case; Eq. 4, 6, 12, 17 for the formal case.

    One spike arriving at a fully connected layer reaches every one of the
    N_out targets, so the fan-out factor is N_out. No coordinate decoding is
    required, since the arriving index is already the address, which is why
    mac_addr is zero here but not for convolution.
    """

    kind = "fc"
    module_types = (nn.Linear,)
    
    def __init__(self, fc_acc_literal: bool = False):
        self.fc_acc_literal = fc_acc_literal

    def build_spec(self, name: str, m: nn.Module, idx: int, x: torch.Tensor, out: torch.Tensor) -> LayerSpec:
        # in_features / out_features are safe to read directly: a LazyLinear
        # has materialised by the time any hook fires.
        return LayerSpec(
            name=name, kind=self.kind, exec_order=idx,
            has_bias=m.bias is not None,
            Nin=m.in_features, Nout=m.out_features,
        )

    def snn_counts(self, s: LayerSpec, T: int) -> Dict[str, float]:
        if s.analog_input:
            # Static frame-based encoding: dense multiply-accumulates, T times.
            return _dense_fc_counts(s, repeats=T, is_snn_encoder=True)

        th_in, th_out = s.theta_in, s.theta_out

        # Eq. 2 as printed gives theta * Nin * Nout, which contradicts Eq. 8
        # (theta * Nout + Nout). Since theta is a count throughout, the Nin
        # factor double-counts; theta * Nout is used by default.
        fan = (s.Nin * s.Nout) if self.fc_acc_literal else s.Nout

        return {
            # Eq. 2 - membrane leak only for leaky neurons
            "mac": T * s.Nout if s.neuron_is_lif else 0.0,
            "acc": (th_in * fan + (T * s.Nout if s.has_bias else 0.0) + th_out),                       # soft reset
            # Eq. 5 (input FIFO) + Eq. 8 (weights, bias) + Eq. 10 (potentials)
            "rd": (th_in
                   + th_in * s.Nout + (s.Nout if s.has_bias else 0.0)
                   + (th_in + 1) * s.Nout),
            # Eq. 13 (output spikes) + Eq. 15 (potentials)
            "wr": th_out + th_in * s.Nout + s.Nout,
            # Eq. 17
            "mac_addr": 0.0,
            "acc_addr": th_in * s.Nout,
        }


    def ann_counts(self, s: LayerSpec) -> Dict[str, float]:
        return _dense_fc_counts(s)
    
    def memory_words(self, s: LayerSpec, spiking: bool) -> int:
        return _standard_memory_words(s, spiking)
    
    def n_out_positions(self, s: LayerSpec) -> int:
        return s.Nout

    def n_weights(self, s: LayerSpec) -> int:
        return s.Nin * s.Nout

    def n_bias(self, s: LayerSpec) -> int:
        return s.Nout if s.has_bias else 0

    def n_in_elements(self, s: LayerSpec) -> int:
        return s.Nin
    
DEFAULT_REGISTRY = HandlerRegistry([Conv2dHandler(), LinearHandler()])