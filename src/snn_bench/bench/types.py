from dataclasses import dataclass, field
from typing import Optional
from enum import Enum 
import pandas as pd

import torch.nn as nn
from spikingjelly.activation_based import neuron

from enum import Enum
from typing import Iterable, List, Optional, Union


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
SYNAPTIC_TYPES = (nn.Conv2d, nn.Linear)
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


class BenchLevel(Enum):
    LEVEL_ONE   = 1   # computational activity [measured on any hardware]
    LEVEL_TWO   = 2   # analytical energy [Lemaire equations + constants]
    LEVEL_THREE = 3   # empirical measurement [NVML / psutil on the host]
    LEVEL_ALL   = 4

@dataclass
class BenchResult:
    level: BenchLevel
    result: pd.DataFrame


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