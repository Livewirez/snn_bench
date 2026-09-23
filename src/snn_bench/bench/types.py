from dataclasses import dataclass
from typing import Optional

import torch.nn as nn
from spikingjelly.activation_based import neuron


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
 