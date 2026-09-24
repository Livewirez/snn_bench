from typing import Callable, Optional, Union, Iterable, Tuple, Dict, Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch.optim import Optimizer
import spikingjelly
from spikingjelly.activation_based import neuron, layer, functional, surrogate, encoding


from abc import ABC, abstractmethod


SpikingJellyEncoder = Union[
    spikingjelly.activation_based.encoding.StatelessEncoder,
    spikingjelly.activation_based.encoding.StatefulEncoder,
]


class SNNModule(ABC, nn.Module):
    def reset_state(self):
        functional.reset_net(self)      # resets all neurons inside the model

    @abstractmethod
    def forward(self, x: torch.Tensor, /) -> torch.Tensor:
        """Return logits (e.g. firing rate averaged over T)."""
        ...
    
    @abstractmethod
    def get_output_layer(self) -> neuron.BaseNode: ...

@runtime_checkable
class CriterionProtocol(Protocol):
    # The "/" makes the parameters positional-only. Without it, mypy/pyright
    # require implementations to use the exact names `pred` and `target`,
    def __call__(self, pred: torch.Tensor, target: torch.Tensor, /) -> torch.Tensor: ...


CriterionType = CriterionProtocol

OptimizerFactory = Callable[
    [Iterable[torch.nn.Parameter]],
    Optimizer
]

__all__ = [
    "CriterionProtocol",
    "CriterionType",
    "OptimizerFactory",
    "SpikingJellyEncoder",
    "SNNModule"
]
