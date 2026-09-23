from typing import Callable, Optional, Union, Iterable, Tuple, Dict, Protocol, runtime_checkable
import torch
from torch.optim import Optimizer

from .train_ann import train_ann
from .eval_ann import eval_ann
from .snn import SNN          # example class
from .train_snn import train_snn
from .eval_snn import eval_snn
from .ann import ANN          # example class

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
    "train_ann",
    "eval_ann",
    "train_snn",
    "eval_snn",
    "ANN",
    "ResNet50Modified",
    "ResNet18Modified",
    "SNN",
    "CriterionType",
    "OptimizerFactory"
]

