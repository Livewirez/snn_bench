from typing import Callable, Optional, Union, Iterable, Tuple, Dict, Protocol, runtime_checkable

import torch
from torch.optim import Optimizer


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
]
