"""
snn_bench -- a modular framework for benchmarking accuracy retention and energy
efficiency in ANN-to-SNN conversion.

MSc Data Science dissertation (unit 6G7V0007): "Evaluating Accuracy Retention
and Energy Efficiency in ANN to SNN Conversion".

Components (one module each, independently testable):
    data        -- dataset ingestion (MNIST, FashionMNIST, ...), ImageDataset 
    models      -- [
        ann         -- ANN model + training/evaluation
        direct_snn  -- direct SNN training with surrogate gradients (SpikingJelly)
    ]
    convert     -- ANN -> SNN conversion (rate coding, max / 99.9% normalisation)
    evaluate    -- accuracy + per-layer spike counting
    energy      -- analytical energy (Lemaire AC/MAC + memory) + Davidson-Furber check
    plots       -- result visualisation (accuracy-vs-latency, spikes, energy)
"""

__version__ = "0.1.64"


from typing import Union
import torch

Device = Union[torch.device, str]

from . import data, benchmark, config, constants  # always importable (pure torch)

# direct_snn and convert need SpikingJelly; import lazily so the package still
# loads on a machine without it.
try:
    from . import models, conversion_mod, conversion, evaluate, train, evaluate    # noqa: F401
    _SPIKINGJELLY = True
except Exception as err:  # pragma: no cover - depends on optional dependency
    print(f"Unexpected {err=}, {type(err)=}")
    _SPIKINGJELLY = False
    raise

__all__ = [
    "data", "benchmark", "config", "constants", "models", "train", "evaluate",
    "conversion_mod", "conversion", "evaluate", "train",  "__version__",
    "Device"
]





