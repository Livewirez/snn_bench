"""
ANN2SNN conversion recipes for SpikingJelly (Recipe API).

Implements / extends rate-coding style conversion algorithms from the literature
so they plug into:

    recipe = SomeRecipe(dataloader=calibration_loader, ...)
    snn = ann2snn.FXConverter(recipe=recipe).convert(ann)
"""


from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Union

import torch
import torch.nn as nn
from torch import fx
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# ---------------------------------------------------------------------------
# Optional import of SpikingJelly pieces. When SJ is not installed the file
# still documents the API; at runtime the user must have spikingjelly.
# ---------------------------------------------------------------------------
try:
    from spikingjelly.activation_based import neuron as sj_neuron
    from spikingjelly.activation_based.ann2snn import FXConversionRecipe
    from spikingjelly.activation_based.ann2snn.modules import VoltageScaler
except ImportError:  # pragma: no cover
    FXConversionRecipe = object  # type: ignore
    VoltageScaler = None  # type: ignore
    sj_neuron = None  # type: ignore
    
    
    
def _default_if_factory(scale: float) -> nn.Module:
    """
    Build the classic rate-coding block:
        VoltageScaler(1/s) -> IFNode(v_threshold=1.0, soft reset) -> VoltageScaler(s)
    so that average postsynaptic potential ≈ ANN activation.
    """
    if sj_neuron is None or VoltageScaler is None:
        raise ImportError("spikingjelly is required to build IF conversion blocks")
    return nn.Sequential(
        VoltageScaler(1.0 / scale),
        sj_neuron.IFNode(v_threshold=1.0, v_reset=None, detach_reset=True),  # soft reset
        VoltageScaler(scale),
    )


def _is_relu_module(m: nn.Module) -> bool:
    return isinstance(m, (nn.ReLU, nn.ReLU6))


def _collect_relu_names(fx_model: fx.GraphModule) -> list[str]:
    names = []
    for name, mod in fx_model.named_modules():
        if name and _is_relu_module(mod):
            names.append(name)
    return names


class _ActivationObserver(nn.Module):
    """
    Records per-layer activation statistics used for threshold / scale estimation.
    mode:
      - "max"      : running maximum (MaxNorm)
      - "percentile": running high percentile (RobustNorm / 99.9%)
      - float in (0,1]: scale * max
    """

    def __init__(self, mode: Union[str, float] = "max", momentum: float = 0.1):
        super().__init__()
        self.mode = mode
        self.momentum = momentum
        self.register_buffer("scale", torch.tensor(1.0))
        self._n = 0

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Work on positive activations (ReLU outputs)
        vals = x.detach().float().abs()
        if self.mode == "max":
            cur = vals.max()
        elif isinstance(self.mode, str) and self.mode.endswith("%"):
            p = float(self.mode.rstrip("%")) / 100.0
            cur = torch.quantile(vals.flatten(), p)
        elif isinstance(self.mode, (int, float)):
            cur = vals.max() * float(self.mode)
        else:
            raise ValueError(f"Unknown observer mode: {self.mode}")

        if self._n == 0:
            self.scale.copy_(cur.clamp(min=1e-8))
        else:
            # EMA for stability across calibration batches
            self.scale.mul_(1 - self.momentum).add_(cur.clamp(min=1e-8), alpha=self.momentum)
        self._n += 1
        return x


class RateCodingBaseRecipe(FXConversionRecipe):
    """
    Base FX recipe that:
      1. (optional) fuses Conv-BN
      2. inserts activation observers on ReLU sites
      3. calibrates scales from a dataloader
      4. replaces each ReLU with VoltageScaler -> IFNode -> VoltageScaler

    Subclasses only need to set `mode` (or override calibrate/replace).
    """

    def __init__(
        self,
        dataloader: Iterable,
        mode: Union[str, float] = "max",
        fuse_flag: bool = True,
        momentum: float = 0.1,
        neuron_factory: Optional[Callable[[float], nn.Module]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dataloader = dataloader
        self.mode = mode
        self.fuse_flag = fuse_flag
        self.momentum = momentum
        self.neuron_factory = neuron_factory or _default_if_factory
        self.device = device
        self._scales: Dict[str, float] = {}

    # ---- FX step hooks (order fixed by FXConverter) ----

    def validate(self, converter, model: nn.Module) -> None:
        # Soft check: warn if no ReLU modules (conversion will be a no-op)
        has_relu = any(_is_relu_module(m) for m in model.modules())
        if not has_relu:
            raise ValueError(
                "No nn.ReLU / nn.ReLU6 modules found. "
                "Rate-coding conversion expects module ReLUs (not F.relu)."
            )

    def before_trace(self, converter, model: nn.Module) -> nn.Module:
        if not self.fuse_flag:
            return model
        # Fuse Conv-BN in eval mode (standard for conversion)
        model = model.eval()
        try:
            from torch.nn.utils.fusion import fuse_conv_bn_eval
            # Simple sequential fuse: walk named modules
            # (Full SJ implementation is more thorough; this is a practical subset.)
            modules = dict(model.named_modules())
            for name, mod in list(modules.items()):
                if isinstance(mod, nn.Sequential):
                    i = 0
                    while i < len(mod) - 1:
                        if isinstance(mod[i], (nn.Conv1d, nn.Conv2d, nn.Conv3d)) and isinstance(
                            mod[i + 1], (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
                        ):
                            fused = fuse_conv_bn_eval(mod[i], mod[i + 1])
                            mod[i] = fused
                            del mod[i + 1]
                        else:
                            i += 1
        except Exception:
            pass  # fusion is best-effort
        return model

    def insert_observers(self, converter, fx_model: fx.GraphModule) -> fx.GraphModule:
        # Replace each ReLU with Observer so calibrate() can run the graph
        for name in _collect_relu_names(fx_model):
            parent_name, _, child_name = name.rpartition(".")
            parent = fx_model.get_submodule(parent_name) if parent_name else fx_model
            setattr(parent, child_name, _ActivationObserver(mode=self.mode, momentum=self.momentum))
        return fx_model

    def calibrate(self, converter, fx_model: fx.GraphModule) -> None:
        device = self.device or next(fx_model.parameters()).device
        fx_model.eval().to(device)
        with torch.no_grad():
            for batch in self.dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(device)
                fx_model(x)
        # Harvest scales
        self._scales.clear()
        for name, mod in fx_model.named_modules():
            if isinstance(mod, _ActivationObserver):
                self._scales[name] = float(mod.scale.item())

    def replace(self, converter, fx_model: fx.GraphModule) -> fx.GraphModule:
        for name, scale in self._scales.items():
            parent_name, _, child_name = name.rpartition(".")
            parent = fx_model.get_submodule(parent_name) if parent_name else fx_model
            block = self.neuron_factory(scale)
            setattr(parent, child_name, block)
        return fx_model

    def finalize(self, converter, fx_model: fx.GraphModule) -> fx.GraphModule:
        return fx_model


class MaxNormRecipe(RateCodingBaseRecipe):
    """MaxNorm (mode='max') — classic threshold = max activation [1]."""

    def __init__(self, dataloader, fuse_flag: bool = True, **kwargs):
        super().__init__(dataloader=dataloader, mode="max", fuse_flag=fuse_flag, **kwargs)


class RobustNormRecipe(RateCodingBaseRecipe):
    """RobustNorm using 99.9% percentile [1]."""

    def __init__(self, dataloader, percentile: float = 99.9, fuse_flag: bool = True, **kwargs):
        mode = f"{percentile}%"
        super().__init__(dataloader=dataloader, mode=mode, fuse_flag=fuse_flag, **kwargs)


class PercentileNormRecipe(RateCodingBaseRecipe):
    """General percentile / scaled-max norm (mode = p in (0,1] or 'xx.x%')."""

    def __init__(self, dataloader, mode: Union[str, float] = 0.999, fuse_flag: bool = True, **kwargs):
        super().__init__(dataloader=dataloader, mode=mode, fuse_flag=fuse_flag, **kwargs)


# =============================================================================
# 2. Local threshold balancing style (per-channel / local statistics)
#    Inspired by LocalThresholdBalancingRecipe already in SpikingJelly [docs].
# =============================================================================

class _LocalObserver(nn.Module):
    """Per-channel max observer for local threshold balancing."""

    def __init__(self, momentum: float = 0.1):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("scale", None)  # filled on first forward
        self._n = 0

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, ...) -> per-channel max over N and spatial dims
        dims = [0] + list(range(2, x.ndim))
        cur = x.detach().float().abs().amax(dim=dims).clamp(min=1e-8)  # (C,)
        if self.scale is None:
            self.scale = cur.clone()
            self.register_buffer("scale", self.scale)
        else:
            self.scale.mul_(1 - self.momentum).add_(cur, alpha=self.momentum)
        self._n += 1
        return x


class LocalThresholdBalancingStyleRecipe(RateCodingBaseRecipe):
    """
    Approximate local (per-channel) threshold balancing.
    Uses channel-wise scales instead of a single scalar per layer.
    For full fidelity prefer SpikingJelly's built-in LocalThresholdBalancingRecipe.
    """

    def insert_observers(self, converter, fx_model: fx.GraphModule) -> fx.GraphModule:
        for name in _collect_relu_names(fx_model):
            parent_name, _, child_name = name.rpartition(".")
            parent = fx_model.get_submodule(parent_name) if parent_name else fx_model
            setattr(parent, child_name, _LocalObserver(momentum=self.momentum))
        return fx_model

    def calibrate(self, converter, fx_model: fx.GraphModule) -> None:
        device = self.device or next(fx_model.parameters()).device
        fx_model.eval().to(device)
        with torch.no_grad():
            for batch in self.dataloader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                fx_model(x.to(device))
        self._scales.clear()
        for name, mod in fx_model.named_modules():
            if isinstance(mod, _LocalObserver) and mod.scale is not None:
                # Store mean channel scale as scalar for the default factory;
                # advanced users can override neuron_factory to use vector scales.
                self._scales[name] = float(mod.scale.mean().item())


# =============================================================================
# 3. QCFS-style conversion-time approximation (Bu et al., ICLR 2022)
#    Full QCFS trains the ANN with a quantized clip-floor-shift activation.
#    Here we expose a conversion recipe that uses a shifted/clipped scale
#    estimate so that the IF firing rate better matches the quantized ANN.
# =============================================================================

class QCFSStyleRecipe(RateCodingBaseRecipe):
    """
    Conversion-time helper for models trained with QCFS-like activations.

    Parameters
    ----------
    T : int
        Target simulation time-steps (quantization levels ~ T).
    shift : float
        Floor-shift φ in [0, 1], typically 0.5 for zero expected conversion error
        under the analysis of Bu et al.
    mode : still used for the raw activation range estimate.
    """

    def __init__(
        self, dataloader: DataLoader,
        T: int = 32, shift: float = 0.5, mode: Union[str, float] = "max", fuse_flag: bool = True, **kwargs,
    ):
        super().__init__(dataloader=dataloader, mode=mode, fuse_flag=fuse_flag, **kwargs)
        self.T = T
        self.shift = shift

    def replace(self, converter, fx_model: fx.GraphModule) -> fx.GraphModule:
        # Adjust scale by the QCFS shift so that
        #   expected spike count / T ≈ clip-floor-shift(ANN activation)
        for name, scale in list(self._scales.items()):
            # Effective threshold related to λ / T; we keep VoltageScaler interface
            # and bake shift into the scale seen by the IF node.
            adj = scale * (1.0 + self.shift / max(self.T, 1))
            self._scales[name] = adj
        return super().replace(converter, fx_model)


# =============================================================================
# 4. Free-Lunch style light calibration (Li et al., 2021)
#    After a standard Max/Robust conversion, run a short bias/scale calibration
#    on a small calibration set to reduce residual conversion error.
# =============================================================================

class FreeLunchCalibrationRecipe(RateCodingBaseRecipe):
    """
    Standard rate-coding conversion + optional light parameter calibration.

    After replace(), optionally freezes weights and tunes a small set of
    per-layer scale factors on the calibration loader (a cheap "free lunch").
    """

    def __init__(
        self,
        dataloader: DataLoader,
        mode: Union[str, float] = "max",
        calib_iters: int = 50,
        calib_lr: float = 1e-3,
        fuse_flag: bool = True,
        **kwargs,
    ):
        super().__init__(dataloader=dataloader, mode=mode, fuse_flag=fuse_flag, **kwargs)
        self.calib_iters = calib_iters
        self.calib_lr = calib_lr

    def finalize(self, converter, fx_model: fx.GraphModule) -> fx.GraphModule:
        if self.calib_iters <= 0:
            return fx_model
        # Light calibration: only VoltageScaler gains are trainable
        device = self.device or next(fx_model.parameters()).device
        fx_model.to(device)
        params = []
        for m in fx_model.modules():
            if VoltageScaler is not None and isinstance(m, VoltageScaler):
                for p in m.parameters():
                    p.requires_grad_(True)
                    params.append(p)
            else:
                for p in m.parameters():
                    p.requires_grad_(False)
        if not params:
            return fx_model
        opt = torch.optim.AdamW(params, lr=self.calib_lr)
        # Labels are used for supervised calib; if dataloader yields (x,y) use CE,
        # otherwise skip (unsupervised scale matching is left to the user).
        it = iter(self.dataloader)
        for step in range(self.calib_iters):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(self.dataloader)
                batch = next(it)
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                break
            x, y = batch[0].to(device), batch[1].to(device)
            # Multi-step rate average (T small for speed)
            T = 8
            out = 0.0
            for t in range(T):
                out = out + fx_model(x)
            out = out / T
            loss = nn.functional.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        for p in fx_model.parameters():
            p.requires_grad_(False)
        return fx_model



# ALiases

# Built-in SJ equivalent
RateCodingRecipe = MaxNormRecipe  # default max; pass mode= for RobustNorm etc.

# Paper-oriented names
MaxNormConversionRecipe = MaxNormRecipe
RobustNormConversionRecipe = RobustNormRecipe
QCFSConversionRecipe = QCFSStyleRecipe
SNNCalibrationRecipe = FreeLunchCalibrationRecipe


# =============================================================================
# Example usage (documentation only)
# =============================================================================

EXAMPLE = """
from spikingjelly.activation_based import ann2snn, functional
from ann2snn_recipes import (
    MaxNormRecipe, RobustNormRecipe, QCFSStyleRecipe, FreeLunchCalibrationRecipe
)

# 1) Classic MaxNorm (same spirit as SJ RateCodingRecipe(mode="max"))
recipe = MaxNormRecipe(dataloader=calibration_loader)
snn = ann2snn.FXConverter(recipe=recipe).convert(ann)

# 2) RobustNorm 99.9%
recipe = RobustNormRecipe(dataloader=calibration_loader, percentile=99.9)
snn = ann2snn.FXConverter(recipe=recipe).convert(ann)

# 3) QCFS-style scale adjustment for short-latency targets
recipe = QCFSStyleRecipe(dataloader=calibration_loader, T=16, shift=0.5)
snn = ann2snn.FXConverter(recipe=recipe).convert(ann)

# 4) Free-lunch light calibration after conversion
recipe = FreeLunchCalibrationRecipe(
    dataloader=calibration_loader, mode="max", calib_iters=30
)
snn = ann2snn.FXConverter(recipe=recipe).convert(ann)

# Inference (rate coding)
functional.reset_net(snn)
T = 32
out = 0
for t in range(T):
    out = out + snn(x)
pred = out.argmax(1)
"""