from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from spikingjelly.activation_based import functional
from spikingjelly.activation_based import neuron, encoding, functional, surrogate
from spikingjelly import visualizing
from matplotlib import pyplot as plt
import numpy as np

import timm
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import  ImageFolder
from torchvision.transforms import v2

import math
from dataclasses import dataclass, field
from typing import Sequence, Mapping, Any, Tuple, Optional, Callable, Union

Device = Union[torch.device, str]

class SpikingReadout(nn.Module):
    """IF neurons on the logits: firing rate ?= clip(logit / scale, 0, 1)."""
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale
        self.if_node = neuron.IFNode(v_threshold=1.0, v_reset=None)
    def forward(self, x):
        return self.if_node(x / self.scale)
    

def calibrate_scale(model, ds, T=32, n=1000, bs=200, device: Device = 'cuda'):
    model.eval()
    mx = float("-inf")
    with torch.no_grad():
        for i in range(0, n, bs):
            xb = torch.stack([ds[j][0] for j in range(i, min(i + bs, n))]).to(device)
            functional.reset_net(model)                 # reset per batch: state shape follows batch size
            out = sum(model(xb) for _ in range(T)) / T  # [bs, num_classes] time-averaged logits
            mx = max(mx, out.max().item())
    functional.reset_net(model)
    return mx

def make_spiking_head(model, ds, T_cal=32, n=1000, device: Device = 'cuda'):
    readout = SpikingReadout(calibrate_scale(model, ds, T_cal, n)).to(device)
    return nn.Sequential(model, readout).eval(), readout

def plot_readout(v, s, label):
    T = len(s); rate = s.mean(0); pred = int(rate.argmax())
    col = ['tab:green' if k == label else ('tab:red' if k == pred else 'gray') for k in range(10)]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4), gridspec_kw={'width_ratios': [3, 3, 1.5]})
    im = ax[0].imshow(v.T, aspect='auto'); fig.colorbar(im, ax=ax[0])
    ax[0].set(title='Membrane potential', xlabel='step', ylabel='neuron')
    ax[1].eventplot([np.where(s[:, k])[0] for k in range(10)], lineoffsets=range(10), colors=col)
    ax[1].set(title='Output spikes', xlabel='step', xlim=(-.5, T - .5)); ax[1].invert_yaxis()
    ax[2].barh(range(10), rate, color=col); ax[2].invert_yaxis()
    ax[2].set(title=f'Firing rate (label={label}, pred={pred})')
    plt.tight_layout(); plt.show()
    
def run(snn_out, readout, img, T, device: Device = 'cuda'):
    """Returns membrane potentials v and output spikes s, both [T,  num_classes]."""
    x = img.reshape(1, 1, 28, 28).to(device)
    functional.reset_net(snn_out)
    v, s = [], []
    with torch.no_grad():
        for _ in range(T):
            s.append(snn_out(x).squeeze(0).cpu())
            v.append(readout.if_node.v.reshape(-1).cpu())
    return torch.stack(v).numpy(), torch.stack(s).numpy()    
    

def try_visualize_converted(model: nn.Module, dataset: Dataset, T: int, index: int, device: Device = 'cuda', test_transforms: Optional[Sequence[Callable]] = None, figsize: Tuple[int, int] = (12, 8)):
    """
       Visualize Spikes of Converted SNN 
       
    Args:
        model (nn.Module): Model
        dataset (_type_): _description_
        T (_type_): _description_
        index (int, optional): _description_. Defaults to 3373.
        device (str, optional): _description_. Defaults to 'cuda'.

    Raises:
        RuntimeError: _description_
        RuntimeError: _description_
    Example:
    ```python
        try_visualize_converted(snn, a_test_dataset, T=16, index=3373, device=device)
        # or for the 99.9 % calibrated model
        try_visualize_converted(snn_99, a_test_dataset, T=16, index=3373, device=device)
    ```
    """
    
    model.eval()
    functional.reset_net(model)

    # Locate the last IFNode that produces the 10-class spikes
    # (for the structure you printed it is classifier.spiking_1.if_node)
    output_neuron = None
    for name, module in model.named_modules():
        if isinstance(module, neuron.IFNode) or isinstance(module, neuron.LIFNode):
            output_neuron = module   # keep the last one we encounter
    if output_neuron is None:
        raise RuntimeError("No IFNode / LIFNode found in the model")

    # Storage for membrane potential and spikes
    output_neuron.my_v_seq = []
    output_neuron.my_s_seq = []

    def save_hook(m, x, y):
        # m.v is the membrane potential *after* the current step
        # y is the spike tensor
        output_neuron.my_v_seq.append(m.v.detach().unsqueeze(0))
        output_neuron.my_s_seq.append(y.detach().unsqueeze(0))

    hook = output_neuron.register_forward_hook(save_hook)

    # Load one sample
    img, label = dataset[index]
    print(f"Original label: {label}")
    transforms = (
        test_transforms
        if test_transforms is not None
        else v2.Compose([torchvision.transforms.ToPILImage()])
    )
    plt.imshow(transforms(img), cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

    with torch.no_grad():
        x = img.unsqueeze(0).to(device)          # [1, 1, 28, 28]
        out_fr = 0.
        for t in range(T):
            out_fr += model(x)                   # rate-coding style accumulation
        out_fr = out_fr / T
        pred = out_fr.argmax(1).item()
        print(f"Predicted class (rate coding, T={T}): {pred}")

        # Concatenate recorded sequences
        if len(output_neuron.my_v_seq) == 0:
            raise RuntimeError("Hook collected no data - check that the correct neuron was selected")
        v_t_array = torch.cat(output_neuron.my_v_seq).cpu().numpy().squeeze()  # [T, 10]
        s_t_array = torch.cat(output_neuron.my_s_seq).cpu().numpy().squeeze()  # [T, 10]

        # Visualise
        dpi = 100
        visualizing.plot_2d_heatmap(
            array=v_t_array,
            title='Membrane potentials (output layer)',
            xlabel='simulating step',
            ylabel='neuron index',
            int_x_ticks=True,
            x_max=T,
            figsize=figsize,
            dpi=dpi
        )
        visualizing.plot_1d_spikes(
            spikes=s_t_array,
            title='Output spikes',
            xlabel='simulating step',
            ylabel='neuron index',
            figsize=figsize,
            dpi=dpi
        )
        plt.show()

    hook.remove()
    functional.reset_net(model)
    
    snn_out, readout = make_spiking_head(model, dataset)
    plot_readout(*run(snn_out, readout, img, T=T, device=device), label)
    
    
def trace_converted(model: nn.Module, img, T: int, device: Device = 'cuda', node_name: str ="classifier.spiking_1.if_node"):
    """_summary_

    Args:
        model (nn.Module): _description_
        img (_type_): _description_
        T (int): _description_
        device (Device, optional): _description_. Defaults to 'cuda'.
        node_name (str, optional): _description_. Defaults to "classifier.spiking_1.if_node".

    Returns:
        _type_: _description_
    Example:
    ```python
        img, label = a_test_dataset[3373]
        T = 64
        print(f"true label: {label}")

        for tag, model in [("max", snn), ("99.9%", snn_99)]:
            v_arr, s_arr = trace_converted(model, img, T, device)
            print(f"{tag}: v {v_arr.shape}, spikes/step total {s_arr.sum():.0f}")
            visualizing.plot_2d_heatmap(array=v_arr, title=f"Membrane potential — converted SNN ({tag})",
                                        xlabel="simulating step", ylabel="neuron index",
                                        int_x_ticks=True, x_max=T, figsize=(12, 5), dpi=100)
            visualizing.plot_1d_spikes(spikes=s_arr, title=f"Output spikes — converted SNN ({tag})",
                                    xlabel="simulating step", ylabel="neuron index",
                                    figsize=(12, 5), dpi=100)
            plt.show()
    ```
    """
    model = model.to(device).eval()
    node = model.get_submodule(node_name)

    functional.reset_net(model)                 # reset FIRST
    my_v_seq, my_s_seq = [], []                        # local lists, captured by closure

    def save_hook(m, x, y):
        v = m.v
        if not torch.is_tensor(v) or v.dim() == 0:
            v = torch.zeros_like(y)              # first-step placeholder
        my_v_seq.append(v.detach().cpu())
        my_s_seq.append(y.detach().cpu())

    h = node.register_forward_hook(save_hook)
    with torch.no_grad():
        x = img.unsqueeze(0).to(device)          # [1,1,28,28] analog image
        for _ in range(T):
            model(x)
    h.remove()
    functional.reset_net(model)

    v = torch.cat(my_v_seq).numpy().squeeze()       # [T, 10]
    s = torch.cat(my_s_seq).numpy().squeeze()
    return v.T, s.T                               # -> [neuron, step]


