from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from spikingjelly.activation_based import functional
from spikingjelly.activation_based import neuron, encoding, functional, surrogate
from spikingjelly import visualizing
from matplotlib import pyplot as plt

import timm
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import  ImageFolder

import math
from dataclasses import dataclass, field
from typing import Sequence, Mapping, Any

Device = Union[torch.device, str]


def try_visualize_converted(model: nn.Module, dataset: Dataset, T: int, index: int, device: Device = 'cuda'):
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
    output_neuron.v_seq = []
    output_neuron.s_seq = []

    def save_hook(m, x, y):
        # m.v is the membrane potential *after* the current step
        # y is the spike tensor
        output_neuron.v_seq.append(m.v.detach().unsqueeze(0))
        output_neuron.s_seq.append(y.detach().unsqueeze(0))

    hook = output_neuron.register_forward_hook(save_hook)

    # Load one sample
    img, label = dataset[index]
    print(f"Original label: {label}")
    plt.imshow(torchvision.transforms.ToPILImage()(img), cmap='gray')
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
        if len(output_neuron.v_seq) == 0:
            raise RuntimeError("Hook collected no data – check that the correct neuron was selected")
        v_t_array = torch.cat(output_neuron.v_seq).cpu().numpy().squeeze()  # [T, 10]
        s_t_array = torch.cat(output_neuron.s_seq).cpu().numpy().squeeze()  # [T, 10]

        # Visualise
        figsize = (12, 8)
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
    v_seq, s_seq = [], []                        # local lists, captured by closure

    def save_hook(m, x, y):
        v = m.v
        if not torch.is_tensor(v) or v.dim() == 0:
            v = torch.zeros_like(y)              # first-step placeholder
        v_seq.append(v.detach().cpu())
        s_seq.append(y.detach().cpu())

    h = node.register_forward_hook(save_hook)
    with torch.no_grad():
        x = img.unsqueeze(0).to(device)          # [1,1,28,28] analog image
        for _ in range(T):
            model(x)
    h.remove()
    functional.reset_net(model)

    v = torch.cat(v_seq).numpy().squeeze()       # [T, 10]
    s = torch.cat(s_seq).numpy().squeeze()
    return v.T, s.T                               # -> [neuron, step]


