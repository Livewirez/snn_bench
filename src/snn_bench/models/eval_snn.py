from __future__ import annotations

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from spikingjelly.activation_based import functional
from spikingjelly.activation_based import neuron, encoding, functional, surrogate
from spikingjelly import visualizing
from matplotlib import pyplot as plt
import numpy as np
from torchvision.transforms import v2

import math
from dataclasses import dataclass, field
from typing import Sequence, Mapping, Any, Optional, Callable

from .types import SNNModule, SpikingJellyEncoder
from ..config import Config

@dataclass
class Curve:
    y: Sequence[float]
    label: str
    marker: str = "o"
    ms: float = 3
    color: str | None = None
    ls: str = "-"




def plot_snn_module_spikes(
    snn: SNNModule, test_dataset: Dataset, encoder: SpikingJellyEncoder, 
    config: Config, index: int, T: int = 20, test_transforms: Optional[Sequence[Callable]] = None
):
    #For a more intuitive understanding, we can select a single image for prediction; by changing the index, we can choose different input images.

    # Obtaining the membrane potential of the output layer neurons at this point is somewhat challenging and requires the use of the “hook” feature.

    # Retrieve a single image for prediction

    # Load the raw data
    # Load a single image from the test set
    # Save the data for plotting
    snn.eval()
    functional.reset_net(snn)
    # Register a hook
    output_layer = snn.get_output_layer() # Output layer
    output_layer.v_seq = []
    output_layer.s_seq = []
    def save_hook(m, x, y):
        m.v_seq.append(m.v.unsqueeze(0))
        m.s_seq.append(y.unsqueeze(0))

    hook = output_layer.register_forward_hook(save_hook)

    # Load a single image from the test set
    input, label = test_dataset[index]
    
    transforms = (
        test_transforms
        if test_transforms is not None
        else v2.Compose([torchvision.transforms.ToPILImage()])
    )

    #input = torchvision.transforms.ToPILImage()(input)
    input = transforms(input)
    plt.imshow(input)
    print('Original image')
    print(f"label: {label}")
    plt.show()
    
    T = 20
    # Start making predictions with the neural network
    snn.eval()

    with torch.no_grad():
        img, label = test_dataset[index]
        
        try:
            img = img.to(config.device)
            out_fr = 0.
            for t in range(T):
                encoded_img = encoder(img)
                out_fr += snn(encoded_img)
            out_spikes_counter_frequency = (out_fr / T).cpu().numpy()
            
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            img = img.unsqueeze(0).to(config.device)
            out_fr = 0.
            for t in range(T):
                encoded_img = encoder(img)
                out_fr += snn(encoded_img)
            out_spikes_counter_frequency = (out_fr / T).cpu().numpy()

        output_layer.v_seq = torch.cat(output_layer.v_seq)
        output_layer.s_seq = torch.cat(output_layer.s_seq)
        v_t_array = output_layer.v_seq.cpu().numpy().squeeze()  # v_t_array[i][j] represents the voltage value of neuron i at time j
        s_t_array = output_layer.s_seq.cpu().numpy().squeeze()  # s_t_array[i][j] represents the spike fired by neuron i at time j, which is either 0 or 1

        # Heatmap of membrane potentials and spike output results
        figsize = (12, 8)
        dpi = 100
        visualizing.plot_2d_heatmap(array=v_t_array, title='membrane potentials', xlabel='simulating step',
                                    ylabel='neuron index', int_x_ticks=True, x_max=T, figsize=figsize, dpi=dpi)
        visualizing.plot_1d_spikes(spikes=s_t_array, title='membrane sotentials', xlabel='simulating step',
                                ylabel='neuron index', figsize=figsize, dpi=dpi)

        plt.show()

    hook.remove()

@torch.no_grad()
def accuracy_vs_T(snn: nn.Module, loader: DataLoader, T_max: int, device: torch.device):
    """
    Accuracy vs Timestep
    
    Args:
        snn (nn.Module): Spiking Neural Network or GraphModule (nn.Module) for converted ANN to SNN
        loader (torch.DataLoader): Data Loader class
        T_max (int): Maximum Timestep
        device (torch.device): Device model will run on

    Returns:
        List[float]: list of numbers of accuracy for each timestep
        
    Example:
    ```python
        curve_net = accuracy_vs_T(snn, a_test_dataloader, T_max=64, device=device)
    ```
    """
    
    snn.eval()
    correct = torch.zeros(T_max); total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        functional.reset_net(snn)
        acc, preds = 0., []
        for t in range(T_max):
            acc = acc + snn(x)
            preds.append(acc.argmax(1))          # prediction using spikes up to t
        for t in range(T_max):
            correct[t] += (preds[t] == y).sum().item()
        total += y.size(0)
    return (100.0 * correct / total).tolist()



def plot_accuracy_vs_T(
    ann_acc: float, T_max: int, curves: Sequence[Mapping[str, Any]],  
    title: str = "Accuracy vs latency: ANN-to-SNN conversion on MNIST",
    savepath: str = "accuracy_vs_latency.png",
):
    """Plots the results of the accuracy_vs_T function

    Args:
        ann_acc (float): Accuracy of the ANN model
        T_max (int): same as T_max in accuracy_vs_T
        curves (Sequence[Mapping[str, Any]]): each dict must contain
            "y" (array-like of accuracies) and "label" (str); any other
            key is forwarded to ax.plot as a kwarg (marker, ms, color, ls, ...)
        title (str): plot title
        savepath (str): output image path
        
    Example:
    ```python
        plot_accuracy_vs_T(
            ann_acc=ann_acc,
            T_max=64,
            curves=[
                {"y": curve_max, "marker": "o", "ms": 3, "label": "Conversion (max norm.)"},
                {"y": curve_99,  "marker": "s", "ms": 3, "label": "Conversion (99.9% norm.)"},
                {"y": curve_net, "marker": "*", "ms": 3, "label": "SNN"},
            ],
        )
    ```
    """
    T = range(1, T_max + 1) # x-axis: timesteps 1..T_max
    
    ann_acc_percent = ann_acc * 100
   
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    for curve in curves:
        curve = dict(curve) # immutable dict
        y = curve.pop("y")
        ax.plot(T, y, **curve)       # label, marker, ms, color, ls, etc all flow through

    # ax.plot(T, curve_max,  marker="o", ms=3, label='Conversion (max norm.)')
    # ax.plot(T, curve_99, marker="s", ms=3, label='Conversion (99.9% norm.)')
    # ax.plot(T, curve_net, marker="*", ms=3, label='SNN')
    
    ax.axhline(ann_acc_percent, ls="--", color="grey", label=f'ANN baseline ({ann_acc_percent:.1f}%)')

    ax.set_xlabel("Simulation timesteps  T  (latency)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(title)
    ax.set_xscale("log", base=2)          # timesteps span 1..64 — log base-2 spreads them out
    ax.set_xticks([2**i for i in range(int(math.log2(T_max)) + 1)])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())   # show 1,2,4.. not 2^0,2^1
    ax.grid(alpha=.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.show()
    
    
def plot_accuracy_vs_T_from_curve(
    ann_acc: float, T_max: int, curves: Sequence[Curve],  
    title: str = "Accuracy vs latency: ANN-to-SNN conversion on MNIST",
    savepath: str = "accuracy_vs_latency.png",
):
    """
    Plots the results of the accuracy_vs_T function

    Args:
        ann_acc (float): Accuracy of the ANN model
        T_max (int): same as T_max in accuracy_vs_T
        curves (Sequence[Curve]): each dict must contain
            "y" (array-like of accuracies) and "label" (str); any other
            key is forwarded to ax.plot as a kwarg (marker, ms, color, ls, ...)
        title (str): plot title
        savepath (str): output image path
        
    Example:
    ```python
        plot_accuracy_vs_T(
            ann_acc=ann_acc,
            T_max=64,
            curves=[
                Curve(y=curve_max, marker="o", ms=3, label="Conversion (max norm.)"),
                Curve(y=curve_99, marker="s", ms=3, label="Conversion (99.9% norm.)"),
                Curve(y=curve_net, marker="*", ms=3, label="SNN"),
            ],
        )
    ```
    """
    T = range(1, T_max + 1) # x-axis: timesteps 1..T_max
    
    ann_acc_percent = ann_acc * 100
   
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    for c in curves:
        ax.plot(T, c.y, marker=c.marker, ms=c.ms, label=c.label, color=c.color, ls=c.ls)

    # ax.plot(T, curve_max,  marker="o", ms=3, label='Conversion (max norm.)')
    # ax.plot(T, curve_99, marker="s", ms=3, label='Conversion (99.9% norm.)')
    # ax.plot(T, curve_net, marker="*", ms=3, label='SNN')
    
    ax.axhline(ann_acc_percent, ls="--", color="grey", label=f'ANN baseline ({ann_acc_percent:.1f}%)')

    ax.set_xlabel("Simulation timesteps  T  (latency)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(title)
    ax.set_xscale("log", base=2)          # timesteps span 1..64 — log base-2 spreads them out
    ax.set_xticks([2**i for i in range(int(math.log2(T_max)) + 1)])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())   # show 1,2,4.. not 2^0,2^1
    ax.grid(alpha=.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.show()
    
    
def trace_neuron(snn: nn.Module, node, img, T: int, device: torch.device):
    """
    Record membrane potential v and spikes of one IF node over T steps.
    
    Example:
    
    ```python
    # grab the final spiking layer of each converted model
    node_max = snn.get_submodule("classifier.spiking_1.if_node") # converted snn1(mode=max)
    node_99  = snn_99.get_submodule("classifier.spiking_1.if_node")  # converted snn2 (mode=99.9%)

    img, label = a_test_dataset[3373] # random index
    T = 64

    for tag, s_net, node in [("max", snn, node_max), ("99.9%", snn_99, node_99)]:
        v_arr, s_arr = trace_neuron(s_net, node, img, T, device)
        visualizing.plot_2d_heatmap(array=v_arr, title=f"Membrane potential — converted SNN ({tag})",
                                    xlabel="timestep", ylabel="output neuron",
                                    int_x_ticks=True, x_max=T, figsize=(12, 5), dpi=100)
        visualizing.plot_1d_spikes(spikes=s_arr, title=f"Output spikes — converted SNN ({tag})",
                                xlabel="timestep", ylabel="output neuron", figsize=(12, 5), dpi=100)
        plt.show()
    ```
    
    """
    snn = snn.to(device).eval()
    v_seq, s_seq = [], []

    def hook(m, x, y):
        v_seq.append(m.v.detach().float().cpu().unsqueeze(0))   # membrane BEFORE reset
        s_seq.append(y.detach().float().cpu().unsqueeze(0))     # output spikes (0/1)

    h = node.register_forward_hook(hook)
    functional.reset_net(snn)
    with torch.no_grad():
        x = img.unsqueeze(0).to(device) # [1,1,28,28] — converted model takes analog input
        for _ in range(T):
            snn(x)  # feed analog image each step; IF accumulates
    functional.reset_net(snn)
    h.remove()

    v = torch.cat(v_seq).numpy().squeeze() # [T, n_neurons]
    s = torch.cat(s_seq).numpy().squeeze()
    return v.T, s.T   # transpose -> [neuron, step] for the plotters

def plot_loss_curves(epoch, loss_train_store, loss_cv_store, accuracy_rate_train_store, accuracy_rate_cv_store):
    # Plot the loss curve
    print('---------Graph showing how the loss function changes over training epochs---------')
    plt.plot(np.arange(epoch), loss_train_store, np.arange(epoch), loss_cv_store)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["loss train", 'loss cv'])
    plt.title('loss curve')
    plt.show()

    # Plot the accuracy curve
    print('---------Graph showing how accuracy changes over training epochs---------')
    plt.plot(np.arange(epoch), accuracy_rate_train_store, np.arange(epoch), accuracy_rate_cv_store)
    plt.xlabel('epoch')
    plt.ylabel('accuracy rate')
    plt.legend(["accuracy rate train", "accuracy rate cv"])
    plt.title('accuracy rate curve')
    plt.show()