import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import numpy as np
from torchsummary import summary # Libraries for previewing neural network architectures
from spikingjelly.activation_based import neuron, layer, functional, surrogate, encoding
from spikingjelly import visualizing
from matplotlib import pyplot as plt
import time

from snn_bench.config import Config


class DirectSNN(nn.Module):
    """Same topology as the ANN: 2xConv+Pool → FC 512 → FC 256 → FC 10.
    ReLU replaced by LIF; dropout omitted (or use layer.Dropout)."""

    def __init__(self, num_classes=10, tau=2.0):
        super().__init__()
        surr = surrogate.ATan()

        self.features = nn.Sequential(
            # Block 1  (matches ANN: 32 filters, 3×3, pool/2)
            layer.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surr),
            layer.MaxPool2d(kernel_size=2),   # 28 → 14

            # Block 2  (matches ANN: 64 filters, 3×3, pool/2)
            layer.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surr),
            layer.MaxPool2d(kernel_size=2),   # 14 → 7
        )

        self.classifier = nn.Sequential(
            layer.Flatten(),
            # 64 * 7 * 7 = 3136  (for 28×28 input)
            layer.Linear(64 * 7 * 7, 512, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surr),
            # optional: layer.Dropout(0.4)
            layer.Linear(512, 256, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surr),
            layer.Linear(256, num_classes, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surr),  # output spikes
        )

    def forward(self, x):
        # x: [N, 1, 28, 28]  or already spikes from encoder
        x = self.features(x)
        x = self.classifier(x)
        return x


# Define network architecture
class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(in_features=28 * 28, out_features=10, bias=False)
        self.lif_layer_1 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.lif_layer_1(x)
        # x = F.softmax(x, dim=0) #softmax activation layer is used for multivariate classification
        return x




def test_snn(config: Config):
    # Instantiate neural network
    snn = SNN()
    snn = snn.to(config.device) # Convert neural network

    # Print neural network information
    print ("Neural network information")
    print(snn)

    # Test the output format of the neural network
    # Set the input that meets the requirements
    input = torch.ones([1, 1, 28, 28])
    input = input.to(config.device)
    print ("Input data format", input.shape)

    output = snn(input)
    print ("Output data format", output.shape)

    # View the size of the spiking neural network model
    summary(snn, input_size=[[1, 28, 28]])

    # After optimizing the parameters once, you need to reset the state of the network, because SNN neurons have "memory”
    functional.reset_net(snn)