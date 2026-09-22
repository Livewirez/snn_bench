import torch
import torch.nn as nn

class ANN(nn.Module):
    @property
    def has_backbone(): return False

    def __init__(self, num_classes=8):
        super().__init__()
        # convolutions & pooling layers
        self.features = nn.Sequential(
            # Block 1: 1 conv layer
            nn.LazyConv2d(32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # from 224 to 112

            # Block 2: 1 conv layer
            nn.LazyConv2d(64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # from 112 to 56
        )

        # fully-connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),  # flat tensor to vector
            nn.LazyLinear(512),
            nn.ReLU(),
            
            nn.Dropout(p=0.4),
            nn.LazyLinear(256),
            nn.ReLU(),
            
            nn.LazyLinear(num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output