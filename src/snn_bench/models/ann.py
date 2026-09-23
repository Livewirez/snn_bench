import torch
import torch.nn as nn
import torchvision

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
    
    
class ResNet50Modified(nn.Module):
    """
    https://pytorch.org/hub/nvidia_deeplearningexamples_resnet50/
    https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
    https://medium.com/@deepvisionkararhaider/resnet-50-explained-step-by-step-the-easiest-guide-to-deep-residual-networks-7616f4f45046
    https://arxiv.org/pdf/1512.03385
    """
    @property
    def has_backbone(): return True

    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.backbone  = torchvision.models.resnet50(weights="IMAGENET1K_V2", progress=True)
        in_features  = self.backbone.fc.in_features   # 2048 the standard feature embedding vector size

        # replace the original fc layer (a Linear(2048, 1000) trained on ImageNet's 1000 classes) with a passthrough.
        # The backbone now outputs the raw 2048-dimensional feature vector instead of 1000 class logits.
        self.backbone.fc = nn.Identity()

        # Fully conncted layer head outside self.backbone -> stays trainable during phase 1 freeze
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))
    
    
class ResNet18Modified(nn.Module):
    """
    https://pytorch.org/hub/nvidia_deeplearningexamples_resnet50/
    https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
    https://medium.com/@deepvisionkararhaider/resnet-50-explained-step-by-step-the-easiest-guide-to-deep-residual-networks-7616f4f45046
    https://arxiv.org/pdf/1512.03385
    """

    @property
    def has_backbone(): return True

    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.backbone  = torchvision.models.resnet18(weights="IMAGENET1K_V1", progress=True)
        in_features  = self.backbone.fc.in_features   # 2048 the standard feature embedding vector size

        # replace the original fc layer (a Linear(2048, 1000) trained on ImageNet's 1000 classes) with a passthrough.
        # The backbone now outputs the raw 2048-dimensional feature vector instead of 1000 class logits.
        self.backbone.fc = nn.Identity()

        # Fully conncted layer head outside self.backbone -> stays trainable during phase 1 freeze
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))