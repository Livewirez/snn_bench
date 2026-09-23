import torch
from torch.optim import Optimizer



from .train_ann import train_ann
from .eval_ann import eval_ann
from .snn import SNN, DirectSNN          # example class
from .train_snn import train_snn
from .eval_snn import eval_snn
from .ann import ANN, ResNet50Modified, ResNet18Modified          # example class
from .types import  CriterionType, OptimizerFactory


__all__ = [
    "train_ann",
    "eval_ann",
    "train_snn",
    "eval_snn",
    "ANN",
    "ResNet50Modified",
    "ResNet18Modified",
    "SNN",
    "DirectSNN",
    "CriterionType",
    "OptimizerFactory"
]
