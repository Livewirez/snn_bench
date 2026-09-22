from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torchmetrics import ConfusionMatrix

from pathlib import Path

from tqdm import tqdm
from typing import Tuple, Union, Callable, List

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 128


@dataclass(frozen=True)
class EvaluationResult:
    confusion_matrix: ConfusionMatrix
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    all_preds: list = field(default_factory=list)
    all_labels: list = field(default_factory=list)

@dataclass
class Config:
    num_classes: int
    class_names: List[str]
    checkpoint_dir: str
    batch_size: int = BATCH_SIZE
    device: torch.device = DEVICE
    early_stop_counter: int  = 0
    early_stop_patience: int  = 5
    freeze_epoch_limit: int  = 5
    max_epoch_limit: int = 30
    model_auto_save_dir: str = './model_save/auto_save/'
    model_history_save_dir: str = './model_save/history_save/'