import os, re, random, json
from typing import Callable, Optional, Union, Iterable, Tuple, Dict, Protocol, runtime_checkable
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


from torch.optim import Optimizer, Adam, AdamW, SGD
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import timm
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import  ImageFolder
from collections import Counter
from torchvision.transforms import ToTensor
from torchmetrics import MeanMetric, Accuracy
from torchmetrics import ConfusionMatrix, Accuracy, Precision, Recall, F1Score

from pathlib import Path

from tqdm import tqdm


from ..config import Config, EvaluationResult

from .types import CriterionType, CriterionProtocol

from .eval_ann import (
    plot_model_curves,
    evaluate_model,
    show_model_results,
    plot_confusion_matrix,
)

OptimizerFactory = Callable[
    [Iterable[torch.nn.Parameter]],
    Optimizer
]


class TrainOptions:
    def __init__(self, criterion: Optional[CriterionType] = None, optimizer: Optional[OptimizerFactory] = None, max_epoch: int = 30):
        self.optimizer = optimizer
        self.criterion = criterion
        self.max_epoch = max_epoch

    @classmethod
    def get_default_none(cls) -> "TrainOptions":
        return cls(
            criterion=None,
            optimizer=None,
        )
        
    @classmethod
    def get_default(cls) -> "TrainOptions":
        return cls(
            criterion=nn.CrossEntropyLoss(label_smoothing=0.1),
            optimizer=lambda params: torch.optim.AdamW(
                params,
                lr=1e-3,
                weight_decay=1e-4,
            ),
        )

DEFAULT_OPTIONS = TrainOptions.get_default()



def get_backbone(model: nn.Module):
    """Return the backbone module for freezing (works for all model types)."""
    for attr in ["backbone", "base", "base_model", "features"]:
        if hasattr(model, attr):
            return getattr(model, attr)
    return None

# a function for training one epoch
def train_one_epoch(model: nn.Module, dataloader: DataLoader, config: Config, optimizer: Optimizer, criterion: CriterionType, epoch: int, max_epoch: int) -> Tuple[float, float]:
    # Prepare for storing loss and accuracy
    losses = MeanMetric().to(config.device)
    acc = Accuracy(task='multiclass', num_classes=config.num_classes).to(config.device)
    model.train() # set model to train mode
    # a loop to iterate input(X)[image] and label(Y) for all mini-batches
    for X, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{max_epoch} Training Loop", leave=False):
        X, y = X.to(config.device), y.to(config.device)
        optimizer.zero_grad() # reset optimizer
        preds = model(X) # model forward
        loss = criterion(preds, y) # calculate loss
        loss.backward() # compute gradients via backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step() # perform gradient descent
        preds = preds.argmax(dim=1) # obtain the final predicted class
        losses.update(loss, X.size(0)) # store loss per batch
        acc.update(preds, y) # store accuracy per batch
    return losses.compute().item(), acc.compute().item()

# a function for validating one epoch
def validate_one_epoch(model: nn.Module, dataloader: DataLoader, config: Config, optimizer: Optimizer, criterion: CriterionType, epoch: int, max_epoch: int) -> Tuple[float, float]:
    # Prepare for storing loss and accuracy
    losses = MeanMetric().to(config.device)
    acc = Accuracy(task='multiclass', num_classes=config.num_classes).to(config.device)
    model.eval() # set model to validation mode
    with torch.no_grad(): # disables gradient computation for evaluation
        # a loop to iterate input(X)[image] and label(Y) for all mini-batches
        for X, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{max_epoch} Validation Loop", leave=False):
            X, y = X.to(config.device), y.to(config.device)
            preds = model(X) # model forward
            loss = criterion(preds, y) # calculate loss
            preds = preds.argmax(dim=1) # obtain the final predicted class
            losses.update(loss, X.size(0)) # store loss per batch
            acc.update(preds, y) # store accuracy per batch
    return losses.compute().item(), acc.compute().item()



def train_model(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
    config: Config, options: TrainOptions = DEFAULT_OPTIONS, num_epochs: int = 30, 
    model_name: str = "Default", use_two_phase: bool =True
) -> pd.DataFrame:
    """
    Two-phase fine-tuning strategy:
      Phase 1 (epochs 1-5):  Freeze base_model -> train classifier head only at learning_rate(lr=1e-3).
                              Lets the head adapt to your 8 classes before touching base_model.
      Phase 2 (epochs 6-30): Unfreeze all -> fine-tune end-to-end at learning_rate(lr=1e-5).
                              Small LR prevents destroying pretrained features.

    Also includes:
      - ReduceLROnPlateau: halves LR if val loss doesn't improve for 3 epochs
      - Early stopping: stops if no improvement for 5 epochs
      - Best checkpoint: saves state dict when val loss improves
    """
    model = model.to(config.device)
    
    criterion = (
        options.criterion
        if options.criterion is not None
        else nn.CrossEntropyLoss(label_smoothing=0.1)
    )
    
    if not isinstance(criterion, CriterionProtocol):
        raise TypeError(f"criterion must be callable, got {type(criterion)}")

    FREEZE_EPOCH_LIMIT = config.freeze_epoch_limit

    history = pd.DataFrame()
    best_val_loss   = float("inf")
    EARLY_STOP_COUNTER  = config.early_stop_counter
    EARLY_STOP_PATIENCE  = config.early_stop_patience
    
    CHECKPOINT_DIR = Config.create_path(config.checkpoint_dir) 
    CHECKPOINT_PATH = CHECKPOINT_DIR / f"best_{model_name.replace(' ', '_')}.pth"

    # Phase 1: freeze backbone(base_model), train classifier head
    backbone = get_backbone(model)
    if use_two_phase and backbone is not None:
        print(f"\n  Phase 1: freezing backbone(base_model), training classifier  head only (5 epochs, lr=1e-3)")
        for p in backbone.parameters():
            p.requires_grad = False
        
        optimizer = (
            options.optimizer(filter(lambda p: p.requires_grad, model.parameters()))
            if options.optimizer is not None
            else torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

        for epoch in range(FREEZE_EPOCH_LIMIT):
            # Train one epoch
            train_loss, train_acc = train_one_epoch(model, train_loader, config, optimizer, criterion, epoch, 5)
            # Validate one epoch
            val_loss, val_acc = validate_one_epoch(model, val_loader, config, optimizer, criterion, epoch, 5)

            # store and print loss and accuracy per epoch
            statistics = pd.DataFrame({
                "epoch": [epoch+1],
                "model": [model.__class__.__name__],
                "model_name": [model_name],
                "train_loss": [train_loss],
                "train_acc": [train_acc],
                "val_loss": [val_loss],
                "val_acc": [val_acc],
                "optimizer": [optimizer.__class__.__name__]
            })
            history = pd.concat([history, statistics], ignore_index=True)
            print(f"  Epoch {epoch+1}/{FREEZE_EPOCH_LIMIT} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | val={val_loss:.4f} | val_acc={val_acc:.3f}")
            print(statistics.to_dict(orient="records")[0])

            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), CHECKPOINT_PATH)

        # Unfreeze base_model for phase 2
        for p in backbone.parameters():
            p.requires_grad = True
        print(f"\n  Phase 2: unfreezing all, fine-tuning end-to-end (lr=1e-5)")
    else:
        print(f"\n  Training from scratch: no backbone(base_model) to freeze")

    # Phase 2 (or full training for no backbone models)
    optimizer = (
        options.optimizer(model.parameters())
        if options.optimizer is not None
        else torch.optim.AdamW(model.parameters(), lr=1e-4 if backbone is None else 1e-5)
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    for epoch in range(FREEZE_EPOCH_LIMIT if use_two_phase and backbone else 0, num_epochs):
        # Train one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, config, optimizer, criterion,
            epoch,
            num_epochs
        )
        # Validate one epoch
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, config, optimizer, criterion,
            epoch,
            num_epochs
        )
        # store and print loss and accuracy per epoch
        statistics = pd.DataFrame({
            "epoch": [epoch+1],
            "model": [model.__class__.__name__],
            "model_name": [model_name],
            "train_loss": [train_loss],
            "train_acc": [train_acc],
            "val_loss": [val_loss],
            "val_acc": [val_acc],
            "optimizer": [optimizer.__class__.__name__]
        })
        history = pd.concat([history, statistics], ignore_index=True)
        print(f"  Epoch {epoch+1}/{num_epochs} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | val={val_loss:.4f} | val_acc={val_acc:.3f}")
        print(statistics.to_dict(orient="records")[0])

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  LR: {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            EARLY_STOP_COUNTER = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  ✔ Val loss improved -> checkpoint saved")
        else:
            EARLY_STOP_COUNTER += 1
            print(f"  No improvement ({EARLY_STOP_COUNTER}/{EARLY_STOP_PATIENCE})")
            if EARLY_STOP_COUNTER >= EARLY_STOP_PATIENCE:
                print(f"  (<->) Early stopping triggered.")
                break

    # Load best weights
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=config.device))
    print(f"  ✔✔ Best checkpoint loaded: val_loss={best_val_loss:.4f}")
    return history


def start_training_ann_loops(
    models: Dict[str, Tuple[nn.Module, bool]], 
    tr_loader: DataLoader, v_loader: DataLoader, te_loader: DataLoader, 
    config: Config, options: TrainOptions = DEFAULT_OPTIONS
) -> Tuple[pd.DataFrame, Dict[str, EvaluationResult], Dict[str, EvaluationResult]]:
    results = {}  # collect evaluation result for comparisons
    history_results = pd.DataFrame() # for full results from model loop
    evaluation_results = {}

    for model_name, (model, use_two_phase) in models.items():
        print(f"\n{'='*60}")
        print(f"  Training: {model_name}")
        #n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #print(f"  Trainable params: {n_params:,}")
        print(f"{'='*60}")

        statistics = train_model(
            model, tr_loader, v_loader, config, options,
            num_epochs=config.max_epoch_limit, model_name=model_name, use_two_phase=use_two_phase
        )

        history_results = pd.concat([history_results, statistics], ignore_index=True)

        plot_model_curves(statistics, model_name)

        evaluation_result = evaluate_model(model, te_loader, config)
        evaluation_results[model_name] = evaluation_result

        results[model_name] = show_model_results(evaluation_result, model_name, config)

        # Only plot confusion matrix for best models (saves time)
        # if "GoogLeNet" in model_name or "SE" in model_name:
        if any(name in model_name for name in ["Resnet50", "GoogLeNet", "VGG16"]):
            plot_confusion_matrix(evaluation_result, model_name, config)

    return history_results, evaluation_results, results


def start_training_ann_loop(
    model: nn.Module, tr_loader: DataLoader, v_loader: DataLoader, te_loader: DataLoader, 
    config: Config, options: TrainOptions = DEFAULT_OPTIONS, use_two_phase: bool = False
) -> Tuple[pd.DataFrame, Dict[str, EvaluationResult], Dict[str, EvaluationResult]]:
    results = {}  # collect evaluation result for comparisons
    history_results = pd.DataFrame() # for full results from model loop
    evaluation_results = {}
    
    model_name = model.__class__.__name__

    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    #n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f"  Trainable params: {n_params:,}")
    print(f"{'='*60}")

    statistics = train_model(
        model, tr_loader, v_loader, config, options,
        num_epochs=config.max_epoch_limit, model_name=model_name, use_two_phase=use_two_phase
    )

    history_results = pd.concat([history_results, statistics], ignore_index=True)

    plot_model_curves(statistics, model_name)

    evaluation_result = evaluate_model(model, te_loader, config)
    evaluation_results[model_name] = evaluation_result

    results[model_name] = show_model_results(evaluation_result, model_name, config)

    # Only plot confusion matrix for best models (saves time)
    # if "GoogLeNet" in model_name or "SE" in model_name:
    if any(name in model_name for name in ["Resnet50", "GoogLeNet", "VGG16"]):
        plot_confusion_matrix(evaluation_result, model_name, config)
        
    return history_results, evaluation_results, results