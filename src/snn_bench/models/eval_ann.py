import torch
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import pandas as pd

from tqdm import tqdm
from typing import Callable, Optional, Union, Iterable, Tuple, List
from torchmetrics import MeanMetric, Accuracy
from torchmetrics import ConfusionMatrix, Accuracy, Precision, Recall, F1Score
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from snn_bench.config import EvaluationResult, Config


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, config: Config) -> Tuple[List, List]:
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        preds = model(images.to(config.device)).argmax(1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
    return all_preds, all_labels


def evaluate_model(model: nn.Module, loader: DataLoader, config: Config) -> EvaluationResult:
    # prepare for storing evaluation metrics
    test_acc = Accuracy(task="multiclass", num_classes=config.num_classes).to(config.device)
    test_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=config.num_classes).to(config.device)
    test_precision = Precision(task="multiclass", num_classes=config.num_classes, average='macro').to(config.device)
    test_recall = Recall(task="multiclass", num_classes=config.num_classes, average='macro').to(config.device)
    test_f1_score = F1Score(task="multiclass", num_classes=config.num_classes, average='macro').to(config.device)

    # model = model.to(DEVICE)
    all_preds, all_labels = [], []

    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        for X, y in tqdm(loader, desc=f"Evaluation Loop", leave=False):
            X, y = X.to(config.device), y.to(config.device)
            preds = model(X) # model forward
            preds = preds.argmax(dim=1) # obtain the final predicted class
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

            # store loss and accuracy per batch
            test_confusion_matrix.update(preds, y)
            test_acc.update(preds, y)
            test_precision.update(preds, y)
            test_recall.update(preds, y)
            test_f1_score.update(preds, y)

    return EvaluationResult(
        confusion_matrix=test_confusion_matrix,
        accuracy=test_acc.compute().item(),
        precision=test_precision.compute().item(),
        recall=test_recall.compute().item(),
        f1_score=test_f1_score.compute().item(),
        all_preds=all_preds,
        all_labels=all_labels
    )

def show_results(preds, labels, model_name: str, config: Config):
    print(f"\n{'─'*55}\n  {model_name}\n{'─'*55}")
    print(classification_report(labels, preds, target_names=config.class_names, digits=3))
    macro_f1 = f1_score(labels, preds, average="macro")
    print(f"  Macro F1: {macro_f1:.4f}")
    return macro_f1

def show_model_results(result: EvaluationResult, model_name: str, config: Config) -> EvaluationResult:
    print(f"\n{'─'*55}\n  {model_name}\n{'─'*55}")
    print(classification_report(result.all_labels, result.all_preds, target_names=config.class_names, digits=3))
    macro_f1 = f1_score(result.all_labels, result.all_preds, average="macro")
    print(f"  Macro F1: {macro_f1:.4f}\n")

    # Print the Results
    print("  Confusion Matrix:", result.confusion_matrix.compute(), "\n")
    print(f"  Accuracy: {result.accuracy}")
    print(f"  Precision: {result.precision}")
    print(f"  Recall: {result.recall}")
    print(f"  F1 Score: {result.f1_score}")
    return result


def plot_confusion_matrix(result: EvaluationResult, model_name: str, config: Config):
    sns.heatmap(
        result.confusion_matrix.compute().cpu(),
        annot=True,
        fmt="d", cmap="Blues",
        xticklabels=config.class_names,
        yticklabels=config.class_names
    )
    # Labels and title
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    fname = f"cm_{model_name.replace(' ','_').lower()}.png"
    plt.savefig(fname, dpi=150)
    plt.show()

def plot_cm(preds, labels, model_name: str, config: Config) -> str:
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=config.class_names, yticklabels=config.class_names, ax=ax)
    ax.set(title=f"Confusion matrix — {model_name}",
           ylabel="True", xlabel="Predicted")
    plt.xticks(rotation=45, ha="right");
    plt.tight_layout()

    fname = f"cm_{model_name.replace(' ','_').lower()}.png"
    plt.savefig(fname, dpi=150)

    plt.show()

    return fname


def plot_curves(tr: list, vl: list, model_name: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(tr, marker="o", ms=4, label="Train loss") # Train Loss
    ax.plot(vl, marker="s", ms=4, label="Val loss") # Val Loss
    ax.set(title=f"Loss curves — {model_name}", xlabel="Epoch", ylabel="Loss")
    ax.legend()
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()

    fname = f"loss_{model_name.replace(' ','_').lower()}.png"
    plt.savefig(fname, dpi=150)
    plt.show()

    return fname

def plot_model_curves(h: pd.DataFrame, model_name: str) -> str:
    # Create a figure with two subplots (side by side)
    plt.figure(figsize=(14 , 5))
    # Plot Loss Curve (Train + Validation)
    plt.subplot(1, 2, 1) # 1 row, 2 Columns, first plot
    plt.plot(h['epoch'], h['train_loss'], marker="o", ms=4, label="Train", color="blue")
    plt.plot(h['epoch'], h['val_loss'], marker="s", ms=4, label="Validation", color="red")
    plt.title(f"Loss Curves - {model_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    # Plot Accuracy Curve (Train + Validation)
    plt.subplot(1, 2, 2) # 1 row, 2 Columns, second plot
    plt.plot(h['epoch'], h['train_acc'], marker="o", ms=4, label="Train", color="blue")
    plt.plot(h['epoch'], h['val_acc'], marker="s", ms=4, label="Validation", color="red")
    plt.title(f"Accuracy Curves - {model_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    # Adjust layout and show he plots
    plt.tight_layout()

    fname = f"loss_{model_name.replace(' ','_').lower()}.png"
    plt.savefig(fname, dpi=150)

    plt.show()

    return fname



# Load and preprocess the image
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)

# Predict using the model
def predict_(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()


def predict(model, image, device):
    model.eval()

    # NumPy -> Tensor if necessary
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    image = image.float()

    # (C, H, W) -> (1, C, H, W)
    if image.ndim == 3:
        image = image.unsqueeze(0)

    # (H, W) -> (1, 1, H, W)
    elif image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)

    return probabilities.cpu().numpy().flatten()

# Visualization
def visualize_predictions(original_image, probabilities, class_names, real_class):
    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))

    # Display image
    axarr[0].imshow(original_image)
    axarr[0].axis("off")
    axarr[0].set_title(f"Real Class: {real_class}")

    # Display predictions
    axarr[1].barh(class_names, probabilities)
    axarr[1].set_xlabel("Probability")
    axarr[1].set_title("Class Predictions")
    axarr[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()


# Visualization
def visualize_predictions_colored(original_image, probabilities, class_names, real_class):
    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))

    # --------------------------------------------------
    # Prepare image for matplotlib
    # --------------------------------------------------
    if isinstance(original_image, torch.Tensor):
        original_image = original_image.detach().cpu().squeeze().numpy()
    else:
        original_image = np.asarray(original_image).squeeze()

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    max_idx = np.argmax(probabilities)
    predicted_class = class_names[max_idx]

    is_correct = predicted_class == real_class
    title_color = "green" if is_correct else "red"

    # --------------------------------------------------
    # Left: Image
    # --------------------------------------------------
    axarr[0].imshow(original_image, cmap="gray")
    axarr[0].axis("off")

    axarr[0].set_title(
        f"Real: {real_class}\nPredicted: {predicted_class}",
        color=title_color,
        fontsize=14,
        fontweight="bold"
    )

    # --------------------------------------------------
    # Right: Bar chart
    # --------------------------------------------------
    bar_colors = []

    for i, cls in enumerate(class_names):
        prob = probabilities[i]

        if prob == 0:
            bar_colors.append("lightgray")

        elif cls == predicted_class:
            if predicted_class == real_class:
                bar_colors.append("green")
            else:
                bar_colors.append("red")

        elif cls == real_class:
            bar_colors.append("green")

        else:
            bar_colors.append("pink")

    bars = axarr[1].barh(
        class_names,
        probabilities,
        color=bar_colors
    )

    # Color y-axis labels
    for label in axarr[1].get_yticklabels():
        cls = label.get_text()

        if cls == real_class:
            label.set_color("green")
            label.set_fontweight("bold")

        elif cls == predicted_class and predicted_class != real_class:
            label.set_color("red")
            label.set_fontweight("bold")

    axarr[1].set_xlabel("Probability")
    axarr[1].set_title("Class Predictions")
    axarr[1].set_xlim(0, 1)

    # Show probability values
    for bar, prob in zip(bars, probabilities):
        axarr[1].text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.2f}",
            va="center"
        )

    plt.tight_layout()
    plt.show()


def test_eval(model, index, dataset, config, transforms=None):
    # Load a single image
    index = 4373  # must be < 5000 if using your split test_dataset
    input, label = dataset[index]

    # Keep tensor shape: (1, 28, 28)
    image_tensor = input

    probabilities = predict(model, image_tensor, config.device)

    print(f"True label: {label}")
    print(f"Predicted label: {probabilities.argmax()}")

    class_names = [str(i) for i in range(10)] # a_test_dataset.dataset.classes
    visualize_predictions_colored(
        input,
        probabilities,
        class_names,
        str(label)
    )

