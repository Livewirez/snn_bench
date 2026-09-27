# snn_bench
A Project for independently testable components for data ingestion, ANN training, ANN to SNN conversion, evaluation and results visualization.


## Installation

```bash
pip install git+https://github.com/fangwei123456/spikingjelly.git
pip install snn-bench
```

## Troubleshooting
For issues related to dependencies or versions, you can use:

```bash
pip install snn-bench --upgrade
```


# Example Usage
```python

!pip install torchsummary
!pip install torchmetrics
!pip install git+https://github.com/fangwei123456/spikingjelly.git

%matplotlib inline

# import
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import numpy as np
from torchsummary import summary # Libraries for previewing neural network architectures
from spikingjelly.activation_based import neuron, encoding, functional, surrogate
from spikingjelly import visualizing
from matplotlib import pyplot as plt
import time
import os, re, random, json
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

!pip install snn-bench --upgrade

train_loader, val_loader, test_loader, meta = snn_bench.data.get_loaders(name='mnist') # as well as kmnist and fashionmnist
NUM_CLASSES  = meta.num_classes
CLASS_NAMES  = numeric_labels
print(f"Classes ({NUM_CLASSES}): {CLASS_NAMES}")
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

cfg = config.Config(NUM_CLASSES, CLASS_NAMES, "./model_save/checkpoints")

ann = models.ann.ANN(NUM_CLASSES)

#options = train_ann.TrainOptions.get_default()

ann_history_results, ann_evaluation_results, ann_results = train_ann.start_training_ann_loop(ann, train_loader, val_loader, test_loader, cfg)


snn = models.snn.DirectSNN(NUM_CLASSES)
snn.to(cfg.device)

snn_options = train_snn.TrainOptions.get_default_none()

loss_train_store, loss_cv_store, accuracy_rate_train_store, accuracy_rate_cv_store, epoch = train_snn.start_training_snn_loop(snn, train_loader, test_loader, cfg, snn_options)


#ANN2SNN
from snn_bench.conversion import Converter
import snn_bench.conversion_mod.recipes as snn_bench_recipes

conv_snn_max = Converter.handle_rate_coded(ann, train_dataset, cfg)
conv_snn_99 =  Converter.handle_rate_coded(ann, train_dataset, cfg, mode="99.9%")
conv_snn_half =  Converter.handle_rate_coded(ann, train_dataset, cfg, mode=1.0/2)
conv_snn_balanced = Converter.handle_rate_coded_threshold_balanced(ann, train_dataset, cfg)

ann_bench_result, totals, snn_models_bench_results = Benchmark.run(ann, {
    "Converted (max-norm)": conv_snn_max,
    "Converted (99.9% norm.)": conv_snn_99,
    "Converted (Half)": conv_snn_half,
    "Converted (balanced)": conv_snn_balanced,
    "Converted (DirectSNN)": snn,
}, test_loader, cfg.device)


from snn_bench.benchmark import BenchLevel 


Benchmark.query_levels(totals, BenchLevel.LEVEL_ONE)

ret = Benchmark.accuracy_retention(
    ann_results['ANN'].accuracy,  # baseline accuracy as a float
    {
        "Converted (max-norm)": conv_snn_max,
        "Converted (99.9% norm.)": conv_snn_99,
        "Converted (Half)": conv_snn_half,
        "Converted (balanced)": conv_snn_balanced,
        "Converted (DirectSNN)": snn,
    },    # the model itself
    loader=test_loader, device=cfg.device,
    Ts=[1, 2, 4, 8, 16, 32, 64],
)

print(ret.to_string(index=False))
print(Benchmark.best_retention(ret).to_string(index=False))

Benchmark.plot_accuracy_retention(ret)                 # picks the best T for the bar chart
Benchmark.plot_accuracy_retention(ret, at_T=8)         # or force one


ann_eff_raw = Benchmark.calculate_energy_efficiency(ann_bench_result["accuracy"],ann_bench_result["totals"]["E_total"])

eff = Benchmark.calculate_energy_efficiency_from_summary(
    snn_models_bench_results["Converted (99.9% norm.)"][0], n_classes=NUM_CLASSES, min_accuracy=0.5
)
Benchmark.plot_energy_efficiency(eff, ann_efficiency=ann_eff, prefix="Converted (99.9% norm.) — ")

from pathlib import Path
filepath_l1 = Path("results/level_one.csv")
filepath_l2 = Path("results/level_two.csv")
filepath_l3 = Path("results/level_thee.csv")
filepath_l1.parent.mkdir(parents=True, exist_ok=True)
filepath_l2.parent.mkdir(parents=True, exist_ok=True)
filepath_l3.parent.mkdir(parents=True, exist_ok=True)

l1 = Benchmark.query_levels(totals, BenchLevel.LEVEL_ONE)
l1.to_csv(filepath_l1)
l2 = Benchmark.query_levels(totals, BenchLevel.LEVEL_TWO)
l2.to_csv(filepath_l2)
l3 = Benchmark.query_levels(totals, BenchLevel.LEVEL_THREE)
l3.to_csv(filepath_l3)
```