from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Optional, Union

from .bench import (
    
    EnergyModel,
    LayerSpec,
    INT32_MODEL,
    
    accuracy_vs_T,
    run_benchmark,
    plot_all,
    
    count_activity,
    validate_activity,
    level1_summary,
    
    snn_layer_counts,
    _dense_counts,
    ann_layer_counts,
    layer_memory_kB,
    counts_to_energy,
    analytical_energy,
    ann_baseline,
    
    _PowerSampler,
    _energy_J,
    measure_idle_power,
    measure_empirical,
    
    Source,
    _label,
    accuracy_retention,
    best_retention,
    plot_accuracy_retention,
    plot_retention_vs_energy
)


__all__ = [
    "accuracy_vs_T",
    "run_benchmark",
    "plot_all",
    
    "count_activity",
    "validate_activity",
    "level1_summary",
    
    "snn_layer_counts",
    "_dense_counts",
    "ann_layer_counts",
    "layer_memory_kB",
    "counts_to_energy",
    "analytical_energy",
    "ann_baseline",
    
    "_PowerSampler",
    "_energy_J",
    "measure_idle_power",
    "measure_empirical",
    
    "Source",
    "_label",
    "accuracy_retention",
    "best_retention",
    "plot_accuracy_retention",
    "plot_retention_vs_energy"
]

class Benchmark:
    def run(
        snn: nn.Module, loader: DataLoader, device: Union[torch.device, str],
        Ts: List[int] = (1, 2, 4, 8, 16, 32, 64),
        max_batches_l1: Optional[int] = 20,
        energy_model: EnergyModel = INT32_MODEL,
        run_level3: bool = True,
        l3_min_duration_s: float = 20.0,
        l3_repeats: int = 3,
        measure_accuracy: bool = True,
        acc_max_batches: Optional[int] = None,
        fc_acc_literal: bool = False,
        strict_validation: bool = True,
        verbose: bool = True
    ):
        return run_benchmark(
            snn, loader, device,
            Ts=Ts,
            max_batches_l1=max_batches_l1,
            energy_model=energy_model,
            run_level3=run_level3,
            l3_min_duration_s=l3_min_duration_s,
            l3_repeats=l3_repeats,
            measure_accuracy=measure_accuracy,
            acc_max_batches=acc_max_batches,
            fc_acc_literal=fc_acc_literal,
            strict_validation=strict_validation,
            verbose=verbose
        )
        
    
    def plot_all(df: pd.DataFrame, prefix: str = ""):
        return plot_all(df, prefix)
    
    def accuracy_retention(
        baseline: Source,
        models: Dict[str, Source],
        loader: Optional[DataLoader] = None,
        device: Optional[Union[torch.device, str]] = None,
        Ts: Optional[List[int]] = None,
        max_batches: Optional[int] = None
    ) -> pd.DataFrame:
        return accuracy_retention(
            baseline, models,
            loader=loader, device=device,
            Ts=Ts, max_batches=max_batches
        )
    
    def best_retention(df: pd.DataFrame) -> pd.DataFrame:
        return best_retention(df)

    def plot_accuracy_retention(
        df: pd.DataFrame,
        annotate: bool = True,
        at_T: Optional[int] = None,
        target: float = 0.95
    ):
        return plot_accuracy_retention(df, annotate=annotate, at_T=at_T, target=target)

    def plot_retention_vs_energy(
        df: pd.DataFrame,
        summaries: Dict[str, pd.DataFrame],
        E_ann: float
    ): 
        return plot_retention_vs_energy(
            df, summaries, E_ann
        )