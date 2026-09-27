from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Optional, Union, Iterable, Sequence

from .config import DEVICE

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
    plot_retention_vs_energy,
    crossing_points,
    
    profile_ann,
    totals_at,
    ann_baseline,
    build_totals,
    memory_summary,
    plot_totals,
    plot_memory_accesses,
    level_columns,
    query_levels,
    calculate_energy_efficiency,
    calculate_energy_efficiency_reference,
    calculate_energy_efficiency_from_summary,
    plot_energy_efficiency,
    
    BenchLevel, 
    BenchColumnMapper,
)

class Benchmark:
    
    def run(
        ann_model: nn.Module, snn_models: Dict[str, nn.Module], 
        test_dataloader: DataLoader, device: Union[torch.device, str] = DEVICE,
        at_T: int | Iterable[int] | None = None, **run_benchmark_args
    ):
        ann_bench_result = profile_ann(ann_model, test_dataloader, device)
        #   ann["accuracy"]   -> float, e.g. 0.9864
        #   ann["totals"]     -> dict of analytical energy totals
        #   ann["per_layer"]  -> DataFrame, one row per layer
        #   ann["empirical"]  -> Level 3 dict
        #   ann["point"]      -> one-row DataFrame
        
        snn_models_bench_results = {}
        
        for name, snn in snn_models.items():
            sum_metrics, layers = run_benchmark(snn, test_dataloader, device, **run_benchmark_args)
            snn_models_bench_results[name] = (sum_metrics, layers)

        totals = build_totals(ann_bench_result, snn_models_bench_results)
        print(totals.to_string(index=False))
        print(memory_summary(ann_bench_result, totals, at_T=at_T).to_string(index=False))

        plot_totals(ann_bench_result, totals, at_T=at_T)
        plot_memory_accesses(ann_bench_result, snn_models_bench_results, at_T=at_T)
        
        return ann_bench_result, totals, snn_models_bench_results
        
    def build_totals(ann_bench_result: Dict, models: Dict[str, Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]]) -> pd.DataFrame:
        return build_totals(ann_bench_result, models)
    
    def totals_at(totals: pd.DataFrame, at_T: int | Iterable[int] | None = None) -> pd.DataFrame:
        return totals_at(totals, at_T)
    
    
    def query_levels(
        totals: pd.DataFrame,
        level: Union[BenchLevel, Iterable[BenchLevel]] = BenchLevel.LEVEL_ALL,
        models: Optional[Union[str, Iterable[str]]] = None,
        Ts: Optional[Union[int, Iterable[int]]] = None,
        include_ann: bool = True,
        dropna_columns: bool = True
    ) -> pd.DataFrame:
        return query_levels(
            totals,
            level=level,
            models=models,
            Ts=Ts,
            include_ann=include_ann,
            dropna_columns=dropna_columns
        )
        
    def plot_totals(ann_bench_result: Dict, totals: pd.DataFrame, at_T: int | Iterable[int] | None = None):
        return plot_totals(ann_bench_result, totals, at_T)
        
    def plot_memory_accesses(ann_bench_result: Dict, models: Dict[str, Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]], at_T: int | Iterable[int] | None = None):
        return plot_memory_accesses(ann_bench_result, models, at_T)
        
    def memory_summary(ann_bench_result: Dict, totals: pd.DataFrame, at_T: int | Iterable[int] | None = None) -> pd.DataFrame:
        return memory_summary(ann_bench_result, totals, at_T)
    
    def run_benchmark(
        snn: nn.Module, loader: DataLoader, device: Union[torch.device, str] = DEVICE,
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
        device: Optional[Union[torch.device, str]] = DEVICE,
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
        
    def calculate_energy_efficiency(
        accuracy: Union[float, Sequence[float]],
        energy_joules: Union[float, Sequence[float]],
        chance_level: Optional[float] = None,
        min_accuracy: Optional[float] = None
    ):
        return calculate_energy_efficiency(
            accuracy, energy_joules, chance_level, min_accuracy
        )
        
    def calculate_energy_efficiency_from_summary(
        summary: pd.DataFrame,
        n_classes: int = 10,
        min_accuracy: Optional[float] = 0.5,
        energy_col: str = "E_snn"
    ) -> pd.DataFrame:
        return calculate_energy_efficiency_from_summary(
            summary, n_classes, min_accuracy, energy_col
        )
    
    def calculate_energy_efficiency_reference(ann_bench_result: Dict, n_classes: int =10, energy_key: str="E_total", chance: bool =True):
        return calculate_energy_efficiency_reference(
            ann_bench_result, n_classes, energy_key, chance
        )
        
    def plot_energy_efficiency(eff: pd.DataFrame, ann_efficiency: Optional[float] = None, prefix: str = ""):
        return  plot_energy_efficiency(
            eff, ann_efficiency, prefix
        )
        
    def crossing_points(ann_result: Union[Dict, pd.DataFrame], summary: pd.DataFrame, acc_tolerance: float = 0.01) -> Dict:
        return crossing_points(ann_result, summary, acc_tolerance)
        
        
        
__all__ = [
    "Benchmark",
    
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
    "plot_retention_vs_energy",
    
        
    "profile_ann",
    "totals_at",
    "ann_baseline",
    "build_totals",
    "memory_summary",
    "plot_totals",
    "plot_memory_accesses",
    "level_columns",
    "query_levels",
    "calculate_energy_efficiency",
    "calculate_energy_efficiency_reference",
    "calculate_energy_efficiency_from_summary"
    "plot_energy_efficiency"
    
    "BenchLevel", 
    "BenchColumnMapper",
]

