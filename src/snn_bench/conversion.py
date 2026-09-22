import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import spikingjelly.activation_based.ann2snn as ann2snn

from .conversion_mod import recipes
from .conversion_mod import eval_conv_snn

__all__ = [
    "recipes",
    "eval_conv_snn",
]

from .config import Config




class Converter:
    """
    === ann2snn ===
    ['ChannelVoltageScaler', 'ConversionRecipe', 'Converter', 'FXConversionRecipe', 'FXConverter', 'LocalThresholdBalancingRecipe', 'ModuleConversionRecipe', 'ModuleConverter', 'Qwen2SNNCalibration', 'Qwen2SNNConfig', 'Qwen2SNNModel', 'Qwen2SNNRecipe', 'RateCodingRecipe', 'STATransformerRecipe', 'SignedQCFSSequenceEncoder', 'SpikeZIPTFQANNRecipe', 'TransformerTDEquivalentRecipe', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', 'calibrate_qwen2_snn', 'converter', 'delay', 'download_url', 'estimate_delay_start', 'modules', 'operators', 'qcfs', 'recipes', 'utils']

    === converter ===
    ['Converter', 'FXConversionRecipe', 'FXConverter', 'ModuleConversionRecipe', 'ModuleConverter', 'Optional', 'TransformerTDEquivalentRecipe', 'Union', '_FX_TRACE_LOCK', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '_resolve_device', '_symbolic_trace', 'fx', 'logger', 'nn', 'threading', 'time', 'torch', 'types']

    === modules ===
    ['ChannelVoltageScaler', 'VoltageHook', 'VoltageScaler', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 'base', 'functional', 'nn', 'torch']

    === utils ===
    ['__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '_download_without_resume', '_validate_download_response', 'download_url', 'logger', 'os', 're', 'requests', 'time', 'tqdm']

    """
    def handle_rate_coded(model: nn.Module, train_dataset: Dataset, config: Config, mode: str ="max"):
        calibration_data_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False
        )
        
        recipe = ann2snn.RateCodingRecipe(
            dataloader=calibration_data_loader,
            mode="max",
        )
        converted_snn = ann2snn.FXConverter(recipe).convert(model)
        
        return converted_snn