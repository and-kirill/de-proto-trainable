# Copyright (c) 2026 Kirill Andreev <and.kirill@gmail.com>
# SPDX-License-Identifier: MIT

"""
Optimization config parser with sanity checks
"""
import os
from dataclasses import dataclass
import torch
import numpy as np

from optimization.lr_scheduler import LearningRateScheduleConfig
from optimization.convergence import ConvergenceConfig


OUTPUT_DIR = "output"

@dataclass
class DeGridConfig:
    """
    LLR grid configuration
    """
    llr_max: int    # Maximum considered LLR (higher values are clipped)
    llr_scale: int  # The number of grid points per unit LLR value (1 / llr_scale resolution)
    device: str     # Torch device ("cpu" or "cuda:0")
    dtype: str      # Torch datatype for PMF values ("float64" is preferrable)

    def __str__(self):
        msg = "-" * 24 + " LLR grid parameters: " + "-" * 24 + "\n"
        msg += f" Max. LLR value (clipping used):  {self.llr_max}\n"
        msg += f" LLR resolution:                  1/{self.llr_scale}\n"
        msg += f" Device:                          {self.device}\n"
        msg += f" PMF values data type:            {self.dtype}\n"
        return msg

    def __post_init__(self):
        device = torch.device(self.device)
        getattr(torch, self.dtype)  # Check for AttributeError
        if not self.device.startswith("cuda"):
            return
        if not torch.cuda.is_available():
            raise ValueError("Cuda is not available. Select CPU")
        if device.index >= torch.cuda.device_count():
            raise ValueError("Device count exceeds the number of available devices")


@dataclass
class ProtographConfig:
    """
    Protograph configuration
    """
    init_pcm: str       # Path for input parity check matrix. Loaded usingg np.loadtxt
    punctured: int      # The number of punctured columns. First columns are assumed punctured

    def __post_init__(self):
        if not os.path.isfile(self.init_pcm):
            raise ValueError("File containing input parity check matrix does not exist")
        self.pcm_np = np.loadtxt(self.init_pcm, dtype=np.float64)
        if np.any(self.pcm_np < 0) or np.any(self.pcm_np > 1):
            raise ValueError("Parity check matrix elements must be inside [0; 1] interval")
        if self.punctured < 0 or self.punctured > self.pcm_np.shape[1]:
            raise ValueError(f"The number of punctured positions ({self.punctured}) is invalid")

    def __str__(self):
        pcm_shape = self.pcm_np.shape
        coding_rate = (pcm_shape[1] - pcm_shape[0]) / (pcm_shape[1] - self.punctured)
        msg = "-" * 24 + " Decoder parameters: " + "-" * 25 + "\n"
        msg += f" Initiall PCM:                    {self.init_pcm}\n"
        msg += f" PCM shape                        {pcm_shape[0]} X {pcm_shape[1]}\n"
        msg += f" The number of punctured columns: {self.punctured}\n"
        msg += f" Coding rate:                     {coding_rate:1.3f}\n"
        return msg


@dataclass
class DecoderConfig:
    """
    Decoder configuration
    """
    type: str       # Supported: nms (flooding normalized min-sum), spa (flooding sum-product)
    col_scales: str = "" # Path to per-column normalization scales (applicable only to min-sum)
    train_scales: int = 0

    def __post_init__(self):
        if self.type not in ["spa", "nms"]:
            raise ValueError(f"Unknown decoder implementation: {self.type}")
        if self.type == "nms":
            self.col_scales_array = np.loadtxt(self.col_scales, dtype=np.float64)
            if np.min(self.col_scales_array) < 0.0: #  or np.max(self.col_scales_array) > 1.0:
                raise ValueError("The normalizing coefficient must be in (0, 1) interval")

    def __str__(self):
        if self.type.lower() == "nms":
            msg = "-" * 16 + " Normalized min-sum density evolution. " + "-" * 15 + "\n"
            val_min = np.min(self.col_scales_array)
            val_max = np.max(self.col_scales_array)
            msg += f" NMS check node update scales in [{val_min:1.3f}; {val_max:1.3f}] interval\n"
            msg += f" Column scales stored in:         {self.col_scales}\n"
            if self.train_scales:
                msg += " Train protograph and column scales jointly\n"
            else:
                msg += " Train protograph only. Scales remain constant.\n"
        else:
            msg = "-" * 19 + " Sum-product density evolution: "+ "-" * 19 + "\n"
        return msg


@dataclass
class OptimizerGdConfig:
    """
    Optimization configuration for Gradient descend (GD)-based optimizator
    """
    snr_db: float               # Operational signal-to-noise ratio
    n_iter: int                 # The number of decoding iterations
    ber_threshold: float        # Minimum BER value. If hit, SNR is decreased
    snr_step: float             # The value of SNR decrease when minimum BER is reached
    lr_schedule: LearningRateScheduleConfig | dict
    convergence: ConvergenceConfig | dict

    def __post_init__(self):
        if isinstance(self.lr_schedule, dict):
            self.lr_schedule = LearningRateScheduleConfig(**self.lr_schedule)
        if isinstance(self.convergence, dict):
            self.convergence = ConvergenceConfig(**self.convergence)
        if self.n_iter <= 1:
            raise ValueError("The number of decoding iterations must be positive integer")

    def __str__(self):
        msg = "-" * 11 + " Gradient descend-based optimization parameters: " + "-" * 10 + "\n"
        msg += f"  Initial Signal-to-Noise ratio: {self.snr_db:+1.3f}\n"
        msg += f"  Num. of decoding itertions:     {self.n_iter}\n"
        msg += "  SNR decrease policy:            "
        msg += f"at BER={self.ber_threshold:1.4e} by {self.snr_step:1.3f} dB\n"
        msg += str(self.lr_schedule)
        msg += str(self.convergence)
        return msg


class Config:  # pylint: disable=too-few-public-methods
    """
    Aggregated configuration, that includes:
     - LLR grid configuration
     - Protograph configuration
     - Decoder configuration
     - Optimizaer configuration
    """
    def __init__(self, epoch, **kwargs):
        self.grid = DeGridConfig(**kwargs["grid"])
        data_dir = os.path.join(OUTPUT_DIR, kwargs["data_dir"])
        self.data_dir = os.path.abspath(data_dir)
        if epoch > 0:
            kwargs["protograph"]["init_pcm"] = f"{self.data_dir}/pcm_soft_{epoch}.txt"
            kwargs["decoder"]["col_scales"] = f"{self.data_dir}/scales_{epoch}.txt"
        self.protograph = ProtographConfig(**kwargs["protograph"])
        self.decoder = DecoderConfig(**kwargs["decoder"])

        self.optimization = OptimizerGdConfig(**kwargs["optimization"])

    def __str__(self):
        msg = "Density evolution over relaxed protograph\n"
        msg += self.grid.__str__()
        msg += self.protograph.__str__()
        msg += self.decoder.__str__()
        msg += self.optimization.__str__()
        msg += "-" * 70 + "\n"
        msg += f"Data directory:  {self.data_dir}\n"
        msg += "-" * 70 + "\n"
        return msg
