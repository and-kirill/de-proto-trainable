# Copyright (c) 2026 Kirill Andreev <and.kirill@gmail.com>
# SPDX-License-Identifier: MIT

"""
Print DE-predicted BER for different setups
"""

import torch
import numpy as np

from implementation.density_evolution import DensityEvolutionNms, DensityEvolutionSpa


def evaluate_de(pcm_np, snr_db, n_iter, config, col_scales=None):
    """
    Get DE-predicted BER for single point
    """
    if config.decoder.type == "nms":
        if col_scales is None:
            col_scales = config.decoder.col_scales_array
        de = DensityEvolutionNms(
            pcm_np,
            config.protograph.punctured,
            config.grid,
            col_scales
        )
    else:
        de = DensityEvolutionSpa(
            pcm_np,
            config.protograph.punctured,
            config.grid
        )
    with torch.no_grad():
        loss = de.loss(n_iter, snr_db)
        ber = np.exp(loss.detach().cpu().numpy())
    return ber


def de_curves(filename, snr_range, iter_range, config, col_scales=None):
    """
    Print predicted DE performance for different iteration count
    """

    pcm_np = np.loadtxt(filename)

    print("SNR ", end="")
    for i in iter_range:
        print(f"{i} ", end="")
    print("")
    for snr_db in snr_range:
        print(f"{snr_db:1.3f} ", end="")
        for n_iter in iter_range:
            ber = evaluate_de(
                pcm_np, snr_db, n_iter, config, col_scales=col_scales
            )
            print(f"{ber:1.6e} ", end="")
        print("")
