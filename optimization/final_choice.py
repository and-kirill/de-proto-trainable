# Copyright (c) 2026 Kirill Andreev <and.kirill@gmail.com>
# SPDX-License-Identifier: MIT

"""Selection of an integer parity-check matrix after optimization."""

import itertools
import os

import numpy as np

from tools.de_curve import evaluate_de


def choice(config, epoch, snr_db):
    """Select and save the best integer PCM from the final ensemble.

    The relaxed matrix is loaded from ``pcm_soft_<epoch>.txt`` in the
    configured data directory. Every non-integer entry is independently
    replaced with zero or one, and every resulting matrix is evaluated at
    ``snr_db`` using the decoder, DE grid, puncturing, and iteration count
    from ``config``. For NMS decoding, column scales are loaded from the same
    accepted epoch.

    The search is exhaustive and evaluates ``2**N`` matrices, where ``N`` is
    the number of non-integer entries. The best matrix found so far is saved
    after every improvement, and the final result is written to ``final.txt``
    in the configured data directory.

    Args:
        config: Complete training configuration used by the optimizer.
        epoch: Number of the final accepted training epoch.
        snr_db: Working signal-to-noise ratio in dB at the end of training.

    Returns:
        Path to the saved integer PCM file.
    """
    filename = os.path.join(config.data_dir, f"pcm_soft_{epoch}.txt")
    output_filename = os.path.join(config.data_dir, "final.txt")
    pcm_np = np.loadtxt(filename)
    col_scales = None
    if config.decoder.type == "nms":
        scales_filename = os.path.join(config.data_dir, f"scales_{epoch}.txt")
        col_scales = np.loadtxt(scales_filename)
    n_iter = config.optimization.n_iter

    ber = evaluate_de(
        pcm_np, snr_db, n_iter, config, col_scales=col_scales
    )
    print(f"Original BER: {ber:1.6e}")
    condition = (pcm_np - 1/2) ** 2 - 1/4 != 0
    n_pos = len(pcm_np[condition])
    pcm_test = np.copy(pcm_np)
    tuples = itertools.product([0, 1], repeat=n_pos)
    best_tuple = None
    best_ber = np.inf
    for values in tuples:
        pcm_test[condition] = np.array(values)
        ber = evaluate_de(
            pcm_test, snr_db, n_iter, config, col_scales=col_scales
        )
        if ber < best_ber:
            best_ber = ber
            best_tuple = values
            print("Best tuple updated", best_tuple)
            pcm_np[condition] = np.array(best_tuple)
            np.savetxt(output_filename, pcm_np, fmt="%d")
        print(values, f"BER: {ber:1.6e}")
    print(best_tuple)
    pcm_np[condition] = np.array(best_tuple)
    ber = evaluate_de(
        pcm_np, snr_db, n_iter, config, col_scales=col_scales
    )
    print(f"Final BER: {ber:1.6e}")
    np.savetxt(output_filename, pcm_np, fmt="%d")
    return output_filename
