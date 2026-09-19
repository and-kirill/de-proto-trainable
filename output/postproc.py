# Copyright (c) 2026 Kirill Andreev <and.kirill@gmail.com>
# SPDX-License-Identifier: MIT

"""
Visuyalize GD-based training process
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
from matplotlib.animation import PillowWriter  # pylint: disable=wrong-import-position


FPS = 24
OUT_FILENAME = "entropies"
OUT_MOV = "animated"


def entropy(p):
    """
    Evaluate the ensemble size
    """
    # Clip values very close to 0 and 1 to prevent log(0) errors
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def load_pcm_list(fig_dir):
    """
    Load the series of soft PCMs.
    """
    pcm_files = epoch_files(fig_dir, "pcm_soft")
    if not pcm_files:
        return None
    print(f"Found data for {len(pcm_files)} iterations")
    return np.stack([np.loadtxt(filename) for filename in pcm_files])


def epoch_files(src_dir, prefix):
    """Return history files ordered by their numeric epoch suffix."""
    files = Path(src_dir).glob(f"{prefix}_*.txt")
    return sorted(files, key=lambda path: int(path.stem.rsplit("_", 1)[1]))


def plot_entropies(pcm_list, fig_dir):
    """
    Plot per-row, per-column, per-matrix entropies (aka ensemble size)
    """
    h_elements = entropy(pcm_list)

    _, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # Plot 1
    axs[0].plot(np.sum(h_elements, axis=1))
    axs[0].set_title('Per-col entropies')

    # Plot 2
    axs[1].plot(np.sum(h_elements, axis=2))
    axs[1].set_title('Per-row entropies')

    axs[2].plot(np.sum(np.sum(h_elements, axis=2), axis=1))
    axs[2].set_title('Sum-element entropies')

    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    filename = f"{fig_dir}_{OUT_FILENAME}.png"
    plt.savefig(filename)
    plt.close()
    return f"Generated figure:    {filename}\n"


def plot_integer_fraction(src_dir):
    """
    Plot non-ineteger fraction and BER as a function of epoch count
    """
    sparsity = []
    ber_series = []
    epoch = 0
    while True:
        try:
            pcm = np.loadtxt(f"{src_dir}/pcm_soft_{epoch}.txt")
            ber = np.loadtxt(f"{src_dir}/ber_{epoch}.txt")
        except FileNotFoundError:
            break
        scores = 1/4 - (pcm - 1/2) ** 2
        n_converged = np.sum(scores < 1e-9)
        sparsity.append(n_converged / np.prod(pcm.shape))
        ber_series.append(ber)
        epoch += 1

    _, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    # Plot 1
    axs[0].plot(sparsity)
    axs[0].set_title('Ensemble integer-valued fraction')
    axs[0].grid()

    # Plot 2
    axs[1].semilogy(ber_series)
    axs[1].set_title('Predicted BER')
    axs[1].grid()

    filename = f"{src_dir}_int_fraction.png"
    plt.savefig(filename)
    plt.close()
    return f"Generated figure:    {filename}\n"

def plot_col_scales(src_dir):
    """
    Plot non-ineteger fraction and BER as a function of epoch count
    """
    scales = []
    epoch = 0
    while True:
        try:
            scales.append(np.loadtxt(f"{src_dir}/scales_{epoch}.txt"))
        except FileNotFoundError:
            break
        epoch += 1

    # Column scales are not applicable to sum product, no files will be stored
    if epoch == 0:
        return ""

    scales = np.array(scales)
    for i in np.arange(scales.shape[1]):
        plt.plot(scales[:, i])

    plt.grid()

    filename = f"{src_dir}_col_scales.png"
    plt.savefig(filename)
    plt.close()
    return f"Generated figure:    {filename}\n"


def write_animation(fig_dir):
    """
    Render a GIF directly from history files, without ffmpeg or temporary PNGs.
    """
    animation = Path(f"{fig_dir}_{OUT_MOV}.gif")
    pcm_files = epoch_files(fig_dir, "pcm_soft")
    if not pcm_files:
        return ""

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    writer = PillowWriter(fps=FPS)
    tiny = np.finfo(float).tiny

    try:
        with writer.saving(fig, animation, dpi=100):
            for pcm_file in pcm_files:
                epoch = int(pcm_file.stem.rsplit("_", 1)[1])
                grad_file = Path(fig_dir) / f"pcm_grad_{epoch}.txt"
                if not grad_file.is_file():
                    print(f"Skipping epoch {epoch}: {grad_file} is missing")
                    continue

                pcm = np.loadtxt(pcm_file)
                grad = np.loadtxt(grad_file)
                frame_data = (
                    (np.log(np.maximum(np.abs(grad), tiny)), {}),
                    (-np.sign(grad), {"vmin": -1, "vmax": 1}),
                    (pcm, {"vmin": 0, "vmax": 1}),
                )
                titles = (
                    "Estimated gradients (log-magnitudes)",
                    "Gradient signs",
                    "Parity check matrix",
                )
                for axis, (data, options), title in zip(
                    axes, frame_data, titles
                ):
                    axis.clear()
                    axis.imshow(data, cmap="coolwarm", **options)
                    axis.set_title(f"{title}, epoch {epoch}")
                writer.grab_frame()
    finally:
        plt.close(fig)

    return f"Generated animation: {animation}\n"


def count_flips(fig_dir):
    """
    Evaluate the equivalent bit flip count based on gradient signs changes
    """
    grad_list = []
    epoch = 0
    while True:
        filename = f"{fig_dir}/pcm_grad_{epoch}.txt"
        try:
            grad = np.loadtxt(filename)
        except FileNotFoundError:
            break
        grad_list.append(grad)
        epoch += 1
    grad_list = np.stack(grad_list, axis=0)
    grad_signs = np.sign(grad_list)
    diff = np.diff(grad_signs, axis=0)
    flip_cnt = np.sum(diff != 0, axis=0)
    flip_epochs = np.sum(np.sum(np.sum(diff, axis=2), axis=1) != 0)
    flip_per_epoch = np.mean(np.sum(np.sum(diff != 0, axis=2), axis=1))
    msg = "-" * 60 + "\n"
    msg += f"Max flips per entry:     {np.max(flip_cnt)}\n"
    msg += f"Total bits flipped:      {np.sum(diff != 0)}\n"
    msg += f"Total epochs with flips: {flip_epochs} / {epoch} epochs\n"
    msg += f"Average bits flipped:    {flip_per_epoch:1.3f} per epoch\n"
    return msg


def postproc_dir(dir_name):
    """
    Postproc single directory
    """
    pcm_list = load_pcm_list(dir_name)
    if pcm_list is None:
        return 0
    msg = "=" * 60 + "\n"
    msg += f"Processed directory: {dir_name}\n"
    msg += plot_entropies(pcm_list, dir_name)
    msg += plot_integer_fraction(dir_name)
    msg += plot_col_scales(dir_name)
    msg += write_animation(dir_name)
    msg += count_flips(dir_name)
    msg += "=" * 60
    return msg


def main():
    """
    Main function: oif directpry is not specified via argv, postproc all directories
    """
    if len(sys.argv) == 2:
        all_dirs = [sys.argv[1]]
    else:
        all_dirs = [d.name for d in Path('.').iterdir() if d.is_dir()]
    msgs = [0] * len(all_dirs)
    for i, dir_name in enumerate(all_dirs):
        msgs[i] = postproc_dir(dir_name)
    for i, dir_name in enumerate(all_dirs):
        print(msgs[i])

if __name__ == "__main__":
    main()
