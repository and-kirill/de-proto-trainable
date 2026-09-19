# Copyright (c) 2026 Kirill Andreev <and.kirill@gmail.com>
# SPDX-License-Identifier: MIT

"""
Visualize check node processing PDFs
"""
import time
from functools import wraps

import torch
import matplotlib.pyplot as plt
import numpy as np

from implementation.density_evolution import PdfFactory
from implementation.node_operations import cnop_sp_pairwise
from implementation.node_operations import pdf2tail_parts, tail_parts2pdf



def minsum_abs_tail(
        tail_x_pos, px_pos, tail_x_neg, px_neg, center_x,
        tail_y_pos, py_pos, tail_y_neg, py_neg, center_y):
    """
    Fast differentiable min-sum rule in sign-split unnormalized tail representation.
    Keeping tails unnormalized preserves the full-PDF subgradient at zero sign mass.
    """
    center = center_x + center_y - center_x * center_y

    p_pos = px_pos * py_pos + px_neg * py_neg
    p_neg = px_pos * py_neg + px_neg * py_pos

    tail_pos = tail_x_pos * tail_y_pos + tail_x_neg * tail_y_neg
    tail_neg = tail_x_pos * tail_y_neg + tail_x_neg * tail_y_pos

    return [tail_pos, p_pos, tail_neg, p_neg, center]


def time_it(func):
    """
    Print wallclock time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        duration = end_time - start_time
        print(f"'{func.__name__}' executed in {duration:1.4e} seconds.")
        return result
    return wrapper


def channel_tensor(snr_range, pdf_factory):
    """
    Generate a list of channel LLR PDFs in accordance with channel SNR
    """
    return torch.vstack([
        pdf_factory.bi_awgn_input(snr_db) for snr_db in snr_range
    ])


@time_it
def example_sp_cn(px, py, pdf_factory):
    """
    Example of SP check node processing
    """
    spa_idx, spa_w = pdf_factory.get_spa_tensor()
    pz = cnop_sp_pairwise(px, py, spa_idx, spa_w)
    return pz.detach().cpu().numpy()


@time_it
def example_ms_cn(px, py, pdf_factory):
    """
    Example of MS check node processing
    Convertion from PDF to CDF is used
    """
    px_parts = pdf2tail_parts(px, pdf_factory.n_points())
    py_parts = pdf2tail_parts(py, pdf_factory.n_points())
    pz_parts = minsum_abs_tail(*px_parts, *py_parts)
    return tail_parts2pdf(*pz_parts)

def make_plot(snr_range, llr_scale=20, llr_max=50):
    """
    Visualize MS and SP checnode processing over channel LLR PDFs
    """
    pdfgen = PdfFactory(
        llr_scale=llr_scale,
        llr_max=llr_max,
        device="cpu",
        dtype="float64"
    )

    px = channel_tensor(snr_range, pdfgen)
    pz_spa = example_sp_cn(px, px, pdfgen)
    pz_nms = example_ms_cn(px, px, pdfgen)

    for i in range(len(snr_range)):
        plt.plot(pdfgen.grid(), pz_spa[i], 'r-')
        plt.plot(pdfgen.grid(), pz_nms[i], 'g-')

    plt.grid()
    plt.xlim([-5, 15])
    plt.ylim([0, 0.025])
    plt.savefig("pdfs.png")
    plt.close()


if __name__ == "__main__":
    make_plot(
        np.arange(-4.0, 5.0),  # NSR range, dB
        llr_max=50,
        llr_scale=20
    )
