# Copyright (c) 2026 Kirill Andreev <and.kirill@gmail.com>
# SPDX-License-Identifier: MIT

"""
Node operations for quantized density evolution.
Functions below operate over both relaxed / unrelaxed graphs
"""

import torch
import numpy as np


###############################################################################
# Variable node routines
###############################################################################

def convolve_fft(pdf_x_fft, pdf_y_fft, n_points):
    """
    FFT-based convolution
    Inputs are presented in frequency domain
    For non-cyclic convolution and a pairwise application, the FFT size should be doubled
    """
    # pylint: disable=e1102
    n_total = 2 * n_points + 1
    fft_size = 2 * n_total
    pdf_z = torch.fft.irfft(pdf_x_fft * pdf_y_fft, n=fft_size, dim=-1)

    return torch.cat([
        pdf_z[..., : (n_points + 1)].sum(dim=-1, keepdim=True),
        pdf_z[..., (n_points + 1) : (n_total + n_points - 1)],
        pdf_z[..., (n_total + n_points - 1) :].sum(dim=-1, keepdim=True),
    ], dim=-1)


def vnop_pairwise(r_msg, input_fft):
    """
    Variable node operation with pairwise application of convolutions
    This version is more memory efficient
    especially when considering backpropagation
    """
    # pylint: disable=e1102
    n_checks = r_msg.shape[0]
    n_points = r_msg.shape[2] // 2
    fft_size = 2 * (2 * n_points + 1)

    r_msg = r_msg.permute(1, 0, 2)
    r_msg_fft = torch.fft.rfft(r_msg, n=fft_size, dim=-1)

    prefix = [r_msg[:, 0, :]]
    for i in range(1, n_checks - 1):
        # pylint: disable=e1102
        pdf_x_fft = torch.fft.rfft(prefix[-1], n=fft_size, dim=-1)
        prefix.append(convolve_fft(pdf_x_fft, r_msg_fft[:, i, :], n_points))

    suffix = [None] * (n_checks - 1)
    suffix[-1] = r_msg[:, -1, :]
    for i in reversed(range(1, n_checks - 1)):
        pdf_x_fft = torch.fft.rfft(suffix[i], n=fft_size, dim=-1)
        suffix[i - 1] = convolve_fft(pdf_x_fft, r_msg_fft[:, i, :], n_points)

    out_list = [suffix[0]]
    for i in range(1, n_checks - 1):
        pdf_x_fft = torch.fft.rfft(prefix[i - 1], n=fft_size, dim=-1)
        pdf_y_fft = torch.fft.rfft(suffix[i], n=fft_size, dim=-1)
        out_list.append(convolve_fft(pdf_x_fft, pdf_y_fft, n_points))
    out_list.append(prefix[-1])

    out = torch.stack(out_list, dim=0)
    out_fft = torch.fft.rfft(out, n=fft_size, dim=-1)

    q_msg_tensor = torch.abs(convolve_fft(out_fft, input_fft, n_points))
    return q_msg_tensor / q_msg_tensor.sum(dim=-1, keepdim=True)


def vnop_full(r_msg, input_fft):
    """
    Variable node operation using a full-size FFT
    !NOTE: With the use opf backpropagation, it may result to OOM error
    input_fft must be of proper size
    """
    # pylint: disable=e1102

    n_checks = r_msg.shape[0]
    n_points = r_msg.shape[2] // 2

    n_total = 2 * n_points + 1
    fft_size = n_total * n_checks

    fourier = torch.fft.rfft(r_msg.permute(1, 0, 2), n=fft_size, dim=-1)
    prefix = torch.cumprod(fourier, dim=1)
    suffix = torch.cumprod(fourier.flip(dims=[1]), dim=1).flip(dims=[1])

    except_i_fft = torch.cat([
        suffix[:, 1:2],
        prefix[:, :-2] * suffix[:, 2:],
        prefix[:, -2:-1]
    ], dim=1)

    # NOTE: negative values may appear here (of magnitude close to eps)
    result = torch.fft.irfft(except_i_fft * torch.unsqueeze(input_fft, 1), dim=-1)
    result = result / torch.sum(result, dim=-1, keepdim=True)

    i_start = (n_checks - 1) * n_points + 1
    i_end   = n_total + i_start - 2

    result = torch.cat([
        result[..., :i_start].sum(dim=-1, keepdim=True),
        result[..., i_start:i_end],
        result[..., i_end:].sum(dim=-1, keepdim=True),
    ], dim=-1)

    return result.permute(1, 0, 2)


def output_ber(r_msg, input_distr):
    """
    Evaluate the output BER given R-messages PDFs and channel distrinbution
    """
    # pylint: disable=e1102
    n_checks = r_msg.shape[0]
    n_points = r_msg.shape[2] // 2
    n_total = 2 * n_points + 1
    fft_size = 2 * n_total

    r_fft = torch.fft.rfft(r_msg, n=fft_size, dim=-1)
    llr_distr = input_distr
    for j in range(n_checks):
        llr_distr = convolve_fft(torch.fft.rfft(llr_distr, n=fft_size, dim=-1), r_fft[j], n_points)

    return llr_distr[:, : n_points + 1].sum(dim=-1).mean()


###############################################################################
# MIN-SUM check node routines
# For min-sum based check node, we use a low-complexity implementation via
# CCDF product
###############################################################################


def pdf2ccdf(x, flip=False):
    """
    Convert PDF into complementary CDF along the last dimension.
    """
    if flip:
        cdf = torch.cumsum(torch.flip(x, dims=(-1,)), dim=-1)
    else:
        cdf = torch.cumsum(x, dim=-1)
    prob = cdf[..., -1:]
    denom = torch.where(prob > 0, prob, torch.ones_like(prob))
    return 1.0 - cdf / denom, prob


def pdf2tail_parts(pdf, n_points):
    """
    Convert full PDF into sign-split unnormalized tail representation.
    """
    cx_pos, px_pos = pdf2ccdf(pdf[..., n_points + 1 :], flip=False)
    cx_neg, px_neg = pdf2ccdf(pdf[..., :n_points], flip=True)
    center = pdf[..., n_points : n_points + 1]
    return [px_pos * cx_pos, px_pos, px_neg * cx_neg, px_neg, center]


def cdf2pdf(x, flip=False):
    """
    Convert CDF to PDF along the last dimension.
    """
    pdf = torch.cat([x[..., :1], x[..., 1:] - x[..., :-1]], dim=-1)
    if flip:
        return torch.flip(pdf, dims=(-1,))
    return pdf

def tail_parts2pdf(tail_pos, p_pos, tail_neg, p_neg, center):
    """
    Convert sign-split unnormalized tail representation into full PDF.
    """
    neg = cdf2pdf(p_neg - tail_neg, flip=True)
    pos = cdf2pdf(p_pos - tail_pos, flip=False)
    return torch.cat([neg, center, pos], dim=-1)


def cumprod_ps(tensor):
    """
    Calculate a prefix-suffix for cumprod operation
    """
    pref = torch.cumprod(tensor, dim=1)
    suff = torch.cumprod(tensor.flip(dims=[1]), dim=1).flip(dims=[1])

    batch, _, dim = tensor.shape

    pref_shifted = torch.cat([torch.ones(batch, 1, dim, device=tensor.device), pref[:, :-1]], dim=1)
    suff_shifted = torch.cat([suff[:, 1:], torch.ones(batch, 1, dim, device=tensor.device)], dim=1)

    return pref_shifted * suff_shifted


def prdf2cumprod_parts(q_msg):
    """
    Calcilate four cumprod parts of CDFs for all combinations of signs
    """
    n_points = q_msg.shape[-1] // 2
    tail_x_pos, px_pos, tail_x_neg, px_neg, center_x = pdf2tail_parts(q_msg, n_points)

    return [
        cumprod_ps(px_pos + px_neg),         # Sum  probability
        cumprod_ps(px_pos - px_neg),         # Diff probability
        cumprod_ps(tail_x_pos + tail_x_neg), # Sum  tail
        cumprod_ps(tail_x_pos - tail_x_neg), # Diff tail
        1.0 - cumprod_ps(1.0 - center_x)     # Center prob
    ]


def cnop_ms(q_msg):
    """
    Min-sum check node update rule
    """

    except_p_sum, except_p_diff, except_t_sum , except_t_diff, center_x = prdf2cumprod_parts(q_msg)

    return tail_parts2pdf(
        0.5 * (except_t_sum + except_t_diff), # Positive tail PMF
        0.5 * (except_p_sum + except_p_diff), # Positive tail prob
        0.5 * (except_t_sum - except_t_diff), # Negative tail PMF
        0.5 * (except_p_sum - except_p_diff), # Negative tail prob
        center_x                              # Center probability
    )


###############################################################################
# SUM-PRODUCT check node routines
# We use O(N2) implementation with pointwise function evaluation
# for a sequence of argument pairs
###############################################################################


def spa_fcn_stable(x1, x2):
    """
    Stable implementation of sum-product rule.
    Required to generate a tensor for pairwise sum-product application
    """
    a = np.abs(x1)
    b = np.abs(x2)

    res = (
        np.sign(x1) * np.sign(x2)
        * (
            np. minimum(a, b)
            + np.logaddexp (0.0, -(a + b) )
            - np.logaddexp(0.0, -np.abs(a - b))
        )
    )
    return res


@torch.compile
def cnop_sp_pairwise(px, py, spa_idx, spa_w):
    """
    Process sum-product check node for a pair of PDF tensor
    Fully differentiable version using scatter_add_
    """
    dim, grid = px.shape

    p_cross = px[:, :, None] * py[:, None, :]
    res_tensor = p_cross[:, :, :, None] * spa_w[None, :, :, :]

    batch_offsets = torch.arange(dim, device=px.device)[:, None, None, None] * grid
    flat_indices = (spa_idx[None, :, :, :] + batch_offsets).ravel()
    flat_values = res_tensor.ravel()

    # Create an empty tensor that will safely accept and pass gradients
    flat_pz = torch.zeros(dim * grid, dtype=flat_values.dtype, device=px.device)
    # Differentiable alternative to torch.bincount
    flat_pz.scatter_add_(0, flat_indices, flat_values)

    return flat_pz.reshape(dim, grid)


def cnop_sp(q_msg_w, spa_idx, spa_w):
    """
    Sum-product with pairwise application of 2atanh(tanh(x1/2)tanh(x2/2))
    The function is represented as a tensor
    pz_k = A_{ijk} * px_i * py_j
    """
    # Vectorize left2right:
    prefix = [q_msg_w[:, 0]]
    degree = q_msg_w.shape[1]
    for i in range(1, degree - 1):
        prefix.append(cnop_sp_pairwise(
                prefix[-1], q_msg_w[:, i], spa_idx, spa_w
            )
        )

    suffix = [None] * (degree - 1)
    suffix[-1] = q_msg_w[:, degree - 1]
    for i in reversed(range(1, degree - 1)):
        suffix[i - 1] = cnop_sp_pairwise(
            suffix[i],  q_msg_w[:, i], spa_idx, spa_w
        )

    output_dist = [suffix[0]]
    for i in range(1, degree - 1):
        output_dist.append(
            cnop_sp_pairwise(prefix[i - 1], suffix[i], spa_idx, spa_w)
        )
    output_dist.append(prefix[-1])
    return torch.stack(output_dist, dim=1)
