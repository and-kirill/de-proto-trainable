# Copyright (c) 2026 Kirill Andreev <and.kirill@gmail.com>
# SPDX-License-Identifier: MIT

"""
convolve_ffterentiable density evolution implementation with soft parity check matrix elements
Each soft element is interpreted as a probability of the correspoding element to be one
Density evolution considers a proto-graph and flooding normalized min-sum decoding
"""
import torch
import numpy as np
from scipy.special import erf  # pylint: disable=no-name-in-module

from .node_operations import vnop_pairwise as vnop  # Use pairwise VNOP
from .node_operations import cnop_ms, output_ber
from .node_operations import spa_fcn_stable, cnop_sp


class PdfFactory:
    """
    Probability mass function factory with some useful distributions
    """
    def __init__(self, llr_scale, llr_max, device, dtype):
        self.llr_scale = llr_scale
        self.llr_max = llr_max
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)

    def n_points(self):
        """
        Get the nu,ber of points in negative/positive tails
        """
        return self.llr_scale * self.llr_max

    def shape(self):
        """
        Get total number of points in LLR PMF. Odd number always
        """
        return 2 * self.n_points() + 1

    def grid(self):
        """
        Get list of LLR points wehere PDF mass is concentrated
        """
        return np.arange(self.shape()) / self.llr_scale - self.llr_max

    def numpy2torch(self, x, dtype=None):
        """
        Convert numpy array to torch tensor with proper device mapping
        """
        if dtype is None:
            dtype=self.dtype
        return torch.as_tensor(x, device=self.device, dtype=dtype)

    def bi_awgn_input(self, snr_db):
        """
        QWGN-BPSK LLR distribution under all-zero codeword assumption
        """
        sigma = np.sqrt(10 ** (-snr_db / 10) / 2)
        llr_mean = 2.0 / sigma**2
        llr_sigma = np.sqrt(2.0 * llr_mean)
        # Make a linear transform to standard Gaussian
        vals = (self.grid() + 1.0 / (2 * self.llr_scale) - llr_mean) / (np.sqrt(2.0) * llr_sigma)
        # Calculate CDF values
        cdf_vals = erf(vals[:-1]) / 2.0 + 0.5
        # Calculate PDF -- a difference of CDF
        pdf_vals = np.hstack([cdf_vals, [1]]) - np.hstack([[0], cdf_vals])
        return self.numpy2torch(pdf_vals)

    def scaling_tensor(self, nms_scale):
        """
        Create a differentiable linear-interpolation tensor for NMS scaling.

        Rows select output LLR bins and columns select input bins. Integer
        bin selection is piecewise constant, while interpolation weights keep
        their autograd connection to ``nms_scale``.
        """
        int_shift = self.llr_scale * self.llr_max
        grid_shape = 2 * int_shift + 1
        scale = torch.as_tensor(nms_scale, device=self.device, dtype=self.dtype)
        offsets = torch.arange(
            -int_shift, int_shift + 1, device=self.device, dtype=self.dtype
        )
        output_positions = offsets * scale + int_shift
        lower_float = torch.floor(output_positions)
        upper_weight = output_positions - lower_float

        lower_idx = lower_float.to(torch.long).clamp(0, grid_shape - 1)
        upper_idx = (lower_idx + 1).clamp(0, grid_shape - 1)
        columns = torch.arange(grid_shape, device=self.device)
        flat_indices = torch.cat([
            lower_idx * grid_shape + columns,
            upper_idx * grid_shape + columns,
        ])
        weights = torch.cat([1.0 - upper_weight, upper_weight])
        flat_tensor = torch.zeros(
            grid_shape * grid_shape, device=self.device, dtype=self.dtype
        )
        return flat_tensor.scatter_add(0, flat_indices, weights).reshape(
            grid_shape, grid_shape
        )

    def trainable_col_scales(self, col_scales):
        """Build and stack interpolation tensors for trainable NMS scales."""
        scaling_tensors = [self.scaling_tensor(s) for s in col_scales]
        return torch.stack(scaling_tensors)


    def get_spa_tensor(self):
        """
        Get sum-product interpolated indices tensor
        pz_k = A_{ijk} * px_i * py_j
        """
        int_shift = self.llr_scale * self.llr_max
        grid_shape = 2 * int_shift + 1
        x_series = np.arange(grid_shape) / self.llr_scale - self.llr_max

        vals = spa_fcn_stable(x_series, x_series.reshape(-1, 1))
        idx_float = vals * self.llr_scale + int_shift
        idx_int = np.floor(idx_float).astype(np.int32)
        delta_float = np.expand_dims(idx_float - idx_int, -1)

        # Triangular interpolation window (aka linear interpolation)
        idx_shift = np.array([[[0, 1]]])
        w_interp = np.maximum(1.0 - np.abs(idx_shift - delta_float), 0)
        idx_tensor = idx_shift + np.expand_dims(idx_int, -1)

        if self.shape() > torch.iinfo(torch.int16).max:
            raise ValueError("Sum product indices assumed as int16. Type overflow may happen")

        return (
            self.numpy2torch(idx_tensor, dtype=torch.int16), # Indices over third dimension
            self.numpy2torch(w_interp) # Values over all three dimensions
        )

    def delta_input(self):
        """
        Zero-LLR concentrated delta function (empty node output)
        """
        pdf = torch.zeros(self.shape(), device=self.device, dtype=self.dtype)
        pdf[self.llr_max * self.llr_scale] = 1.0
        return pdf

    def infty_input(self):
        """
        PMF with a whole mass concentrated at maximum LLR value
        """
        pdf = torch.zeros(self.shape(), device=self.device, dtype=self.dtype)
        pdf[-1] = 1.0
        return pdf


class DensityEvolutionBase:
    """
    Flooding-based density evolution with trainable PCM.
    Implements basic procedures. NMS/SPA implementation are dericed classes
    """

    def __init__(self, pcm_np, punctured, grid_params):
        # Density evolution grid parameters
        self.params = PdfFactory(**grid_params.__dict__)
        # Create trainable parity check matrix
        self.pcm_soft = torch.tensor(
            pcm_np,
            device=self.params.device, dtype=self.params.dtype,
            requires_grad=True
        )
        self.punctured = punctured

        # Tensors required for density evolution
        self.r_msg = torch.zeros(
            (*self.pcm_soft.shape, self.params.shape()),
            device=self.params.device, dtype=self.params.dtype
        )
        self.q_msg = torch.zeros_like(self.r_msg)

    # @torch.compile
    def run(self, n_iter, snr_db):
        """
        Density evolution single-run given the number of iterations and SNR
        Supports backpropagation for basegraph training
        """
        degree = self.pcm_soft.shape[1]
        input_distr = self.params.bi_awgn_input(snr_db).repeat(degree, 1)
        if self.punctured:
            input_distr[: self.punctured] = self.params.delta_input().view(1, -1)

        # Calculate FFT outside iteration loop to speed-up Q-messages update
        # pylint: disable=e1102
        input_fft = torch.fft.rfft(input_distr, n=2 * input_distr.shape[-1], dim=-1)

        self.q_msg = input_distr.unsqueeze(0).repeat(self.pcm_soft.shape[0], 1, 1)
        self.r_msg[:] = 0

        for _ in range(n_iter):
            self.update_r_msg()
            self.update_q_msg(input_fft)
        return output_ber(self.r_msg, input_distr)

    def update_r_msg(self):
        """
        R-message update
        """
        w = torch.unsqueeze(self.pcm_soft, -1)
        infty_input = self.params.infty_input().view(1, 1, -1)
        delta_input = self.params.delta_input().view(1, 1, -1)
        # Must be implemented in subclass
        out = self.cnop_function(w * self.q_msg + (1 - w) * infty_input)
        self.r_msg = w * out + (1.0 - w) * delta_input

    def cnop_function(self, _):
        """
        Check node operation (CNOP) must be implemented in subclass for MS/SP
        """
        raise NotImplementedError("Must be implemented in subclass")

    def update_q_msg(self, input_fft):
        """
        Variable node processing
        """
        self.q_msg = vnop(self.r_msg, input_fft)

    def loss(self, n_iter, snr_db):
        """
        Loss function is log(BER) for numerical stability
        """
        return torch.log(self.run(n_iter=n_iter, snr_db=snr_db))


class DensityEvolutionNms(DensityEvolutionBase):
    """
    Flooding-based normalized min-sum density evolution with trainable PCM.
    """

    def __init__(self, pcm, n_punctured, grid_params, col_scales):
        super().__init__(pcm, n_punctured, grid_params)
        self.col_scales = torch.tensor(
            col_scales,
            device=self.params.device,
            dtype=self.params.dtype,
            requires_grad=True,
        )
        self.scaling_tensor = self.params.trainable_col_scales(self.col_scales)

    def cnop_function(self, q_msg_w):
        """
        Min-sum function with per-column normalization
        """
        return torch.einsum('bci, cji -> bcj', cnop_ms(q_msg_w), self.scaling_tensor)


class DensityEvolutionSpa(DensityEvolutionBase):
    """
    Flooding-based sum_product density evolution with trainable PCM.
    """

    def __init__(self, pcm, n_punctured, grid_params):
        super().__init__(pcm, n_punctured, grid_params)
        self.spa_idx, self.spa_w = self.params.get_spa_tensor()

    @torch.compile
    def cnop_function(self, q_msg_w):
        """
        Sum-product check node update
        """
        return cnop_sp(q_msg_w, self.spa_idx, self.spa_w)
