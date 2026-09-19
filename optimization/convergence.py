# Copyright (c) 2026 Kirill Andreev <and.kirill@gmail.com>
# SPDX-License-Identifier: MIT

"""Convergence criteria for protograph optimization."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ConvergenceConfig:
    """Stopping conditions for gradient-descent optimization."""

    max_epochs: int
    ensemble_size_patience: int

    def __post_init__(self):
        if self.max_epochs <= 0:
            raise ValueError("Maximum number of epochs must be positive")
        if self.ensemble_size_patience <= 0:
            raise ValueError("Ensemble-size patience must be positive")

    def __str__(self):
        msg = f"  Maximum number of epochs:        {self.max_epochs}\n"
        msg += (
            "  Ensemble-size patience:         "
            f"{self.ensemble_size_patience} epochs\n"
        )
        return msg


class ConvergenceCriterion:
    """Combine all stopping conditions and track their state."""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self._min_ensemble_size = None
        self._epochs_without_improvement = 0
        self._converged = False

    def should_stop(self, completed_epochs):
        """Check conditions that do not require a newly accepted step."""
        if self._converged:
            return True
        if completed_epochs >= self.config.max_epochs:
            self._stop(
                "maximum number of epochs reached (%d)",
                self.config.max_epochs
            )
        return self._converged

    def update(self, completed_epochs, pcm, gradient):
        """Update convergence state after an epoch has been accepted."""
        ensemble_size = self._effective_ensemble_log_size(pcm)
        self.logger.info("log2(# matrices) = %1.3f", ensemble_size)

        if self._is_integer_matrix_limit(pcm, gradient):
            self._stop("gradient points outside the integer PCM boundary")
            return True

        if completed_epochs >= self.config.max_epochs:
            self._stop(
                "maximum number of epochs reached (%d)",
                self.config.max_epochs
            )
            return True

        if (
            self._min_ensemble_size is None
            or ensemble_size < self._min_ensemble_size
        ):
            self._min_ensemble_size = ensemble_size
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1

        if (
            self._epochs_without_improvement
            >= self.config.ensemble_size_patience
        ):
            self._stop(
                "effective ensemble size did not reach a new minimum for %d epochs",
                self.config.ensemble_size_patience
            )
        return self._converged

    def _stop(self, message, *args):
        self.logger.info("Convergence criterion met: " + message, *args)
        self._converged = True

    @staticmethod
    def _is_integer_matrix_limit(pcm, gradient):
        if gradient is None:
            return False
        boundary_directions = (1 - 2 * pcm) * np.sign(gradient)
        return np.sum(boundary_directions) == pcm.size

    @staticmethod
    def _effective_ensemble_log_size(pcm):
        eps = 1e-15
        probabilities = np.clip(pcm, eps, 1 - eps)
        entropy = (
            -probabilities * np.log2(probabilities)
            - (1 - probabilities) * np.log2(1 - probabilities)
        )
        return np.sum(entropy)
