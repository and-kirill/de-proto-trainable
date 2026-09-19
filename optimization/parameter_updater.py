# Copyright (c) 2026 Kirill Andreev <and.kirill@gmail.com>
# SPDX-License-Identifier: MIT

"""State, gradient updates, and history for optimized DE parameters."""

from dataclasses import dataclass
import os

import numpy as np


@dataclass  # pylint: disable=too-many-instance-attributes
class ParameterPlaceholder:
    """A NumPy-backed parameter recreated in every autograd graph."""

    tensor_name: str
    values: np.ndarray
    trainable: bool
    minimum: float
    maximum: float | None
    history_name: str
    gradient_history_name: str | None = None
    gradient: np.ndarray | None = None

    def next_values(self, learning_rate):
        """Apply one detached gradient step and clamp it to valid bounds."""
        if not self.trainable or self.gradient is None:
            return self.values
        updated = self.values - learning_rate * self.gradient
        if self.maximum is None:
            return np.maximum(updated, self.minimum)
        return np.clip(updated, self.minimum, self.maximum)

    def extract(self, de_instance):
        """Detach this parameter and its gradient from a DE instance."""
        tensor = getattr(de_instance, self.tensor_name)
        values = tensor.detach().cpu().numpy()
        gradient = None
        if self.trainable:
            if tensor.grad is None:
                raise ValueError(f"Gradient is missing for {self.tensor_name}")
            gradient = tensor.grad.detach().cpu().numpy()
        return values, gradient

    def save(self, data_dir, epoch):
        """Save the accepted parameter state."""
        np.savetxt(
            os.path.join(data_dir, f"{self.history_name}_{epoch}.txt"),
            self.values,
            fmt="%1.6e"
        )
        if self.gradient_history_name is not None and self.gradient is not None:
            np.savetxt(
                os.path.join(
                    data_dir,
                    f"{self.gradient_history_name}_{epoch}.txt"
                ),
                self.gradient,
                fmt="%1.6e"
            )


class ParameterUpdater:
    """Own all parameters updated outside the PyTorch computation graph."""

    def __init__(
        self,
        pcm,
        data_dir,
        col_scales=None,
        train_scales=False
    ):
        self.data_dir = data_dir
        self._parameters = {
            "pcm": ParameterPlaceholder(
                tensor_name="pcm_soft",
                values=np.array(pcm, copy=True),
                trainable=True,
                minimum=0.0,
                maximum=1.0,
                history_name="pcm_soft",
                gradient_history_name="pcm_grad"
            )
        }
        if col_scales is not None:
            self._parameters["col_scales"] = ParameterPlaceholder(
                tensor_name="col_scales",
                values=np.array(col_scales, copy=True),
                trainable=train_scales,
                minimum=0.0,
                maximum=None,
                history_name="scales"
            )

    @property
    def pcm(self):
        """Return the currently accepted relaxed parity-check matrix."""
        return self._parameters["pcm"].values

    @property
    def pcm_gradient(self):
        """Return the gradient associated with the accepted PCM."""
        return self._parameters["pcm"].gradient

    def next_values(self, learning_rate):
        """Return detached parameter values for a fresh DE instance."""
        return {
            name: parameter.next_values(learning_rate)
            for name, parameter in self._parameters.items()
        }

    def gradient_sign_flips(self, de_instance):
        """Count PCM gradient sign changes against the accepted state."""
        previous_gradient = self.pcm_gradient
        if previous_gradient is None:
            return None
        _, candidate_gradient = self._parameters["pcm"].extract(de_instance)
        return np.count_nonzero(
            np.sign(candidate_gradient) != np.sign(previous_gradient)
        )

    def commit(self, de_instance, epoch, ber):
        """Accept a DE result and persist the complete epoch history."""
        snapshots = {
            name: parameter.extract(de_instance)
            for name, parameter in self._parameters.items()
        }
        for name, parameter in self._parameters.items():
            parameter.values, parameter.gradient = snapshots[name]
            parameter.save(self.data_dir, epoch)
        np.savetxt(
            os.path.join(self.data_dir, f"ber_{epoch}.txt"),
            np.array([ber]),
            fmt="%1.6e"
        )

    def reset_gradients(self):
        """Prevent an update based on gradients from an obsolete SNR."""
        for parameter in self._parameters.values():
            parameter.gradient = None
