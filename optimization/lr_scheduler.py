# Copyright (c) 2026 Kirill Andreev <and.kirill@gmail.com>
# SPDX-License-Identifier: MIT

"""Learning-rate configuration and scheduling policy."""

from dataclasses import dataclass


@dataclass
class LearningRateScheduleConfig:
    """Learning-rate schedule controlled by gradient sign flips."""

    initial_rate: float
    acceleration_factor: float = 1.0
    deceleration_factor: float = 1.0
    acceleration_flip_threshold: int = 0
    deceleration_flip_threshold: int = 0
    rollback_flip_threshold: int = 0

    def __post_init__(self):
        if self.initial_rate <= 0:
            raise ValueError("Initial learning rate must be positive")
        if self.acceleration_factor <= 0:
            raise ValueError("Acceleration factor must be positive")
        if self.deceleration_factor <= 0:
            raise ValueError("Deceleration factor must be positive")
        if self.acceleration_flip_threshold > self.deceleration_flip_threshold:
            raise ValueError(
                "LR acceleration threshold must not exceed the deceleration threshold"
            )
        if self.deceleration_flip_threshold >= self.rollback_flip_threshold:
            raise ValueError(
                "LR deceleration threshold must be smaller than the rollback threshold"
            )

    def __str__(self):
        msg = f"  Initial learning rate (LR):     {self.initial_rate:1.3e}"
        if self.acceleration_factor <= 1.0:
            return msg + " (fixed)\n"
        msg += "\n"
        msg += "  LR is increased by factor:      "
        msg += (
            f"{self.acceleration_factor:1.3e} at "
            f"<{self.acceleration_flip_threshold} GD sign flips\n"
        )
        msg += "  LR is decreased by factor:      "
        msg += (
            f"{self.deceleration_factor:1.3e} at "
            f">{self.deceleration_flip_threshold} GD sign flips\n"
        )
        msg += (
            "  Rollback to the previous epoch: "
            f"at >{self.rollback_flip_threshold} GD sign flips\n"
        )
        return msg


class LearningRateScheduler:
    """
    Own the learning rate and all rules that change it.

    ``update`` returns whether the optimizer should accept the current step.
    """

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self._rate = config.initial_rate

    @property
    def current_rate(self):
        """Return the currently active learning rate."""
        return self._rate

    def reset(self):
        """Reset the learning rate to its initial value."""
        self._rate = self.config.initial_rate

    def update(self, n_grad_flips):
        """Update LR after a step and report whether that step is accepted."""
        if (
            n_grad_flips > self.config.rollback_flip_threshold
            and self._rate > self.config.initial_rate
        ):
            self.reset()
            self.logger.info("Rollback: LR reset to %1.4e", self._rate)
            return False

        if self.config.acceleration_factor <= 1.0:
            return True

        if n_grad_flips > self.config.deceleration_flip_threshold:
            self._rate = max(
                self._rate * self.config.deceleration_factor,
                self.config.initial_rate
            )
            self.logger.info("Learning rate decelerated to %1.3e", self._rate)
        elif n_grad_flips < self.config.acceleration_flip_threshold:
            self._rate *= self.config.acceleration_factor
            self.logger.info("Learning rate accelerated to %1.3e", self._rate)
        return True
