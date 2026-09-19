# Copyright (c) 2026 Kirill Andreev <and.kirill@gmail.com>
# SPDX-License-Identifier: MIT

"""
Helper functions for optimization
"""

import sys
import os
import logging
import numpy as np

from implementation.density_evolution import DensityEvolutionNms, DensityEvolutionSpa
from .lr_scheduler import LearningRateScheduler
from .convergence import ConvergenceCriterion
from .parameter_updater import ParameterUpdater


def enable_log(filename):
    """
    Enable logging with proper formats
    """
    logger = logging.getLogger("optimization")
    logger.setLevel(logging.DEBUG)
    print("LOG enable to", filename)
    handlers = [
        logging.FileHandler(filename), # Log to file
        logging.StreamHandler(stream=sys.stdout) # Log to stdout
    ]

    fmt = logging.Formatter('%(asctime)s %(name)s-%(levelname)s: %(message)s')

    for handler in handlers:
        handler.setFormatter(fmt)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

    return logger


class GdOptimizer:
    """
    Gradient descend based optimizaer
    """
    def __init__(self, config, last_epoch):
        self.config = config
        os.makedirs(self.config.data_dir, exist_ok=True)
        self.logger = enable_log(f"{self.config.data_dir}.log")
        self.epoch = last_epoch
        self.last_accepted_epoch = last_epoch if last_epoch > 0 else None
        self.lr_schedule = LearningRateScheduler(
            self.config.optimization.lr_schedule,
            self.logger
        )
        self.convergence_criterion = ConvergenceCriterion(
            self.config.optimization.convergence,
            self.logger
        )
        col_scales = None
        if self.config.decoder.type == "nms":
            col_scales = self.config.decoder.col_scales_array
        self.parameter_updater = ParameterUpdater(
            pcm=self.config.protograph.pcm_np,
            data_dir=self.config.data_dir,
            col_scales=col_scales,
            train_scales=bool(self.config.decoder.train_scales)
        )

    def run(self):
        """
        Optimize the parity check matrix and return the final epoch and SNR.

        Trivial gradient descend with fixed learning rate is considered
        """
        self.logger.info(self.config)
        while not self.convergence_criterion.should_stop(self.epoch):
            self.run_epoch()
        return self.last_accepted_epoch, self.config.optimization.snr_db

    def instantiate_de_impl(self):
        """
        Get density evolution instance based on config
        """
        parameters = self.parameter_updater.next_values(
            self.lr_schedule.current_rate
        )
        if self.config.decoder.type == "nms":
            return DensityEvolutionNms(
                parameters["pcm"],
                self.config.protograph.punctured,
                self.config.grid,
                parameters["col_scales"]
            )
        return DensityEvolutionSpa(
            parameters["pcm"],
            self.config.protograph.punctured,
            self.config.grid
            )

    def check_snr_decrease(self, ber):
        """
        Check and apply SNR decrease
        """
        if ber > self.config.optimization.ber_threshold:
            return False
        self.config.optimization.snr_db -= self.config.optimization.snr_step
        self.logger.info("SNR decreased to %1.3f dB.", self.config.optimization.snr_db)
        self.lr_schedule.reset()
        self.parameter_updater.reset_gradients()
        return True

    def run_epoch(self):
        """
        Run single epoch
        """
        opt_params = self.config.optimization
        de = self.instantiate_de_impl()
        self.logger.info("Running epoch %d.", self.epoch)
        loss = de.loss(n_iter=opt_params.n_iter, snr_db=opt_params.snr_db)
        loss_val = loss.detach().cpu().item()
        ber = np.exp(loss_val)

        self.logger.info(
            "Loss = %1.8e, BER = %1.8e, SNR = %1.3f dB.",
            loss_val, ber, opt_params.snr_db
        )

        if self.check_snr_decrease(ber):
            return
        # Calculate gradients
        loss.backward()

        n_grad_flips = self.parameter_updater.gradient_sign_flips(de)
        if n_grad_flips is not None:
            self.logger.info(
                "Gradient sign difference: %d positions",
                n_grad_flips
            )
            if not self.lr_schedule.update(n_grad_flips):
                return

        self.parameter_updater.commit(de, self.epoch, ber)
        self.last_accepted_epoch = self.epoch
        self.logger.info(
            "Update saved, LR = %1.4e",
            self.lr_schedule.current_rate
        )
        self.epoch += 1
        self.convergence_criterion.update(
            self.epoch,
            self.parameter_updater.pcm,
            self.parameter_updater.pcm_gradient
        )
