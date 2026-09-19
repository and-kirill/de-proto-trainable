# Copyright (c) 2026 Kirill Andreev <and.kirill@gmail.com>
# SPDX-License-Identifier: MIT

"""
Main script for protograph training experiments
"""

import sys
import json
from config import Config, OptimizerGdConfig

from optimization.final_choice import choice
from optimization.optimizer import GdOptimizer

def load_config(last_epoch, config_path):
    """
    Load configuration from JSON file
    Config creation tries to check the input data
    """
    try:
        with open(config_path, "r", encoding="utf-8") as fhandle:
            data = json.load(fhandle)
        return Config(last_epoch, **data)
    except FileNotFoundError as e:
        print("File not found", e)
    except AttributeError as e:
        print("Attribute error: ", e)
    except ValueError as e:
        print("Value Error", e)
    except TypeError as e:
        print("Type Error", e)
    sys.exit(-1)

def instantiate_optimizer(config, last_epoch):
    """
    Instantiate a proper optimizer based on config
    """
    if isinstance(config.optimization, OptimizerGdConfig):
        return GdOptimizer(config, last_epoch)
    raise ValueError("Unknown optimization method")

def main():
    """
    Optimization main-loop.
    """
    if len(sys.argv) < 2:
        print(f"Usage: {__file__} <config_path>.json <last_epoch>")
        sys.exit(0)
    last_epoch = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    config = load_config(last_epoch, sys.argv[1])
    optimizer = instantiate_optimizer(config, last_epoch)
    final_epoch, snr_db = optimizer.run()
    if final_epoch is None:
        raise RuntimeError("Training finished without an accepted epoch")
    choice(config, final_epoch, snr_db)
    print(config)


if __name__ == "__main__":
    main()
