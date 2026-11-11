# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Global configuration for SwiftTailor project paths."""

import os
from pathlib import Path

import torch
from attr import dataclass


@dataclass
class _GlobalConfig:
    """Resolved project paths and the default compute device."""

    # Root of the swifttailor repository.
    # Resolved as 2 levels up from src/swifttailor/config.py, or via env var.
    root = Path(os.environ.get("SWIFTTAILOR_ROOT", Path(__file__).resolve().parents[2]))

    # Data directories
    data_dir = root / "data"
    gcd_data_dir = root / "data" / "test_set"

    # Model weights (downloaded separately via scripts/download_weights.sh)
    weights_dir = root / "weights"

    # Configs and assets
    config_dir = root / "configs"
    assets_dir = root / "assets"
    cache_dir = root / "cache"

    device = "cuda" if torch.cuda.is_available() else "cpu"


GLOBAL_CONFIG = _GlobalConfig()
