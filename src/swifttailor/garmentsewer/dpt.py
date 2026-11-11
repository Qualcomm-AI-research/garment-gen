# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""DPT-based geometry-image model with optional stitching/normal/boundary losses."""

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger

# pytorch3d is only needed for stitching loss (training).
# Import lazily so inference works without pytorch3d installed.
try:
    from pytorch3d.loss import chamfer_distance
    from pytorch3d.structures import Pointclouds
except ImportError:
    chamfer_distance = None
    Pointclouds = None
from torch import nn

from .dpt_utils.model import DPT

logger = get_logger(__name__, log_level="INFO")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class DPTGIM(DPT):
    """DPT variant predicting geometry images, with configurable auxiliary losses."""

    def __init__(self, *args, pretrained_ckpt: Optional[str] = None, **kwargs):
        """Configure loss flags/weights and optionally load a pretrained checkpoint."""
        super().__init__(*args, **kwargs)
        self._use_stitching_loss = kwargs.get("use_stitching_loss", False)
        self._stitching_loss_weight = kwargs.get("stitching_loss_weight", 1)
        logger.info(
            f"[MODEL] Using stitching loss: {self._use_stitching_loss} - Weight: {self._stitching_loss_weight}"
        )

        self._use_normal_loss = kwargs.get("use_normal_loss", False)
        self._normal_loss_weight = kwargs.get("normal_loss_weight", 1)
        self._has_color = kwargs.get("has_color", True)
        logger.info(
            f"[MODEL] Using normal loss: {self._use_normal_loss} - Weight: {self._normal_loss_weight}"
        )

        self._use_boundary_loss = kwargs.get("use_boundary_loss", False)
        self._boundary_width = kwargs.get("boundary_width", 5)
        self._boundary_loss_weight = kwargs.get("boundary_loss_weight", 1)
        logger.info(
            f"[MODEL] Using boundary loss: {self._use_boundary_loss} - Weight: {self._boundary_loss_weight}"
        )

        if pretrained_ckpt is not None:
            logger.info(f"[MODEL] Loading pretrained checkpoint: {pretrained_ckpt}")
            self.load_state_dict(torch.load(pretrained_ckpt, map_location="cpu"), strict=True)

    def forward(self, **kwargs):
        """Dispatch to the method named by ``forward_type``, defaulting to ``model_forward``."""
        fn = self.model_forward
        if "forward_type" in kwargs:
            fn_name = kwargs["forward_type"]
            assert hasattr(self, fn_name), f"{self.__class__.__name__} has no function {fn_name}."
            fn = getattr(self, fn_name)
            del kwargs["forward_type"]

        return fn(**kwargs)

    def model_forward(self, **kwargs):
        """Run the underlying DPT forward pass."""
        return super().forward(**kwargs)

    def train_forward(self, **kwargs):
        """Training forward pass (not implemented)."""
        raise NotImplementedError
