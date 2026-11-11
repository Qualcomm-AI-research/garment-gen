# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Wraps train/validation/test dataset splits behind a single interface."""

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import hydra
from torch import Tensor
from transformers import PreTrainedTokenizer

from swifttailor.data.datasets.gcd_mm_dataset import GCDMM
from swifttailor.data.patterns.pattern_converter import NNSewingPattern

log = logging.getLogger(__name__)


@dataclass
class DataWrapperConfig:
    """Paths for the split definition file and output directory."""

    split_file: str
    output_dir: str


class DataWrapper:
    """Holds the dataset splits and delegates token/pattern operations to them."""

    def __init__(
        self,
        dataset,
        split_file: str,
        output_dir: str,
        collate_fn: str,
    ):
        """Store config, resolve the collate function, and load the splits."""
        self.dataset = dataset
        self.split_file = split_file
        self.output_dir = output_dir
        self.collate_fn = hydra.utils.get_method(collate_fn)
        self.training: Optional[GCDMM] = None
        self.validation: Optional[GCDMM] = None
        self.test: Optional[GCDMM] = None
        self.load_split()

    def set_token_indices(self, token2idx: Dict[str, int]):
        """Propagate the token-to-index mapping to every loaded split."""
        self.training.set_token_indices(token2idx)
        self.validation.set_token_indices(token2idx)
        if self.test:
            self.test.set_token_indices(token2idx)

    def get_all_token_names(self):
        """Return all token names from the training split."""
        return self.training.get_all_token_names()

    def get_mode_names(self):
        """Return the mode names from the training split."""
        return self.training.get_mode_names()

    def decode(self, output_ids: Tensor, tokenizer: PreTrainedTokenizer):
        """Decode model output ids into patterns via the training split."""
        return self.training.decode(output_ids, tokenizer)

    def evaluate_patterns(
        self, pred_patterns: List[NNSewingPattern], gt_patterns: List[NNSewingPattern]
    ):
        """Score predicted patterns against ground-truth patterns."""
        return self.training.evaluate_patterns(pred_patterns, gt_patterns)

    @property
    def panel_edge_type_indices(self):
        """Panel edge-type indices from the training split."""
        return self.training.panel_edge_type_indices

    @property
    def gt_stats(self):
        """Ground-truth statistics from the training split."""
        return self.training.gt_stats

    def load_split(self):
        """Instantiate train/validation/test datasets from the split file.

        Raises:
            ValueError: If no valid split file is available.
        """
        if self.split_file is not None and os.path.exists(self.split_file):
            log.info("Loading data split from {}".format(self.split_file))
            with open(self.split_file, "r") as f_json:
                split_dict = json.load(f_json)
        else:
            log.error(
                "No valid split file provided. Please provide a split file to load the data split."
            )
            raise ValueError(
                "No valid split file provided. Please provide a split file to load the data split."
            )
        self.training = hydra.utils.instantiate(
            self.dataset,
            load_by_dataname=split_dict["train"],
        )

        self.validation = hydra.utils.instantiate(
            self.dataset,
            load_by_dataname=split_dict["full_test"],
            is_training=False,
        )
        if "test" in split_dict:
            self.test = hydra.utils.instantiate(
                self.dataset,
                load_by_dataname=split_dict["full_test"],
                is_training=False,
            )
        else:
            self.test = None

        log.info(
            "Dataset split: {} / {} / {}".format(
                len(self.training) if self.training else None,
                len(self.validation) if self.validation else None,
                len(self.test) if self.test else None,
            )
        )

        return self.training, self.validation, self.test
