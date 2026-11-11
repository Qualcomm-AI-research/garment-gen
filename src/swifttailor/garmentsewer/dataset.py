# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""GarmentCode dataset yielding geometry images, condition maps, and stitching pairs."""

import json
import logging
from itertools import product
from typing import Dict, List, Literal, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from accelerate.logging import get_logger
from torch.utils.data import default_collate

from swifttailor.config import GLOBAL_CONFIG

logger = get_logger(__name__, log_level="INFO")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def log(*args, **kwargs):
    """Emit an info-level log line."""
    logger.info(*args, **kwargs)


class GarmentCodeDataset(torch.utils.data.Dataset):
    """Dataset of GarmentCode samples exposing GIM, condition, and stitching data."""

    SUPPORTED_VERSIONS = ["v3_512"]
    SUPPORTED_CONDITION_VERSIONS = ["v3_512", "v3_512_final"]
    MAX_BUCKET = 160

    def __init__(
        self,
        split: Literal["train", "validation", "test"] = "train",
        resolution=512,
        subset=0.01,
        n_samples: Optional[int] = None,
        version: str = "",
        condition_version: str = "",
        max_bucket: int = MAX_BUCKET,
        dataset_length: Optional[int] = None,
        n_actual_samples: Optional[int] = None,
        return_stitching_map_instead: bool = False,
    ):
        """Validate versions and load the official split's sample paths and text."""
        assert version in self.SUPPORTED_VERSIONS, (
            f"Version {version} is not supported. "
            f"Supported versions are: {self.SUPPORTED_VERSIONS}"
        )
        assert condition_version in self.SUPPORTED_CONDITION_VERSIONS, (
            f"Condition version {condition_version} is not supported. "
            f"Supported versions are: {self.SUPPORTED_CONDITION_VERSIONS}"
        )
        log(f"Resolution: {resolution}. Subset: {subset}")

        self.split = split
        self.resolution = resolution
        self._subset = subset
        self._n_samples = n_samples
        self._version = "" if version == "" else f"_{version}"
        self._condition_version = "" if condition_version == "" else f"_{condition_version}"
        self.return_stitching_map_instead = return_stitching_map_instead
        self._text = self._extract_text()
        self._max_bucket = max_bucket
        self.dataset = self._extract_official_split()
        if n_actual_samples is not None:
            self.dataset = self.dataset[:n_actual_samples]
        self._dataset_length = dataset_length
        self.ori_imgs_nums = len(self.dataset)

    def update_samples(self, samples):
        """Replace the sample list and refresh cached lengths."""
        self.dataset = samples
        self._dataset_length = len(self.dataset)
        self.ori_imgs_nums = len(self.dataset)

    def get_max_number_panels(self):
        """Return (and cache) the panel-name count from ``panel_names.txt``."""
        if hasattr(self, "max_n_panels"):
            return self.max_n_panels
        with open(
            GLOBAL_CONFIG.config_dir / "panel_names.txt",
            "r",
        ) as f:
            panel_names = f.readlines()
        self.max_n_panels = len(panel_names)
        return self.max_n_panels

    def _extract_text(self) -> dict:
        """Load the ``idx -> descriptive_text`` map, or ``{}`` if the parquet is missing."""
        parquet_path = GLOBAL_CONFIG.gcd_data_dir / "gcd-mm.parquet"
        if not parquet_path.exists():
            log(
                f"[WARNING] {parquet_path} not found — text metadata unavailable. "
                "Samples will be loaded via update_samples() without text filtering."
            )
            return {}
        df = pd.read_parquet(parquet_path)
        mmap = dict(zip(df["idx"], df["descriptive_text"]))
        return mmap

    def _extract_official_split(self) -> np.ndarray:
        """Resolve the split's sample paths, then subset and filter them."""
        split_path = GLOBAL_CONFIG.gcd_data_dir / "gcd_mm_split.json"
        if not split_path.exists():
            log(
                f"[WARNING] {split_path} not found — official split unavailable. "
                "Use update_samples() to supply sample paths directly."
            )
            return np.array([])
        with open(split_path, "r") as f:
            split: Dict[str, List[str]] = json.load(f)

        samples = split[self.split]

        # Add absolute path to samples
        samples = [GLOBAL_CONFIG.gcd_data_dir / "GarmentCodeData_v2" / sample for sample in samples]

        # Add prefix path "data" to sample_id due to extraction
        # ../sample_id -> ../data/sample_id
        samples = [sample.parent / "data" / sample.name for sample in samples]
        np_samples = np.array(samples)
        log(f"Total orignal samples: {len(np_samples)}")
        if self._subset < 1:
            np_samples = self._extract_subset(np_samples)
        np_samples = self._filter_existing_samples(np_samples)
        if self._n_samples is not None:
            np_samples = np_samples[: self._n_samples]
        log(f"Total valid samples after (sampling and) filtering: {len(np_samples)}")

        return np_samples

    def _extract_subset(self, data):
        """Return the leading ``subset`` fraction of the samples."""
        n = data.shape[0]
        indices = np.array(list(range(int(self._subset * n))))
        data = data[indices]

        log(f"Subset with ratio {self._subset} of" f" original data: {data.shape[0]}")

        return data

    def _filter_existing_samples(self, samples: np.ndarray) -> np.ndarray:
        """Keep only default-body samples with GIM, condition, and text present."""
        assert hasattr(self, "_text"), "Text has not been initialized."

        is_valid_path = [
            "default_body" in str(sample)
            and (sample / f"{sample.stem}_gim{self._version}.npz").exists()
            and (sample / f"{sample.stem}_condition{self._condition_version}.npz").exists()
            and sample.stem in self._text
            for sample in samples
        ]

        valid_indices = np.where(is_valid_path)[0]

        return samples[valid_indices]

    def _get_label_idx_mapping(self):
        """Map each (upper, waistband, bottom) component combination to a label index."""
        design_path = GLOBAL_CONFIG.root / "assets" / "design_params" / "default.yaml"
        design_params = yaml.safe_load(open(design_path, "r"))

        component_names = ["upper", "wb", "bottom"]
        components = [design_params["design"]["meta"][name]["range"] for name in component_names]
        idx = {comb: i for i, comb in enumerate(product(*components))}

        assert idx[(None, None, None)] == len(idx) - 1, (
            "The last index should be reserved for " "the default combination (None, None, None)"
        )
        return idx

    def prepare_gim(self, gim):
        """Split a 4-channel GIM into coordinates, mask, and stitching info.

        Returns:
            The coordinate tensor, occupancy mask, and either grouped stitching
            pairs or the raw stitching map (for inference).
        """
        # H, W, C == 512, 512, 4
        assert gim.shape[0] == gim.shape[1], "Input gim must be square"
        assert gim.shape[2] == 4, "Input gim must have 4 channels (x, y, z, stitching map)"

        stitching_map = gim[:, :, 3:4]
        gim = gim[:, :, :3]

        # Calculate mask
        mask = np.any(gim != 0, axis=-1)[..., np.newaxis]

        # Group stitching
        if not self.return_stitching_map_instead:
            stitching_pairs = self.group_stitching(gim, mask, stitching_map)
        gim = torch.from_numpy(gim.transpose(2, 0, 1)).float()

        return (
            gim,
            mask,
            (
                stitching_pairs  # type: ignore - use for training
                if not self.return_stitching_map_instead
                else stitching_map  # use for inference
            ),
        )

    def extract_stitching_boundaries(self, stitching_map, occupancy, buckets):
        """Trace panel contours and bucketize their per-vertex stitching values.

        Returns:
            The boundary UV coordinates, their stitch bucket ids, and per-vertex
            panel ids.
        """
        assert stitching_map.shape == occupancy.shape, (
            "Stitching map and occupancy must have the same shape.\n"
            f"Got stitching_map.shape: {stitching_map.shape} and "
            f"occupancy.shape: {occupancy.shape}"
        )
        contours, _ = cv2.findContours(
            occupancy.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        list_boundaries = []
        gim_panel_ids = []
        for panel_id, raw_contour in enumerate(contours):
            if len(raw_contour) < 2:
                continue
            contour = raw_contour.squeeze()

            # swap x and y due to cv2
            contour[:, [0, 1]] = contour[:, [1, 0]]
            contour_stitching = stitching_map[contour[:, 0], contour[:, 1]]
            for i in range(1, len(contour_stitching)):
                if contour_stitching[i] != contour_stitching[i - 1]:
                    break
            list_boundaries.append(np.concatenate([contour[i:], contour[:i]], axis=0))
            gim_panel_ids.extend([panel_id] * len(raw_contour))
        boundaries = np.concatenate(list_boundaries, axis=0)

        # Tuple of (array x, array y), each array is (n,)
        uv_coords = (boundaries[:, 0], boundaries[:, 1])
        # (n,)
        stitching_values = stitching_map[uv_coords]

        mask = (buckets[:, 0] <= stitching_values) & (stitching_values < buckets[:, 1])
        # (n,)
        stitching_ids = np.where(mask)[1]

        return uv_coords, stitching_ids, np.array(gim_panel_ids)

    def create_buckets(self):
        """Return and cache integer stitching-value buckets."""
        if hasattr(self, "_buckets"):
            return self._buckets
        n_buckets = self._max_bucket
        bucket_st = [i for i in range(n_buckets)]
        bucket_en = [i for i in range(1, n_buckets + 1)]
        buckets = list(zip(bucket_st, bucket_en))
        self._buckets = np.array(buckets)
        return self._buckets

    def group_stitching(self, gim, occupancy, stitching_map):
        """Pair up boundary UV points that share a stitch bucket across two panels.

        Returns:
            Per-stitch ``(panel1_uv, panel2_uv)`` coordinate tensor pairs.
        """
        buckets = self.create_buckets()
        uv_coords, stitching_ids, gim_panel_ids = self.extract_stitching_boundaries(
            stitching_map, occupancy, buckets
        )
        stitching_pairs = []

        # Skip bucket 0 (background) and 1 (boundary but not stitched)
        for bucket_id in range(2, self._max_bucket):
            pos = np.where(stitching_ids == bucket_id)
            if len(pos[0]) == 0:
                continue
            x_, y_ = uv_coords[0][pos[0]], uv_coords[1][pos[0]]
            component_ids = gim_panel_ids[pos[0]].astype(np.int32)
            if len(np.unique(component_ids)) == 1:
                half_len = len(component_ids) // 2
                component_ids[half_len:] = component_ids[half_len:] + 1

            # Sort component_ids with bincount
            panel1, panel2 = np.argsort(np.bincount(component_ids))[-2:]
            point_ids_on_panel1 = np.where(component_ids == panel1)[0]
            point_ids_on_panel2 = np.where(component_ids == panel2)[0]

            # Extract uv coordinates of boundary points on panel1 and panel2
            panel1_uv_coords = np.stack(
                [x_[point_ids_on_panel1], y_[point_ids_on_panel1]], axis=-1
            )  # (n, 2)
            panel2_uv_coords = np.stack(
                [x_[point_ids_on_panel2], y_[point_ids_on_panel2]], axis=-1
            )  # (n, 2)

            stitching_pairs.append(
                (torch.from_numpy(panel1_uv_coords), torch.from_numpy(panel2_uv_coords))
            )

        # List[Tuple[Tensor[n, 2], Tensor[m, 2]]] - n and m on each stitch are not fixed
        return stitching_pairs

    def getdata(self, idx):
        """Load and prepare the GIM, condition map, and stitching for one sample.

        Returns:
            The coordinate GIM, masked/normalized condition, and stitching data.
        """
        sample = self.dataset[idx]
        sample_id = sample.stem

        gim_path = sample / f"{sample_id}_gim{self._version}.npz"
        condition_path = sample / f"{sample_id}_condition{self._condition_version}.npz"
        gim = np.load(gim_path)["arr_0"]
        condition = np.load(condition_path)["arr_0"]

        gim, mask, stitching = self.prepare_gim(gim)

        condition = condition * mask
        condition[:, :, -1] = condition[:, :, -1] / self.get_max_number_panels()
        condition = condition.transpose(2, 0, 1)
        condition = torch.from_numpy(condition)

        return (
            gim,
            condition,
            stitching,
        )

    def __getitem__(self, idx):
        """Return one sample, retrying subsequent indices on load errors."""
        for _ in range(10):
            idx = idx % self.actual_len()
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                import traceback

                print(f"Found error at: {self.dataset[idx]}")
                print(f"Error details, {str(e)}")
                print(f"Full traceback:")
                print(traceback.format_exc())
                idx = idx + 1
        raise RuntimeError("Too many bad data.")

    def actual_len(self):
        """Return (and cache) the true number of samples."""
        if hasattr(self, "_actual_length"):
            return self._actual_length
        self._actual_length = len(self.dataset)
        return self._actual_length

    def __len__(self):
        """Return the dataset length, honoring an override if set."""
        return self._dataset_length if self._dataset_length is not None else len(self.dataset)

    @staticmethod
    def collate_fn(batch):
        """Collate a batch, keeping variable-length stitching pairs as a list."""
        new_batch = [item[:-1] for item in batch]
        stitching_pairs = [item[-1] for item in batch]
        return *default_collate(new_batch), stitching_pairs
