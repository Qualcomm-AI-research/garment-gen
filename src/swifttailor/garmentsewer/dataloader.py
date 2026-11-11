# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Factory helpers for building GarmentSewer datasets and dataloaders."""

import importlib
import logging
import time

from accelerate.logging import get_logger
from termcolor import colored
from torch.utils.data import DataLoader

logger = get_logger(__name__, log_level="INFO")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def import_class(class_path: str):
    """Import and return the class named by a dotted ``module.Class`` path."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def build_dataset(dataset_class, **kwargs):
    """Instantiate the dataset class from its dotted path and log its size."""
    logger.info(f"Constructing dataset {dataset_class}...")
    t = time.time()
    ds_class = import_class(dataset_class)
    dataset = ds_class(**kwargs)
    logger.info(
        f"{colored(f'Dataset {dataset_class} constructed: ', 'green', attrs=['bold'])}"
        f"time: {(time.time() - t):.2f} s, length (use/ori): {len(dataset)}/{dataset.ori_imgs_nums}"
    )
    return dataset


def build_dataloader(dataset, batch_size=256, num_workers=4, shuffle=True, **kwargs):
    """Wrap a dataset in a ``DataLoader``, honoring an optional batch sampler."""
    if "pin_memory" not in kwargs:
        kwargs["pin_memory"] = True

    logger.info(
        f"Batch size: {batch_size}. Num workers: {num_workers}. Shuffle: {shuffle}. Kwargs: {kwargs}"
    )
    if "batch_sampler" in kwargs:
        dataloader = DataLoader(
            dataset, batch_sampler=kwargs["batch_sampler"], num_workers=num_workers, **kwargs
        )
    else:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs
        )
    return dataloader
