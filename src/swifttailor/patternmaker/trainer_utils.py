# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Training utilities: tensor-dict device/dtype moves and metric meters."""

import logging
from enum import Enum
from typing import List

import numpy as np
import torch
import torch.distributed as dist


def dict_to_cuda(input_dict):
    """Move all tensors in a (possibly nested) dict to CUDA in place."""
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
        elif isinstance(input_dict[k], dict):
            input_dict[k] = dict_to_cuda(input_dict[k])
    return input_dict


def dict_to_cpu(input_dict):
    """Move all tensors in a (possibly nested) dict to CPU in place."""
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cpu()
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cpu() for ele in v]
        elif isinstance(input_dict[k], dict):
            input_dict[k] = dict_to_cpu(input_dict[k])
    return input_dict


def dict_to_dtype(input_dict, dtype=torch.float32, target_keys=None):
    """Cast tensors in a (possibly nested) dict to ``dtype``, optionally by key."""
    for k, v in input_dict.items():
        if target_keys is not None:
            if k not in target_keys:
                continue

        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.to(dtype=dtype)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.to(dtype=dtype) for ele in v]
        elif isinstance(input_dict[k], dict):
            input_dict[k] = dict_to_dtype(input_dict[k], dtype)
    return input_dict


class Summary(Enum):
    """How an :class:`AverageMeter` renders its end-of-run summary."""

    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        """Store display name/format and summary mode, then reset the statistics."""
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        """Zero the current value, sum, count, and average."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Record ``val`` seen ``n`` times and update the running average."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        """Sum-reduce the running total and count across distributed processes."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        """Return the current and average value formatted for logging."""
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        """Return the end-of-run summary string per the configured summary type.

        Raises:
            ValueError: If the summary type is unrecognized.
        """
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """Formats and logs a set of :class:`AverageMeter`s during training."""

    def __init__(self, logger, local_rank, meters: List[AverageMeter], prefix=""):
        """Store the logger, process rank, meters, and line prefix."""
        self.logger = logger
        self.local_rank = local_rank
        self.meters = meters
        self.prefix = prefix

    def display(self, step):
        """Log the prefix and current meter values for the given step."""
        entries = [self.prefix.format(step)]
        entries += [str(meter) for meter in self.meters]
        master_log(self.local_rank, self.logger, "\t".join(entries))

    def display_summary(self):
        """Log each meter's end-of-run summary."""
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        master_log(self.local_rank, self.logger, " ".join(entries))


def master_log(local_rank: int, logger: logging.Logger, *args):
    """Log a message only from the rank-0 process."""
    if local_rank == 0:
        logger.info(*args)
