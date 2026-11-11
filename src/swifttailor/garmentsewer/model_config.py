# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Dataclass configs for GarmentSewer data, model, and training."""

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BaseConfig:
    """Dataclass config base with dict-style access and JSON display."""

    def get(self, attribute_name, default=None):
        """Return the named attribute, or ``default`` if absent."""
        return getattr(self, attribute_name, default)

    def pop(self, attribute_name, default=None):
        """Remove and return the named attribute, or ``default`` if absent."""
        if hasattr(self, attribute_name):
            value = getattr(self, attribute_name)
            delattr(self, attribute_name)
            return value
        else:
            return default

    def __str__(self):
        """Return the config as pretty-printed JSON."""
        return json.dumps(asdict(self), indent=4)


@dataclass
class DataConfig(BaseConfig):
    """Dataset type and its constructor kwargs."""

    type: Optional[str] = None
    kwargs: Dict = field(default_factory=lambda: dict())


@dataclass
class ModelConfig(BaseConfig):
    """Model type and its constructor kwargs."""

    type: Optional[str] = None
    kwargs: Dict = field(default_factory=lambda: dict())


@dataclass
class TrainerConfig(BaseConfig):
    """Full training configuration: data, model, optimization, and evaluation."""

    # Data config
    data: DataConfig
    # Model config
    model: ModelConfig

    # For debug
    debug: bool = False

    # Training Config
    text_encoder_path: Optional[str] = None
    pretrained_model_path: Optional[str] = None

    output_dir: str = "autoencoderdc-model"
    cache_dir: Optional[str] = None

    train_batch_size: int = 4
    num_workers: int = 4
    seed: int = 43
    num_epochs: int = 100
    # Number of training steps, if not provided, num_epochs will be used
    num_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    grad_checkpointing: bool = False
    gradient_clip: float = 1.0

    checkpointing_steps: int = 500
    checkpoints_total_limit: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None

    lr: float = 1e-4
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    lr_num_cycles: int = 1
    lr_power: float = 1.0

    use_ema: bool = False
    use_8bit_adam: bool = False
    offload_ema: bool = False
    foreach_ema: bool = False
    non_ema_revision: Optional[str] = None
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 0.01
    adam_epsilon: float = 1e-08

    logging_dir: str = "logs"
    report_to: str = "tensorboard"
    tracker_run_name: str = "swift-tailor"

    eval_batch_size: int = 4
    eval_img_idx: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    eval_sampling_step: int = 1000
    eval_metric_step: int = 1000
    eval_subset: float = 0.025
    n_sampling_points: int = 20_000  # For eval metrics

    mixed_precision: str = "no"  # ["no", "fp16", "bf16"]
    enable_xformers_memory_efficient_attention: bool = False
    allow_tf32: bool = False
    set_grads_to_none: bool = False
