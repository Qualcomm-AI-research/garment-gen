# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""
PatternMaker inference script for SwiftTailor.

Generates sewing patterns from image, text, or image+text inputs using
a Vision-Language Model (InternVL3-2B + Qwen2 backbone).

Usage:
    python scripts/run_patternmaker.py --config-name benchmark_inference
"""

import logging
import os
from dataclasses import dataclass, field, fields
from typing import Literal, Optional

import hydra
import torch
import transformers
from omegaconf import OmegaConf

from swifttailor.data.data_wrappers.data_wrapper import DataWrapper, DataWrapperConfig
from swifttailor.data.datasets.constants import IMG_CONTEXT_TOKEN
from swifttailor.patternmaker.inference import extract_model_config
from swifttailor.patternmaker.st_model import SwiftTailorConfig, SwiftTailorForCausalLM
from swifttailor.patternmaker.trainer import ExperimentConfig, Trainer, TrainerConfig

log = logging.getLogger(__name__)


@dataclass
class MainConfig:
    """Hydra config for PatternMaker training/evaluation runs."""

    my_pretrained_path: str
    llm_pretrained_path: str
    sampling_type: Literal["default", "editing", "image_text", "image", "text"] = "default"
    pretrained_config: Optional[str] = None
    resume_path: Optional[str] = None
    log_folder: Optional[str] = None
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data_wrapper: DataWrapperConfig = field(default_factory=DataWrapperConfig)
    model: SwiftTailorConfig = field(default_factory=SwiftTailorConfig)
    precision: Literal["bf16", "fp16"] = "bf16"
    evaluate: bool = False
    conv_type: Literal["internvl2_5"] = "internvl2_5"
    from_start: bool = False
    # When True, each generated pattern is simulated into a draped 3D mesh via
    # GarmentCode (scripts/run_garmentcode.py), retrying until a mesh is produced.
    convert_to_3d: bool = False

    @classmethod
    def assert_all_fields_exist(cls, cfg):
        """Raise if the loaded config is missing any declared field.

        Raises:
            ValueError: If a required field is absent.
        """
        for exp_field in fields(cls):
            if not hasattr(cfg, exp_field.name):
                raise ValueError(f"Configuration is missing required field: {exp_field.name}")


@hydra.main(version_base=None, config_path="../configs/patternmaker", config_name="config")
def main(cfg: MainConfig):
    """Set up data, model, and tokenizer, then run evaluation or training.

    Raises:
        ValueError: If ``sampling_type`` is unknown.
    """
    MainConfig.assert_all_fields_exist(cfg)
    assert cfg.log_folder is not None, "log_folder must be specified in the config."

    # Auto-initialise single-GPU DDP env vars if not set by torchrun
    for key, default in [("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")]:
        if key not in os.environ:
            os.environ[key] = default

    if cfg.sampling_type == "editing":
        cfg.data_wrapper.dataset.sampling_rate = [0, 0, 0, 0, 1]
    elif cfg.sampling_type == "image_text":
        cfg.data_wrapper.dataset.sampling_rate = [0, 0, 0, 1, 0]
    elif cfg.sampling_type == "image":
        cfg.data_wrapper.dataset.sampling_rate = [1, 0, 0, 0, 0]
    elif cfg.sampling_type == "text":
        cfg.data_wrapper.dataset.sampling_rate = [0, 1, 0, 0, 0]
    elif cfg.sampling_type == "default":
        pass
    else:
        raise ValueError(f"Unknown sampling type: {cfg.sampling_type}")

    log.info(f"Working directory : {os.getcwd()}")
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Output directory : {output_dir}")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    master_process = ddp_rank == 0

    data_wrapper: DataWrapper = hydra.utils.instantiate(cfg.data_wrapper, output_dir=output_dir)
    log.info("Dataset created.")

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        experiment_cfg=cfg.experiment,
        data_wrapper=data_wrapper,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        ddp_local_rank=ddp_local_rank,
        precision=cfg.precision,
        output_dir=output_dir,
    )

    # Create model
    tokenizer_path = cfg.llm_pretrained_path or cfg.my_pretrained_path
    assert (
        tokenizer_path is not None
    ), "Either llm_pretrained_path or my_pretrained_path must be specified."
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        cache_dir=None,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    all_new_tokens = data_wrapper.get_all_token_names()
    num_added_tokens = tokenizer.add_tokens(all_new_tokens)
    if master_process:
        log.info(f"Added {num_added_tokens} tokens to the tokenizer.")
    token_name2_idx_dict = {}
    for token in all_new_tokens:
        token_idx = tokenizer(token, add_special_tokens=False).input_ids[0]
        token_name2_idx_dict[token] = token_idx

    if master_process:
        log.info(f"Token name to index dictionary: {token_name2_idx_dict}")
    data_wrapper.set_token_indices(token_name2_idx_dict)

    torch_dtype = torch.float32
    if cfg.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif cfg.precision == "fp16":
        torch_dtype = torch.half

    model_config = extract_model_config(cfg)
    if cfg.my_pretrained_path is not None:
        log.info(f"Loading model from {cfg.my_pretrained_path}")
        model: SwiftTailorForCausalLM = SwiftTailorForCausalLM.from_pretrained(
            cfg.my_pretrained_path,
            config=model_config,
            panel_edge_indices=data_wrapper.panel_edge_type_indices,
            gt_stats=data_wrapper.gt_stats,
            torch_dtype=torch_dtype,
        )
    else:
        model: SwiftTailorForCausalLM = SwiftTailorForCausalLM(
            config=model_config,
            panel_edge_indices=data_wrapper.panel_edge_type_indices,
            gt_stats=data_wrapper.gt_stats,
            llm_pretrained_path=cfg.llm_pretrained_path,
        ).to(torch_dtype)

    model.model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.freeze_vision_tower()
    model.resize_token_embeddings(len(tokenizer))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")

    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if cfg.evaluate:
        trainer.eval_setup(
            config_dict,
            model,
            tokenizer,
            cfg.conv_type,
        )
        trainer.generation_step(
            trainer.start_step,
            tokenizer,
            subset="validation",
            log_folder=cfg.log_folder,
            convert_to_3d=cfg.convert_to_3d,
        )
    else:
        trainer.training_setup(
            config_dict, model, tokenizer, cfg.conv_type, cfg.from_start, cfg.resume_path
        )
        trainer.fit()


if __name__ == "__main__":
    main()
