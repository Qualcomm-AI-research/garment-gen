# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""PatternMaker-only inference: text/image -> sewing pattern.

Runs just Stage 1 (no GarmentSewer reconstruction, no GarmentCode/mesh
dependencies) — use this to generate sewing patterns from your own text
prompt and/or image. For text/image -> 3D mesh, see ``scripts/run_e2e.py``.

Usage:
    # image -> pattern
    python scripts/run_patternmaker_generate.py input_image=path/to/garment.png
    # text -> pattern
    python scripts/run_patternmaker_generate.py input_text="a long-sleeve hooded jacket"
    # image + text -> pattern
    python scripts/run_patternmaker_generate.py input_image=path/to/garment.png input_text="a long dress"
"""

import logging

import hydra
import torch

from swifttailor.patternmaker.inference import (
    build_inference_dataset,
    extract_model_config,
    generate_specs,
    load_patternmaker,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../configs/patternmaker", config_name="pattern_inference"
)
def main(cfg):
    """Generate sewing pattern specs from the configured text/image input."""
    assert cfg.input_text or cfg.input_image, "Set at least one of input_text=... / input_image=..."

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if cfg.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif cfg.precision == "fp16":
        torch_dtype = torch.half
    else:
        torch_dtype = torch.float32

    ds_cfg = cfg.data_wrapper.dataset
    garment_tokenizer = hydra.utils.instantiate(ds_cfg.garment_tokenizer)
    dataset = build_inference_dataset(
        text=cfg.input_text,
        image=cfg.input_image,
        garment_tokenizer=garment_tokenizer,
        panel_classification=ds_cfg.panel_classification,
        image_size=ds_cfg.image_size,
        max_num_patches=ds_cfg.max_num_patches,
        conv_template=ds_cfg.conv_template,
    )

    model_config = extract_model_config(cfg)
    model, tokenizer = load_patternmaker(
        my_pretrained_path=cfg.my_pretrained_path,
        llm_pretrained_path=cfg.llm_pretrained_path,
        model_config=model_config,
        dataset=dataset,
        torch_dtype=torch_dtype,
        device=device,
    )

    spec_paths = generate_specs(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        out_dir=cfg.output_dir,
        torch_dtype=torch_dtype,
        conv_type=cfg.conv_type,
    )

    if not spec_paths:
        log.error("PatternMaker produced no sewing patterns.")


if __name__ == "__main__":
    main()
