# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""End-to-end SwiftTailor: text/image -> sewing pattern -> 3D garment mesh.

Runs both stages in one process:
  1. PatternMaker (Stage 1)  text/image -> sewing pattern (``*_specification.json``)
  2. GarmentSewer (Stage 2)  sewing pattern -> watertight 3D mesh (``.obj``)

Usage:
    # image -> mesh
    python scripts/run_e2e.py input_image=path/to/garment.png
    # text -> mesh
    python scripts/run_e2e.py input_text="a long-sleeve hooded jacket"
    # image + text -> mesh
    python scripts/run_e2e.py input_image=path/to/garment.png input_text="a long dress"
"""

import logging
import os
from pathlib import Path

import hydra
import torch

from swifttailor.config import GLOBAL_CONFIG
from swifttailor.garmentsewer.gim.boxmesh_prep import DEFAULT_RESOLUTION
from swifttailor.garmentsewer.pipeline import load_model, max_panels, run_one
from swifttailor.patternmaker.inference import (
    build_inference_dataset,
    extract_model_config,
    generate_specs,
    load_patternmaker,
)

log = logging.getLogger(__name__)


def _abspath(p):
    """Resolve path against the repo root to survive the cwd switch that ``prepare_companions`` performs."""
    p = str(p)
    return p if os.path.isabs(p) else str((GLOBAL_CONFIG.root / p).resolve())


@hydra.main(version_base=None, config_path="../configs/patternmaker", config_name="e2e_inference")
def main(cfg):
    """Run PatternMaker then GarmentSewer to turn text/image input into 3D meshes."""
    assert cfg.input_text or cfg.input_image, "Set at least one of input_text=... / input_image=..."

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if cfg.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif cfg.precision == "fp16":
        torch_dtype = torch.half
    else:
        torch_dtype = torch.float32

    # ---- Stage 1: PatternMaker (text/image -> sewing pattern) ----
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

    pm_out = os.path.join(cfg.output_dir, "patternmaker")
    spec_paths = generate_specs(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        out_dir=pm_out,
        torch_dtype=torch_dtype,
        conv_type=cfg.conv_type,
    )

    # Free the VLM before loading the GarmentSewer model.
    del model
    torch.cuda.empty_cache()

    if not spec_paths:
        log.error("PatternMaker produced no sewing patterns; nothing to reconstruct.")
        return

    # ---- Stage 2: GarmentSewer (sewing pattern -> 3D mesh) ----
    gs_model, dtype = load_model(cfg.gs_checkpoint, cfg.gs_config, device)
    n_panels = max_panels()
    resolution = cfg.resolution if cfg.resolution is not None else DEFAULT_RESOLUTION
    # Absolute paths: prepare_companions switches cwd to a temp dir internally.
    sim_config = _abspath(cfg.sim_config)
    gc_root = _abspath(cfg.gc_root)
    mesh_out = _abspath(os.path.join(cfg.output_dir, "mesh"))
    os.makedirs(mesh_out, exist_ok=True)

    for spec in spec_paths:
        try:
            run_one(
                _abspath(spec),
                gs_model,
                dtype,
                device,
                n_panels,
                mesh_out,
                gc_root,
                sim_config,
                resolution,
            )
        except Exception as e:
            import traceback

            log.error(f"[e2e][error] {Path(spec).stem}: {type(e).__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
