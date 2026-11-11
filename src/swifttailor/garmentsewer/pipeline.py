# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""GarmentSewer stage: sewing pattern -> 3D garment mesh (sewn + stitched).

Shared core for the SwiftTailor pipelines. Both ``scripts/run_garmentsewer.py``
(pattern -> mesh) and ``scripts/run_e2e.py`` (text/image -> pattern -> mesh)
import these helpers so the reconstruction path is defined once.

Stages
------
  1. BoxMesh prep  (sewing pattern -> boxmesh companions)    gim.boxmesh_prep.prepare_companions
  2. Bridge        (companions -> GarmentSewer inputs)       gim.conditioning.spec_to_gs_inputs
  3. GarmentSewer  (condition -> Geometry Image)             dpt.DPTGIM
  4. Remeshing     (GIM + stitching -> watertight mesh)      gim.remeshing.create_obj_from_gim
"""

from pathlib import Path
from time import time

import torch
import yaml

from swifttailor.config import GLOBAL_CONFIG
from swifttailor.garmentsewer.dpt import DPTGIM
from swifttailor.garmentsewer.gim.boxmesh_prep import prepare_companions
from swifttailor.garmentsewer.gim.conditioning import spec_to_gs_inputs
from swifttailor.garmentsewer.gim.remeshing import MAX_BUCKET, create_obj_from_gim


def max_panels():
    """Return the number of panel names defined in ``panel_names.txt``."""
    with open(GLOBAL_CONFIG.config_dir / "panel_names.txt") as f:
        return len(f.readlines())


def load_model(checkpoint, config_path, device):
    """Build a ``DPTGIM`` from the config, load weights, and return it with its dtype."""
    # PartialState is used for accelerate's logger
    from accelerate import PartialState

    PartialState()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model = DPTGIM(**cfg["model"]["kwargs"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval().to(device)
    dtype = model.pretrained.encoder.layer[0].attention.attention.query.weight.dtype
    return model, dtype


def predict_gim(model, dtype, device, condition_label, max_panels):
    """Condition label map -> predicted geometry image (H, W, 3).

    ``condition_label`` is already the training-time normalisation
    (``label / n_panels``, produced by ``spec_to_gs_inputs``), so it is fed in
    as-is. ``max_panels`` is accepted for signature compatibility but unused.
    """
    seg = condition_label
    x = torch.from_numpy(seg).float()[None, None].repeat(1, 3, 1, 1).to(device, dtype)
    with torch.no_grad():
        pred = model(segmentation_map=x)
    return pred[0].permute(1, 2, 0).float().cpu().numpy()


def run_one(spec_path, model, dtype, device, max_panels, out_dir, gc_root, sim_config, resolution):
    """Run the full pattern-to-mesh pipeline for one spec and write the ``.obj``.

    Returns:
        Path to the written garment mesh.
    """
    sample_id = Path(spec_path).stem.replace("_specification", "")
    print(f"[pipeline] {sample_id}")

    st = time()
    # Stage 1 -- sewing pattern -> boxmesh companions (coarse, no warp/pyrender)
    prepare_companions(spec_path, gc_root, sim_config, resolution=resolution, overwrite=True)

    # Stage 2 -- companions -> GarmentSewer inputs (file-based bridge)
    g = spec_to_gs_inputs(spec_path)
    print("Prepare data: ", time() - st)
    # Stage 3 -- condition -> geometry image
    gim = predict_gim(model, dtype, device, g["condition"][:, :, 2] * g["occupancy"], max_panels)

    # Stage 4 -- remeshing + dynamic stitching -> watertight garment mesh
    out_obj = str(Path(out_dir) / f"{sample_id}.obj")
    create_obj_from_gim(
        gim=gim,
        occupancy=g["occupancy"],
        stitching_map=g["stitching"],
        max_bucket=MAX_BUCKET,
        saving_path=out_obj,
        spec_path=spec_path,
    )
    print(f"[pipeline] wrote {out_obj}")
    return out_obj
