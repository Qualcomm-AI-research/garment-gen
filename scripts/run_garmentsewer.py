# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""SwiftTailor Stage 2: sewing pattern -> 3D garment mesh (sewn + stitched).

For the full text/image -> mesh path (Stage 1 + Stage 2), use
``scripts/run_e2e.py``. This script runs Stage 2 only, starting from an existing
sewing pattern (``*_specification.json``, e.g. a PatternMaker output).

Usage
    python scripts/run_garmentsewer.py --spec path/to/<id>_specification.json
"""

import argparse
import logging
import os
from pathlib import Path

import torch

from swifttailor.garmentsewer.gim.boxmesh_prep import DEFAULT_RESOLUTION
from swifttailor.garmentsewer.pipeline import load_model, max_panels, run_one

logger = logging.getLogger(__name__)


def crawl_specs(path):
    """Return spec JSON files under ``path`` (or the file itself), excluding GT/edited."""
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(
        p
        for p in path.rglob("*_specification.json")
        if "gt_specification" not in p.name and "edited" not in p.name and "editing" not in p.parts
    )


def main():
    """Parse args, load the model, and run the pattern-to-mesh pipeline per spec."""
    ap = argparse.ArgumentParser(description="SwiftTailor sewing-pattern -> mesh pipeline")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--spec", help="a single *_specification.json")
    src.add_argument("--input", help="dir of PatternMaker outputs (crawled for specs)")
    ap.add_argument("--gs-checkpoint", default="weights/garmentsewer/dpt.pt")
    ap.add_argument("--gs-config", default="configs/garmentsewer/dpt_v16.yaml")
    ap.add_argument("--output-dir", default="output/pipeline")
    ap.add_argument("--limit", type=int, default=0, help="cap number of samples (0 = all)")
    ap.add_argument(
        "--gc-root",
        default=str(Path(__file__).resolve().parents[1] / "GarmentCode"),
        help="GarmentCode checkout root (for BoxMesh + body/system paths)",
    )
    ap.add_argument(
        "--sim-config",
        default=str(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "garmentcode"
            / "default_sim_props.yaml"
        ),
        help="sim props YAML (read for the UV-texture render config)",
    )
    ap.add_argument(
        "--resolution",
        type=float,
        default=DEFAULT_RESOLUTION,
        help="boxmesh mesh_resolution; larger = coarser/faster (default %(default)s)",
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    specs = crawl_specs(args.spec or args.input)
    if args.limit:
        specs = specs[: args.limit]
    print(f"[pipeline] {len(specs)} sample(s)")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, dtype = load_model(args.gs_checkpoint, args.gs_config, device)
    n_panels = max_panels()

    for spec in specs:
        try:
            run_one(
                str(spec),
                model,
                dtype,
                device,
                n_panels,
                args.output_dir,
                args.gc_root,
                args.sim_config,
                args.resolution,
            )
        except Exception:
            logger.exception("GarmentSewer failed for %s", Path(spec).stem)


if __name__ == "__main__":
    main()
