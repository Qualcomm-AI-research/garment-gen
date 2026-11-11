# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Benchmark Stage B: run GarmentSewer on PatternMaker-generated patterns and score against GT.

Consumes the output of ``scripts/benchmark/gen_patternmaker_testset.py``. 
For each generated ``<leaf>_specification.json`` it runs the GarmentSewer stage
(boxmesh prep -> conditioning bridge -> DPTGIM -> remeshing, via
``swifttailor.garmentsewer.pipeline.run_one``) to produce a reconstructed mesh.

Usage:
    python scripts/benchmark/gen_garmentsewer_testset.py \
        --patternmaker-dir output/benchmark/patternmaker \
        --output-dir output/benchmark
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from pygarment.meshgen.boxmeshgen import (
    DegenerateTrianglesError,
    MultiStitchingError,
    StitchingError,
)

from swifttailor.config import GLOBAL_CONFIG
from swifttailor.garmentsewer.gim.boxmesh_prep import DEFAULT_RESOLUTION
from swifttailor.garmentsewer.pipeline import load_model, max_panels, run_one

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_LIST = REPO_ROOT / "configs" / "testset_500samples_full_pipeline.txt"


def _abspath(p):
    """Resolve against repo root; relative paths break when prepare_companions switches cwd."""
    p = str(p)
    return p if os.path.isabs(p) else str((GLOBAL_CONFIG.root / p).resolve())


def get_args():
    """Parse the benchmark's input/output paths and GarmentSewer model arguments."""
    p = argparse.ArgumentParser(
        description="GarmentSewer over PatternMaker specs + metrics vs GT meshes."
    )
    p.add_argument(
        "--patternmaker-dir",
        default="output/benchmark/patternmaker",
        help="Stage-A output dir (contains <leaf>/<leaf>_specification.json).",
    )
    p.add_argument("--output-dir", default="output/benchmark")
    p.add_argument("--sample-list", default=str(DEFAULT_SAMPLE_LIST))
    p.add_argument(
        "--data-root",
        default=str(GLOBAL_CONFIG.gcd_data_dir),
        help="Test-set root holding <full_rel>/<leaf>_sim.ply ground-truth meshes.",
    )
    p.add_argument("--gs-checkpoint", default="weights/garmentsewer/dpt.pt")
    p.add_argument("--gs-config", default="configs/garmentsewer/dpt_v16.yaml")
    p.add_argument("--gc-root", default=str(REPO_ROOT / "GarmentCode"))
    p.add_argument(
        "--sim-config",
        default="configs/garmentcode/default_sim_props.yaml",
    )
    p.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION)
    p.add_argument("--limit", type=int, default=0, help="cap number of samples (0 = all)")
    p.add_argument("--n-sampling-points", type=int, default=20_000)
    p.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Only run GarmentSewer generation, skip the metric comparison.",
    )
    return p.parse_args()


def leaf_to_relpath(sample_list):
    """Map leaf name (rand_XXX) -> full relative path (garments_.../rand_XXX)."""
    mapping = {}
    with open(sample_list) as f:
        for line in f:
            rel = line.strip()
            if rel:
                mapping[rel.split("/")[-1]] = rel
    return mapping


def run_garmentsewer(specs, gs_dir, args, device):
    """Run GarmentSewer on each generated spec -> <gs_dir>/<leaf>.obj.

    Each spec is copied into an isolated work dir first, because run_one ->
    prepare_companions writes boxmesh companion files beside the spec (and wipes
    its element dir with overwrite=True). Working on copies keeps the PatternMaker
    output dir -- including the Stage-A GarmentCode meshes -- untouched.
    """
    gs_dir.mkdir(parents=True, exist_ok=True)
    work_root = gs_dir / "_work"
    model, dtype = load_model(args.gs_checkpoint, args.gs_config, device)
    n_panels = max_panels()
    gc_root = _abspath(args.gc_root)
    sim_config = _abspath(args.sim_config)
    out_dir = _abspath(gs_dir)

    produced = {}
    for leaf, spec in specs.items():
        obj = gs_dir / f"{leaf}.obj"
        if obj.exists():
            produced[leaf] = obj
            continue
        work = work_root / leaf
        work.mkdir(parents=True, exist_ok=True)
        work_spec = work / f"{leaf}_specification.json"
        shutil.copy(spec, work_spec)
        try:
            run_one(
                _abspath(work_spec),
                model,
                dtype,
                device,
                n_panels,
                out_dir,
                gc_root,
                sim_config,
                args.resolution,
            )
            produced[leaf] = obj if obj.exists() else None
        except (DegenerateTrianglesError, MultiStitchingError, StitchingError) as error:
            print(f"[garmentsewer][error] {leaf}: {type(error).__name__}: {error}")
            produced[leaf] = None
        except Exception as error:
            print(f"[garmentsewer][error] {leaf}: {type(error).__name__}: {error}")
            produced[leaf] = None
    return produced


def compute_metrics(leaves, gt_paths, pred_paths, n_points, device):
    """Call metrics._compute_metrics over aligned ground-truth and prediction paths."""
    import metrics

    return metrics._compute_metrics(gt_paths, pred_paths, n_sampling_points=n_points, device=device)


def main():
    """Reconstruct meshes from generated specs and score them against GT meshes."""
    args = get_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Absolute paths: run_one -> prepare_companions switches cwd internally and
    # may not restore it on a GarmentCode error, so keep all path checks here
    # cwd-independent.
    pm_dir = Path(_abspath(args.patternmaker_dir))
    data_root = Path(_abspath(args.data_root))
    out_root = Path(_abspath(args.output_dir))
    leaf2rel = leaf_to_relpath(args.sample_list)

    # Collect generated specs from Stage A, in sample-list order.
    specs = {}
    for leaf, rel in leaf2rel.items():
        spec = pm_dir / leaf / f"{leaf}_specification.json"
        if spec.exists():
            specs[leaf] = spec
    if args.limit:
        specs = dict(list(specs.items())[: args.limit])
    if not specs:
        raise SystemExit(
            f"No generated specs found under {pm_dir} (run gen_patternmaker_testset.py first)."
        )
    print(f"[garmentsewer] {len(specs)} generated spec(s) from {pm_dir}")

    # Snapshot the Stage-A GarmentCode simulate-until-success meshes to a stable
    # dir BEFORE Stage B (prepare_companions would otherwise overwrite them).
    gc_dir = out_root / "garmentcode"
    gc_dir.mkdir(parents=True, exist_ok=True)
    gc_meshes = {}
    for leaf in specs:
        src = pm_dir / leaf / leaf / f"{leaf}_sim.obj"
        if src.exists():
            dst = gc_dir / f"{leaf}.obj"
            shutil.copy(src, dst)
            gc_meshes[leaf] = dst
        else:
            gc_meshes[leaf] = None

    # Stage B: GarmentSewer reconstruction.
    gs_dir = out_root / "garmentsewer"
    cwd = os.getcwd()
    gs_meshes = run_garmentsewer(specs, gs_dir, args, device)
    os.chdir(cwd)  # restore in case a GarmentCode error left cwd in a temp dir
    n_gs = sum(v is not None for v in gs_meshes.values())
    print(f"[garmentsewer] produced {n_gs}/{len(specs)} meshes -> {gs_dir}")

    if args.skip_metrics:
        return

    # Stage C: build aligned GT / prediction path lists over samples with a GT mesh.
    leaves, gt_paths, gc_preds, gs_preds = [], [], [], []
    for leaf in specs:
        rel = leaf2rel[leaf]
        gt = data_root / rel / f"{leaf}_sim.ply"
        if not gt.exists():
            print(f"[metrics][skip] missing GT mesh: {gt}")
            continue
        gc_obj = gc_meshes.get(leaf)  # snapshotted GarmentCode sim (Stage A)
        gs_obj = gs_meshes.get(leaf)
        leaves.append(leaf)
        gt_paths.append(gt)
        gc_preds.append(gc_obj if (gc_obj and Path(gc_obj).exists()) else None)
        gs_preds.append(gs_obj if (gs_obj and Path(gs_obj).exists()) else None)

    if not leaves:
        raise SystemExit("No samples with a ground-truth mesh to score.")

    print(
        f"[metrics] scoring {len(leaves)} sample(s): "
        f"garmentcode found {sum(p is not None for p in gc_preds)}, "
        f"garmentsewer found {sum(p is not None for p in gs_preds)}"
    )

    report = {
        "n_samples": len(leaves),
        "garmentcode_found": sum(p is not None for p in gc_preds),
        "garmentsewer_found": sum(p is not None for p in gs_preds),
        "metrics": {
            "garmentcode_vs_gt": compute_metrics(
                leaves, gt_paths, gc_preds, args.n_sampling_points, device
            ),
            "garmentsewer_vs_gt": compute_metrics(
                leaves, gt_paths, gs_preds, args.n_sampling_points, device
            ),
        },
    }

    report_path = out_root / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[metrics] wrote {report_path}")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
