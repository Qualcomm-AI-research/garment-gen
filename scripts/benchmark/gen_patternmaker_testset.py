# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Benchmark Stage A: generate image-and-text patterns with simulate-until-success.

Runs ``scripts/run_patternmaker.py`` in evaluate mode over every sample in a
list file (default ``configs/testset_500samples_full_pipeline.txt``). Each
sample is fed its ``<leaf>_render_front.png`` as the image and the
``description`` from its ``captions.json`` as the text (``sampling_type=image_text``),
and ``convert_to_3d=True`` simulates each generated pattern into a draped mesh
via the GarmentCode simulate-until-success loop.

Usage:
    python scripts/benchmark/gen_patternmaker_testset.py
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_LIST = REPO_ROOT / "configs" / "testset_500samples_full_pipeline.txt"


def get_args():
    """Parse the benchmark's sample-list, path, and limit arguments."""
    p = argparse.ArgumentParser(
        description="PatternMaker image+text benchmark with simulate-until-success."
    )
    p.add_argument(
        "--sample-list",
        default=str(DEFAULT_SAMPLE_LIST),
        help="Text file of sample names (one per line, e.g. "
        "garments_5000_24/default_body/rand_XXX). Default: %(default)s",
    )
    p.add_argument("--output-dir", default="output/benchmark")
    p.add_argument("--my-pretrained-path", default="weights/patternmaker")
    p.add_argument("--llm-pretrained-path", default="cache/InternVL3-2B-Instruct")
    p.add_argument("--limit", type=int, default=0, help="cap number of samples (0 = all)")
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to run run_patternmaker.py with.",
    )
    return p.parse_args()


def main():
    """Write the sample split and invoke run_patternmaker.py in evaluate mode."""
    args = get_args()

    with open(args.sample_list) as f:
        samples = [line.strip() for line in f if line.strip()]
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit(f"No samples found in {args.sample_list}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # full_test drives the validation loader that generation_step iterates; train
    # is a 1-element stub (token names / gt_stats come from the tokenizer).
    split_path = output_dir / "split.json"
    split_path.write_text(json.dumps({"train": samples[:1], "full_test": samples}))
    print(f"[benchmark] {len(samples)} sample(s); split -> {split_path}")

    cmd = [
        args.python,
        str(REPO_ROOT / "scripts" / "run_patternmaker.py"),
        "--config-name",
        "benchmark_inference",
        f"my_pretrained_path={args.my_pretrained_path}",
        f"llm_pretrained_path={args.llm_pretrained_path}",
        "evaluate=True",
        "convert_to_3d=True",
        "sampling_type=image_text",
        f"data_wrapper.split_file={split_path}",
        f"log_folder={output_dir / 'patternmaker'}",
    ]

    # MASTER_ADDR/PORT read from env with fallback -- lets concurrent shards on the
    # same GPU pick distinct ports so DDP init on RANK=0/WORLD_SIZE=1 doesn't collide.
    env = {
        **os.environ,
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": os.environ.get("MASTER_ADDR", "localhost"),
        "MASTER_PORT": os.environ.get("MASTER_PORT", "29500"),
        "WANDB_MODE": "disabled",
    }

    print(f"[benchmark] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
