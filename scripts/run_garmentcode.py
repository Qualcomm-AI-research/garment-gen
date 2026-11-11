# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Simulate a single sewing pattern into a draped 3D garment mesh via GarmentCode.

Usage:
    python scripts/run_garmentcode.py -p path/to/<name>_specification.json
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TEMPLATE = REPO_ROOT / "configs" / "garmentcode" / "system.template.json"
DEFAULT_SIM_CONFIG = REPO_ROOT / "configs" / "garmentcode" / "default_sim_props.yaml"


def get_command_args():
    """Parse the pattern-spec and sim-config command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern_spec",
        "-p",
        help='Pattern specification JSON file ending in "_specification.json".',
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sim_config",
        "-s",
        help="Path to the simulation config.",
        type=str,
        default=str(DEFAULT_SIM_CONFIG),
    )
    return parser.parse_args()


def main():
    """Build a box mesh from the spec and simulate it into a draped garment mesh."""
    args = get_command_args()

    # Resolve to absolute paths before changing directory.
    spec_path = Path(args.pattern_spec).resolve()
    sim_config = Path(args.sim_config).resolve()

    import pygarment

    gc_assets = Path(pygarment.__file__).resolve().parents[1] / "assets"

    work_dir = tempfile.mkdtemp(prefix="st_garment_sim_")
    system = json.loads(SYSTEM_TEMPLATE.read_text())
    (Path(work_dir) / "system.json").write_text(json.dumps(system))
    os.symlink(gc_assets, Path(work_dir) / "assets")
    os.chdir(work_dir)

    try:
        import pygarment.data_config as data_config
        from pygarment.meshgen.boxmeshgen import BoxMesh
        from pygarment.meshgen.sim_config import PathCofig

        _render_stub = types.ModuleType("pygarment.meshgen.render.pythonrender")
        _render_stub.render_images = lambda *a, **k: None
        sys.modules["pygarment.meshgen.render.pythonrender"] = _render_stub
        from pygarment.meshgen.simulation import run_sim

        props = data_config.Properties(str(sim_config))
        props.set_section_stats(
            "sim",
            fails={},
            sim_time={},
            spf={},
            fin_frame={},
            body_collisions={},
            self_collisions={},
        )
        props.set_section_stats("render", render_time={})

        garment_name, _, _ = spec_path.stem.rpartition("_")  # strips '_specification'

        # Write outputs next to the input spec: <spec_dir>/<garment_name>/...
        paths = PathCofig(
            in_element_path=spec_path.parent,
            out_path=spec_path.parent,
            in_name=garment_name,
            body_name="mean_all",
            smpl_body=False,
            add_timestamp=False,
        )

        print(
            f"Generate box mesh of {garment_name} "
            f"with resolution {props['sim']['config']['resolution_scale']}..."
        )
        print("\nGarment load: ", paths.in_g_spec)

        garment_box_mesh = BoxMesh(paths.in_g_spec, props["sim"]["config"]["resolution_scale"])
        garment_box_mesh.load()
        garment_box_mesh.serialize(
            paths, store_panels=False, uv_config=props["render"]["config"]["uv_texture"]
        )
        props.serialize(paths.element_sim_props)

        t0 = time.perf_counter()
        run_sim(
            garment_box_mesh.name,
            props,
            paths,
            save_v_norms=False,
            store_usd=False,  # False for fast simulation
            optimize_storage=False,
            verbose=False,
        )
        elapsed = time.perf_counter() - t0

        props.serialize(paths.element_sim_props)
        print(f"RECORD SIMULATION TIME: {elapsed:.4f} s")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
