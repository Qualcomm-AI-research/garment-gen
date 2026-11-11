# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Generate boxmesh/texture/segmentation companion files for a sewing-pattern spec."""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from swifttailor.config import GLOBAL_CONFIG

DEFAULT_RESOLUTION = 1.0
DEFAULT_TEXTURE_DPI = 100

_COMPANION_SUFFIXES = ("_boxmesh.obj", "_texture.png", "_sim_segmentation.txt")


def companions_exist(spec_path):
    """True iff all three file-based-bridge companions already sit next to the spec."""
    spec_path = Path(spec_path)
    stem = spec_path.stem.replace("_specification", "")
    return all((spec_path.parent / f"{stem}{sfx}").exists() for sfx in _COMPANION_SUFFIXES)


def prepare_companions(
    spec_path,
    gc_root,
    sim_config,
    resolution=DEFAULT_RESOLUTION,
    texture_dpi=DEFAULT_TEXTURE_DPI,
    overwrite=False,
):
    """Write ``*_boxmesh.obj`` / ``*_texture.png`` / ``*_sim_segmentation.txt`` beside ``spec_path``.

    Idempotent: returns immediately if the companions already exist unless
    ``overwrite``. Returns the spec's parent dir (where the companions now live).
    """
    spec_path = Path(spec_path).resolve()
    if not overwrite and companions_exist(spec_path):
        return spec_path.parent

    name = spec_path.stem.replace("_specification", "")

    gc_root_str = str(Path(gc_root).resolve())
    if gc_root_str not in sys.path:
        sys.path.insert(0, gc_root_str)

    import pygarment
    import pygarment.data_config as data_config
    from pygarment.meshgen.boxmeshgen import BoxMesh, Edge
    from pygarment.meshgen.sim_config import PathCofig

    if not hasattr(Edge, "name"):
        Edge.name = "edge"

    # pygarment resolves body/sim paths via `./system.json` and `./assets/...`
    # relative to cwd. Run in a private temp dir with a system.json built
    # from our vendored template and an assets copy from the GarmentCode checkout,
    # so nothing is read from or written into GarmentCode/ itself.
    gc_assets = Path(pygarment.__file__).resolve().parents[1] / "assets"
    system_template = GLOBAL_CONFIG.config_dir / "garmentcode" / "system.template.json"

    previous_cwd = Path.cwd()
    work_dir = tempfile.mkdtemp(prefix="st_boxmesh_")
    try:
        system = json.loads(system_template.read_text())
        (Path(work_dir) / "system.json").write_text(json.dumps(system))
        shutil.copytree(gc_assets, Path(work_dir) / "assets")
        os.chdir(work_dir)

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
        uv_config = dict(props["render"]["config"]["uv_texture"])
        uv_config["dpi"] = texture_dpi
        uv_config["fabric_grain_texture_path"] = None  # unused fabric variant -- skip

        # out_el = out_path / name, so files land in <spec dir>/<name>/ then move up.
        paths = PathCofig(
            in_element_path=spec_path.parent,
            out_path=spec_path.parent,
            in_name=name,
            body_name="mean_all",
            smpl_body=False,
            add_timestamp=False,
        )
        garment = BoxMesh(paths.in_g_spec, resolution)
        garment.load()
        garment.serialize(paths, store_panels=False, uv_config=uv_config)
    finally:
        os.chdir(previous_cwd)
        shutil.rmtree(work_dir, ignore_errors=True)

    # Move just the companions the bridge needs from out_el up beside the spec.
    out_el = Path(paths.out_el)
    for src in (paths.g_box_mesh, paths.g_texture, paths.g_mesh_segmentation):
        src = Path(src)
        dst = spec_path.parent / src.name
        if src.resolve() != dst.resolve():
            shutil.move(str(src), str(dst))
    if out_el.resolve() != spec_path.parent.resolve():
        shutil.rmtree(out_el, ignore_errors=True)

    return spec_path.parent
