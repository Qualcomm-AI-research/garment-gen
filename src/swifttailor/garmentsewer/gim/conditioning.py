# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Bridge boxmesh companion files into GarmentSewer condition/occupancy/stitching inputs."""

from pathlib import Path

from .unwarping import create_condition_from_ply_texture

_DEFAULT_SIZE = 512
_ISLAND_PADDING_PERCENT = 4  # configs/preprocessing.yaml


def _companion_files(spec_path):
    """Return ``(boxmesh mesh, texture, seg)`` sibling files, or ``None``.

    Prefers the GT-dataset ``.ply`` boxmesh, falling back to the ``.obj`` written
    by ``BoxMesh.serialize`` (``boxmesh_prep``); ``read_mesh`` reads either.
    """
    p = Path(spec_path)
    stem = p.stem.replace("_specification", "")
    parent = p.parent
    mesh = parent / f"{stem}_boxmesh.ply"
    if not mesh.exists():
        mesh = parent / f"{stem}_boxmesh.obj"
    texture = parent / f"{stem}_texture.png"
    seg = parent / f"{stem}_sim_segmentation.txt"
    if mesh.exists() and texture.exists() and seg.exists():
        return mesh, texture, seg
    return None


def spec_to_gs_inputs(
    spec_path, size=_DEFAULT_SIZE, island_padding_percent=_ISLAND_PADDING_PERCENT
):
    """Boxmesh companions next to ``spec_path`` -> ``{condition, occupancy, stitching}``."""
    companions = _companion_files(spec_path)
    if companions is None:
        raise FileNotFoundError(f"no boxmesh/texture/seg next to {spec_path}")
    ply_path, texture_path, seg_path = companions

    img = create_condition_from_ply_texture(
        ply_path,
        texture_path,
        seg_path,
        island_padding_percent=island_padding_percent,
        min_uv_size=size,
    )
    return {
        "condition": img,
        "occupancy": img[:, :, 2] > 0,
        "stitching": img[:, :, 3],
    }
