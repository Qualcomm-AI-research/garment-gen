# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Load mesh geometry and per-vertex segmentation labels from OBJ/PLY files."""

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import trimesh


def v_id_map(vertices: np.ndarray) -> Tuple[List[int], np.ndarray]:
    """Map adjacent duplicates to IDs and return the map with unique vertices."""
    # Temporarily resolve the issue of duplicate vertices by trimesh
    # See Loading segmentation and vertex labels for ply mesh in:
    # https://www.research-collection.ethz.ch/handle/20.500.11850/690432
    v_map: List[int] = [0] * len(vertices)
    deduplicated_vertices = [vertices[0]]
    for i in range(1, len(vertices)):
        if all(vertices[i - 1] == vertices[i]):
            v_map[i] = v_map[i - 1]
        else:
            v_map[i] = v_map[i - 1] + 1
            deduplicated_vertices.append(vertices[i])
    return v_map, np.array(deduplicated_vertices)


def read_mesh(
    file_path: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a mesh and return its UVs, deduplicated vertices, faces, and normals.

    Returns:
        UV coords, deduplicated vertex coords, texture faces, vertex faces,
        and deduplicated vertex normals.
    """
    assert Path(file_path).suffix in [
        ".obj",
        ".ply",
    ], "File must be an OBJ or PLY file."

    mesh: trimesh.Trimesh = trimesh.load(file_path)

    # Texture
    # UV coordinates
    uv_coords = np.array(mesh.visual.uv)
    # Create texture faces
    tid_faces = np.array(mesh.faces)

    # 3D vertice
    # Vertex coordinates
    vertex_coords = np.array(mesh.vertices)
    # Vertex faces
    v_map, dedup_vertex_coords = v_id_map(vertex_coords)
    vid_faces = np.array([[v_map[i] for i in face] for face in tid_faces])

    # Vertex normals
    vertex_normals = np.array(mesh.vertex_normals)
    dedup_vertex_normals = vertex_normals[np.unique(v_map)]

    return (
        uv_coords,
        dedup_vertex_coords,
        tid_faces,
        vid_faces,
        dedup_vertex_normals,
    )


def read_segmentation_file(segmentation_path: Union[str, Path]) -> List[str]:
    """Return one label per vertex from a segmentation file."""
    with open(segmentation_path, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines
