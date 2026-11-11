# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Headless garment-only OBJ renderer built on pyrender (EGL on Linux)."""

# Headless OBJ renderer (no PathCofig, no SMPL/body; garment-only framing)
import os
import platform
from typing import Optional

if platform.system() == "Linux":
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import pyrender
import trimesh
from PIL import Image

# -------------------- utils --------------------


def rotate_matrix_y(T, angle_deg):
    """Left-multiply pose ``T`` by a rotation of ``angle_deg`` about the Y axis."""
    a = np.deg2rad(angle_deg)
    R = np.array(
        [[np.cos(a), 0, np.sin(a), 0], [0, 1, 0, 0], [-np.sin(a), 0, np.cos(a), 0], [0, 0, 0, 1]],
        dtype=float,
    )
    return R @ T


def rotate_matrix_x(T, angle_deg):
    """Left-multiply pose ``T`` by a rotation of ``angle_deg`` about the X axis."""
    a = np.deg2rad(angle_deg)
    R = np.array(
        [[1, 0, 0, 0], [0, np.cos(a), -np.sin(a), 0], [0, np.sin(a), np.cos(a), 0], [0, 0, 0, 1]],
        dtype=float,
    )
    return R @ T


def _add_camera(
    scene,
    target_bounds,
    side="front",
    front_camera_location=None,
    yfov=np.pi / 6.0,
    distance_scale=2.5,
):
    """Add a perspective camera framing ``target_bounds`` from the given side."""
    cam = pyrender.PerspectiveCamera(yfov=yfov)
    bmin, bmax = target_bounds
    center = (bmin + bmax) / 2.0
    diag = np.linalg.norm(bmax - bmin)

    if front_camera_location is None:
        fov_deg = np.rad2deg(yfov) * 2
        dist = distance_scale * diag / (2 * np.tan(np.radians(fov_deg / 2)))
        cam_loc = center.copy()
        cam_loc[-1] += max(dist, 1e-3)
    else:
        cam_loc = np.array(front_camera_location, dtype=float)

    T = np.array(
        [[1, 0, 0, cam_loc[0]], [0, 1, 0, cam_loc[1]], [0, 0, 1, cam_loc[2]], [0, 0, 0, 1]],
        dtype=float,
    )
    T = rotate_matrix_x(T, -15)
    T = rotate_matrix_y(T, 20)
    if side == "front":
        pass
    elif side == "back":
        T = rotate_matrix_y(T, 180)
    elif side == "left":
        T = rotate_matrix_y(T, -90)
    elif side == "right":
        T = rotate_matrix_y(T, 90)
    else:
        raise ValueError(f"Invalid side: {side}")
    scene.add(cam, pose=T)


def _load_garment_no_material(
    garm_path, normalized_vertices=False, base_gray=0.6, roughness=0.95, metallic=0.0
):
    """Load and rescale a garment mesh, returning it with a default gray material."""
    g = trimesh.load_mesh(garm_path, process=False)
    if isinstance(g, trimesh.Scene):
        g = trimesh.util.concatenate(tuple(g.dump().geometry.values()))
    if normalized_vertices:
        g.vertices = 400 * g.vertices - 200
    g.vertices = g.vertices / 100.0
    _ = g.vertex_normals  # ensure normals

    default_mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(base_gray, base_gray, base_gray, 1.0),
        metallicFactor=float(metallic),
        roughnessFactor=float(roughness),
        doubleSided=True,
    )
    return g, default_mat


def _add_lights(scene, intensity=20.0):
    """Add a soft three-point point-light setup to the scene."""
    # softer 3-point setup
    positions = [
        np.array([2.0, 2.0, 2.5]),  # key (soft)
        np.array([-2.0, 1.5, 2.0]),  # fill
        np.array([0.0, 3.0, -2.0]),  # rim/back
    ]
    for p in positions:
        L = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=float(intensity))
        T = np.eye(4)
        T[:3, 3] = p
        scene.add(L, pose=T)


def render_obj_simple(
    obj_path: str,
    out_dir: Optional[str] = None,
    resolution=(1080, 1080),
    sides=("front", "back"),
    light_intensity=10.0,
    front_camera_location=None,
    transparent_bg=True,
    distance_scale=2.5,
    normalized_vertices=False,
    base_gray=0.6,
    roughness=0.95,
    metallic=0.0,
    ambient=0.15,  # ambient fill [0..1]
    exposure=1.0,  # post scale on rendered RGBA
):
    """Render an OBJ from each requested side and return the concatenated image.

    Saves a ``*_render.png`` when ``out_dir`` is given.

    Returns:
        The stitched multi-view render as a PIL image.
    """
    stem = os.path.splitext(os.path.basename(obj_path))[0]

    garm, default_mat = _load_garment_no_material(
        obj_path, normalized_vertices, base_gray, roughness, metallic
    )
    pyr_garm = pyrender.Mesh.from_trimesh(garm, material=default_mat, smooth=True)

    panels = []
    for side in sides:
        scene = pyrender.Scene(
            bg_color=(1, 1, 1, 0 if transparent_bg else 1),
            ambient_light=np.array([ambient, ambient, ambient, 1.0], dtype=float),
        )
        scene.add(pyr_garm)

        _add_camera(
            scene,
            garm.bounds,
            side=side,
            front_camera_location=front_camera_location,
            yfov=np.pi / 6.0,
            distance_scale=distance_scale,
        )

        _add_lights(scene, intensity=light_intensity)

        renderer = pyrender.OffscreenRenderer(
            viewport_width=resolution[0], viewport_height=resolution[1]
        )
        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        renderer.delete()

        if exposure != 1.0:
            color = np.clip(color.astype(np.float32) * float(exposure), 0, 255).astype(np.uint8)

        panels.append(color)

    # Concatenate horizontally and save once
    row = np.concatenate(panels, axis=1)  # axis=1 => side-by-side
    img = Image.fromarray(row)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        out_name = os.path.join(out_dir, f"{stem}_render.png")
        img.save(out_name)
        print("Saved:", out_name)

    return img
