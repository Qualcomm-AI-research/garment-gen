# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Unwrap and pack UV islands, then rasterize them into geometry/condition images."""

from math import ceil, floor
from pathlib import Path
from time import time
from typing import List, Optional, Tuple, Union

import cv2
import igl
import imagesize
import numpy as np

from swifttailor.config import GLOBAL_CONFIG

from .object_loading import read_mesh, read_segmentation_file

_MAX_SIZE = 1 << 16
_NON_STITCHED_BOUNDARY = 1


class _Rectangle:
    """Rectangle class for packing UV islands in the unwarp_UV function."""

    def __init__(
        self,
        rect_id: int,
        width,
        height,
        x=0,
        y=0,
    ):
        """Store the rectangle id, size, and placement origin."""
        self.rect_id = rect_id
        self.x = x
        self.y = y
        self.w = width
        self.h = height

    @staticmethod
    def pack_rows(rects: List["_Rectangle"], container_size: int = 1024) -> bool:
        """Shelf-pack rectangles into a square container, setting each ``x``/``y``.

        Returns:
            ``True`` if every rectangle fits within ``container_size``.
        """
        rects = sorted(rects, key=lambda r: r.h, reverse=True)

        cur_y = 0
        cur_x = 0
        max_row_height = 0
        for r in rects:
            if cur_x + r.w > container_size:
                cur_x = 0
                cur_y += max_row_height
                max_row_height = 0

            # Cannot pack this rectangle
            if cur_y + r.h > container_size:
                return False

            r.x = cur_x
            r.y = cur_y

            cur_x += r.w
            max_row_height = max(max_row_height, r.h)

        return True


def _uv_connected_components(face_texture_coords):
    """Label connected UV islands, returning per-vertex/per-face labels and count."""
    # Find connected components of face and vertex texture coords
    face_components = igl.facet_components(face_texture_coords)
    # Some libigl builds return (count, per_face_labels); keep the per-face array.
    if isinstance(face_components, tuple):
        face_components = next(
            np.asarray(x)
            for x in face_components
            if np.asarray(x).shape == (len(face_texture_coords),)
        )
    face_components = np.asarray(face_components)
    num_ccs = int(face_components.max()) + 1

    # Get vertex components from face_components (igl.vertex_components uses a
    # different adjacency matrix and returns mismatched component ids).
    vert_components = np.zeros(face_texture_coords.max() + 1, dtype=int)
    for i in range(num_ccs):
        verts_in_cc = np.unique(face_texture_coords[face_components == i])
        vert_components[verts_in_cc] = i

    return vert_components, face_components, num_ccs


def unwarp_UV(
    texture_coords,
    face_texture_coords,
    padding: float,
) -> Tuple[np.ndarray, List]:
    """Unwrap each UV island and pack them into a shared texture layout.

    Returns:
        The repacked per-vertex UVs and the per-island boundary loops.
    """
    # Unwrap uvs for each connected component------------------------
    vert_components, face_components, num_ccs = _uv_connected_components(face_texture_coords)

    # transform all UVs to update obj file
    all_uvs = np.zeros((texture_coords.shape[0], 2))
    boundary_uv_to_draw = []  # only draw the boundary UVs

    translate_Y = 0
    translate_X = 0

    half_padding = padding / 2

    # Loop through each connected component
    list_rects = []
    for i in range(num_ccs):

        # Get faces and vertices of connected component
        faces_in_cc = np.where(face_components == i)[0]
        face_vts_in_cc = face_texture_coords[faces_in_cc]

        # Find boundary loop
        bound_verts = igl.boundary_loop(face_vts_in_cc)
        bound_vert_pos = texture_coords[bound_verts]

        # get all vertices of connected component
        verts_in_cc = np.where(vert_components == i)[0]
        all_vert_pos = texture_coords[verts_in_cc]

        # Shift component by bounding box
        bbox = bound_vert_pos.min(axis=0), bound_vert_pos.max(axis=0)
        _bbox = all_vert_pos.min(axis=0), all_vert_pos.max(axis=0)

        bbox_len_Y = (bbox[1][1] - bbox[0][1]) + padding
        bbox_len_X = (bbox[1][0] - bbox[0][0]) + padding

        list_rects.append(_Rectangle(i, bbox_len_X, bbox_len_Y))

    # Container size search
    lower, upper = 0, _MAX_SIZE
    container_size = 0
    while lower < upper:
        mid = (lower + upper) // 2
        if _Rectangle.pack_rows(list_rects, mid):
            container_size = mid
            upper = mid
        else:
            lower = mid + 1

    assert container_size > 0, "Cannot pack UV islands into a texture"
    _Rectangle.pack_rows(list_rects, container_size)

    print(f"[DEBUG] Container size: {container_size}")

    if not container_size:
        raise ValueError("Cannot pack UV islands into a texture")

    for i in range(num_ccs):
        ccs_id = list_rects[i].rect_id
        translate_X = list_rects[i].x
        translate_Y = list_rects[i].y

        # Get faces and vertices of connected component
        faces_in_cc = np.where(face_components == ccs_id)[0]
        face_vts_in_cc = face_texture_coords[faces_in_cc]

        # get all vertices of connected component
        verts_in_cc = np.where(vert_components == ccs_id)[0]
        all_vert_pos = texture_coords[verts_in_cc]

        # Find boundary loop
        bound_verts = igl.boundary_loop(face_vts_in_cc)
        bound_vert_pos = texture_coords[bound_verts]
        # Translation to center
        _min_X, _min_Y = bound_vert_pos.min(axis=0)

        # translate boundary positions
        verts_translated_bound = [
            (
                x + translate_X + half_padding - _min_X,
                y + translate_Y + half_padding - _min_Y,
            )
            for x, y in bound_vert_pos
        ]
        boundary_uv_to_draw.append(verts_translated_bound)

        # translate all positions
        verts_translated = np.array(
            [
                (
                    x + translate_X + half_padding - _min_X,
                    y + translate_Y + half_padding - _min_Y,
                )
                for x, y in all_vert_pos
            ]
        )
        all_uvs[verts_in_cc] = verts_translated

    return all_uvs, boundary_uv_to_draw


def normalize_squared_UVs(
    all_uvs,
    padding: float,
) -> Tuple[np.ndarray, float]:
    """Normalize UVs and turn into a square."""
    # Padding to both sides of each UVs, so the padding is halved
    # Max coordinates already include first half of padding
    # --hpd|UVs|hpd--
    half_padding = padding / 2

    # normalize all_uvs
    uv_list_raw = np.array(all_uvs)
    uv_list = uv_list_raw

    norm_x = max(uv_list_raw[:, 0]) + half_padding
    norm_y = max(uv_list_raw[:, 1]) + half_padding
    norm = max(norm_x, norm_y)

    uv_list[:, 0] = uv_list_raw[:, 0] / norm
    uv_list[:, 1] = uv_list_raw[:, 1] / norm

    return uv_list, norm


def _in_triangle(p, a, b, c) -> bool:
    """Check if point p is inside triangle abc."""

    def _sign(a, b, c):
        """Return the signed area (orientation) of triangle ``abc``."""
        return (a[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (a[1] - c[1])

    d1 = _sign(p, a, b)
    d2 = _sign(p, b, c)
    d3 = _sign(p, c, a)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def interpolate_uv(
    uv_coords: np.ndarray,
    norm_vertex_coords: np.ndarray,
    vertex_faces: np.ndarray,
    texture_faces: np.ndarray,
    seg: Tuple[List[str], np.ndarray],
    min_tolerate: float = 0,
    tolerate_eps: float = 1e-6,
    min_uv_size: int = 2048,
    vertex_normals: Optional[np.ndarray] = None,
):
    """Interpolate UV coordinates to create a geometry image."""
    is_multi_seg, seg = seg

    actual_size = min_uv_size
    print(f"[DEBUG] Actual size: {actual_size}")

    width, height = actual_size, actual_size

    # Init image array
    img_array = np.zeros((height, width, 4), dtype=float)
    img_array[:, :, 3] = 0
    visited_array = np.zeros((height, width), dtype=bool)

    # scale to original uv coordinates and flip the V axis
    scaled_coords = uv_coords * [width, height]
    scaled_coords[:, 1] = height - scaled_coords[:, 1]

    print("[DEBUG] Number of original vertices:", scaled_coords.shape[0])
    print(
        "[DEBUG] Number of overlapped vertices: ",
        scaled_coords.shape[0] - np.unique(scaled_coords.astype(int), axis=0).shape[0],
    )

    ############# Texture-uv and 3d-vertex mapping #############
    uv_vertex_mapping = {}
    for face_id in range(texture_faces.shape[0]):
        for uv_id, vertex_id in zip(texture_faces[face_id], vertex_faces[face_id]):
            uv_vertex_mapping[uv_id] = vertex_id

    ############# Tracking #############
    all_time = time()
    cnt = 0

    ############# Draw boundary edges #############
    vert_components, face_components, num_ccs = _uv_connected_components(texture_faces)
    uv_boundary_edges = set()
    for css_id in range(num_ccs):
        faces_in_cc = np.where(face_components == css_id)[0]
        face_vts_in_cc = texture_faces[faces_in_cc]

        # Flip vertically the connect component if the normal is pointing outwards
        face_vts_coords = scaled_coords[face_vts_in_cc]
        v0, v1 = (
            face_vts_coords[:, 1, :] - face_vts_coords[:, 0, :],
            face_vts_coords[:, 2, :] - face_vts_coords[:, 0, :],
        )
        n = np.cross(v0, v1).sum()
        # Flip vertically the connect component if the normal is pointing outwards
        if n < 0:
            min_x, max_x = (np.min(face_vts_coords[:, :, 0]), np.max(face_vts_coords[:, :, 0]))
            face_vts_coords[:, :, 0] = min_x + max_x - face_vts_coords[:, :, 0]
            scaled_coords[face_vts_in_cc] = face_vts_coords

        bound_verts = igl.boundary_loop(face_vts_in_cc)
        bound_3d_verts = [uv_vertex_mapping[uv_id] for uv_id in bound_verts]
        bound_vert_pos = scaled_coords[bound_verts]
        colors = norm_vertex_coords[bound_3d_verts]
        segs = seg[bound_3d_verts]
        is_multi_segs = is_multi_seg[bound_3d_verts]
        if vertex_normals is not None:
            colors = np.concatenate([colors, vertex_normals[bound_3d_verts]], axis=1)

        for v1 in range(len(bound_vert_pos)):
            v2 = (v1 + 1) % len(bound_vert_pos)
            seg1 = segs[v1]
            seg2 = segs[v2]
            is_multi_seg1 = is_multi_segs[v1]
            is_multi_seg2 = is_multi_segs[v2]

            uv_boundary_edges.add(
                (
                    min(bound_3d_verts[v1], bound_3d_verts[v2]),
                    max(bound_3d_verts[v1], bound_3d_verts[v2]),
                )
            )

            x1, y1 = np.ceil(bound_vert_pos[v1]).astype(int)
            x2, y2 = np.ceil(bound_vert_pos[v2]).astype(int)
            c1 = colors[v1]
            c2 = colors[v2]

            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx - dy
            points = []
            while True:
                points.append((x1, y1))
                if x1 == x2 and y1 == y2:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x1 += sx
                if e2 < dx:
                    err += dx
                    y1 += sy

            for i in range(len(points)):
                cnt += 1
                x, y = points[i]
                img_array[y, x, :3] = (c1 * (len(points) - i) + c2 * i) / len(points)
                visited_array[y, x] = True
                if img_array[y, x, 3] > _NON_STITCHED_BOUNDARY:
                    continue
                if seg1 == _NON_STITCHED_BOUNDARY or seg2 == _NON_STITCHED_BOUNDARY:
                    img_array[y, x, 3] = _NON_STITCHED_BOUNDARY
                elif seg1 == seg2:
                    img_array[y, x, 3] = seg1
                else:
                    if is_multi_seg1:
                        img_array[y, x, 3] = seg2
                    else:
                        img_array[y, x, 3] = seg1

    def _interpolate(face_id, cnt):
        """Barycentric-fill one triangle's interior pixels with interpolated vertex colors.

        Returns:
            The running visited-pixel count including this face.
        """
        a_id, b_id, c_id = texture_faces[face_id]
        a = np.ceil(scaled_coords[a_id]).astype(int)
        b = np.ceil(scaled_coords[b_id]).astype(int)
        c = np.ceil(scaled_coords[c_id]).astype(int)

        # bounding box for the triangle
        min_x = max(0, min(a[0], b[0], c[0]))
        max_x = min(width, max(a[0], b[0], c[0]))
        min_y = max(0, min(a[1], b[1], c[1]))
        max_y = min(height, max(a[1], b[1], c[1]))

        # vertex colors = 3d coordinate
        vertex_coords = norm_vertex_coords[vertex_faces[face_id]]
        if vertex_normals is not None:
            vertex_coords = np.concatenate(
                [vertex_coords, vertex_normals[vertex_faces[face_id]]], axis=1
            )
        a_color, b_color, c_color = vertex_coords

        # barycentric interpolation
        v0 = b - a
        v1 = c - a

        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        denom = d00 * d11 - d01 * d01

        if np.isclose(denom, 0):
            denom = 1  # line instead of triangle

        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                p = np.array([x, y])
                if not visited_array[y, x] and _in_triangle(p, a, b, c):
                    cnt += 1
                    v2 = p - a
                    d20 = np.dot(v2, v0)
                    d21 = np.dot(v2, v1)

                    v = (d11 * d20 - d01 * d21) / denom

                    w = (d00 * d21 - d01 * d20) / denom
                    u = 1 - v - w

                    # Compute interpolated color
                    color = a_color * u + b_color * v + c_color * w

                    img_array[y, x, :3] = color
                    visited_array[y, x] = True

        return cnt

    for face_id in range(texture_faces.shape[0]):
        cnt = _interpolate(face_id, cnt)

    # Interpolate

    print("[DEBUG] Number of interpolated vertices (origin inclusive): ", cnt)
    print("[DEBUG] Time: ", time() - all_time)
    return img_array


def create_gim_from_ply_texture(
    ply_file_path: Union[str, Path],
    texture_file_path: Union[str, Path],
    segmentation_file_path: Union[str, Path],
    gim_path: Union[str, Path],
    max_dim: int = 300,
    island_padding_percent: int = 2,
    plot: bool = False,
    use_vertex_normals: bool = False,
    **kwargs,
):
    """Create geometry UV image from ply and segmentation file."""
    (
        uv_coords,
        vertex_coords,
        texture_faces,
        vertex_faces,
        vertex_normals,
    ) = read_mesh(ply_file_path)

    # Load texture image size
    width, height = imagesize.get(texture_file_path)
    uv_coords = uv_coords * [width, height]

    raw_seg = read_segmentation_file(segmentation_file_path)
    is_multi_seg = np.array(
        [True if s.startswith("stitch") and "," in s else False for s in raw_seg]
    )
    seg = np.array(
        [
            (
                int(s.split(",")[0].split("_")[-1]) + _NON_STITCHED_BOUNDARY + 1
                if s.startswith("stitch")
                else _NON_STITCHED_BOUNDARY
            )
            for s in raw_seg
        ]
    )

    assert 0 < island_padding_percent <= 100, "Invalid padding percentage"
    _max_ori_uv_size = max(uv_coords.max(axis=0) - uv_coords.min(axis=0))
    padding = _max_ori_uv_size * (island_padding_percent / 100)

    all_uvs, _ = unwarp_UV(
        uv_coords,
        texture_faces,
        padding=padding,
    )

    uv_list, _ = normalize_squared_UVs(
        all_uvs,
        padding=padding,
    )

    # Create geometry uv file
    norm_vertex_coords = (vertex_coords + max_dim) / (2 * max_dim)
    img_array = interpolate_uv(
        uv_list,
        norm_vertex_coords,
        vertex_faces,
        texture_faces,
        (is_multi_seg, seg),
        vertex_normals=vertex_normals if use_vertex_normals else None,
        **kwargs,
    )

    if plot:
        import matplotlib.pyplot as plt

        plt.imshow(img_array)
        plt.show()

    np.savez_compressed(gim_path, img_array)
    return img_array


def interpolate_seg_stitch(
    uv_coords: np.ndarray,
    panel_label: np.ndarray,
    vertex_faces: np.ndarray,
    texture_faces: np.ndarray,
    seg: Tuple[np.ndarray, np.ndarray],
    min_uv_size: int = 2048,
):
    """Rasterize panels into an ``(H, W, 4)`` image.

    Same packing / boundary logic as :func:`interpolate_uv`, but the
    interpolated 3D vertex coordinates are replaced by the panel's label (its
    "colour"): channels 0-2 hold the per-panel ``panel_label`` (majority vote
    over the connected component's 3D vertices) and channel 3 holds the
    stitching tag drawn along the boundary edges (identical to ``interpolate_uv``).
    The non-zero label region doubles as the occupancy mask.

    Per panel, the boundary line is drawn first (Bresenham, carrying the panel
    label on channels 0-2 and the stitching tag on channel 3) and its integer
    boundary vertices are captured; ``cv2.fillPoly`` then fills the interior from
    those vertices instead of the per-face triangle raster. ``fillPoly``
    over-fills the edge by ~1px, so the silhouette is a slight superset of the
    triangle raster (accepted: never an under-fill).

    ``panel_label`` is indexed by 3D vertex id; ``seg`` is ``(is_multi_seg,
    stitch_seg)`` exactly as passed to :func:`interpolate_uv`.
    """
    is_multi_seg, seg = seg

    actual_size = min_uv_size
    width, height = actual_size, actual_size

    img_array = np.zeros((height, width, 4), dtype=float)

    # scale to original uv coordinates and flip the V axis
    scaled_coords = uv_coords * [width, height]
    scaled_coords[:, 1] = height - scaled_coords[:, 1]

    ############# Texture-uv and 3d-vertex mapping #############
    uv_vertex_mapping = {}
    for face_id in range(texture_faces.shape[0]):
        for uv_id, vertex_id in zip(texture_faces[face_id], vertex_faces[face_id]):
            uv_vertex_mapping[uv_id] = vertex_id

    vert_components, face_components, num_ccs = _uv_connected_components(texture_faces)

    for css_id in range(num_ccs):
        face_vts_in_cc = texture_faces[face_components == css_id]

        # Flip vertically the connect component if the normal is pointing outwards
        face_vts_coords = scaled_coords[face_vts_in_cc]
        v0, v1 = (
            face_vts_coords[:, 1, :] - face_vts_coords[:, 0, :],
            face_vts_coords[:, 2, :] - face_vts_coords[:, 0, :],
        )
        n = np.cross(v0, v1).sum()
        if n < 0:
            min_x, max_x = (np.min(face_vts_coords[:, :, 0]), np.max(face_vts_coords[:, :, 0]))
            face_vts_coords[:, :, 0] = min_x + max_x - face_vts_coords[:, :, 0]
            scaled_coords[face_vts_in_cc] = face_vts_coords

        # Panel "colour" = majority-vote label over the CC's 3D vertices.
        seg_vertex_id = np.array([uv_vertex_mapping[u] for u in np.unique(face_vts_in_cc)])
        labels, counts = np.unique(panel_label[seg_vertex_id], return_counts=True)
        label = labels[np.argmax(counts)]

        bound_verts = igl.boundary_loop(face_vts_in_cc)
        bound_3d_verts = [uv_vertex_mapping[uv_id] for uv_id in bound_verts]
        # Integer boundary vertices: used both to draw the edges and as the
        # fillPoly polygon, so the fill and the drawn boundary share corners.
        bound_vert_pos = np.ceil(scaled_coords[bound_verts]).astype(int)
        segs = seg[bound_3d_verts]
        is_multi_segs = is_multi_seg[bound_3d_verts]

        ############# Draw boundary edges (label + stitching tag) #############
        for v1 in range(len(bound_vert_pos)):
            v2 = (v1 + 1) % len(bound_vert_pos)
            seg1 = segs[v1]
            seg2 = segs[v2]
            is_multi_seg1 = is_multi_segs[v1]

            x1, y1 = bound_vert_pos[v1]
            x2, y2 = bound_vert_pos[v2]

            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx - dy
            while True:
                img_array[y1, x1, :3] = label
                if not img_array[y1, x1, 3] > _NON_STITCHED_BOUNDARY:
                    if seg1 == _NON_STITCHED_BOUNDARY or seg2 == _NON_STITCHED_BOUNDARY:
                        img_array[y1, x1, 3] = _NON_STITCHED_BOUNDARY
                    elif seg1 == seg2:
                        img_array[y1, x1, 3] = seg1
                    else:
                        img_array[y1, x1, 3] = seg2 if is_multi_seg1 else seg1
                if x1 == x2 and y1 == y2:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x1 += sx
                if e2 < dx:
                    err += dx
                    y1 += sy

        ############# Fill the panel interior from the stored boundary verts ###
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [bound_vert_pos.astype(np.int32)], 1, lineType=cv2.LINE_8)
        img_array[mask.astype(bool), :3] = label

    return img_array


def create_condition_from_ply_texture(
    ply_file_path: Union[str, Path],
    texture_file_path: Union[str, Path],
    segmentation_file_path: Union[str, Path],
    gim_path: Optional[Union[str, Path]] = None,
    max_dim: int = 300,
    island_padding_percent: int = 2,
    plot: bool = False,
    **kwargs,
):
    """Panel-segmentation + stitching raster from a ply / texture / seg triple.

    Reuses :func:`create_gim_from_ply_texture`'s packing and rasterization, but
    each pixel is filled with the panel's *normalized* label (``label /
    n_panels``) instead of the interpolated 3D vertex coordinate, while the
    stitching tag is kept on the alpha channel. Returns an ``(H, W, 4)`` image:
    channels 0-2 = normalized panel label (its non-zero region is the occupancy
    mask), channel 3 = stitching map.
    """
    uv_coords, _, texture_faces, vertex_faces, _ = read_mesh(ply_file_path)

    # Load texture image size
    width, height = imagesize.get(texture_file_path)
    uv_coords = uv_coords * [width, height]

    with open(GLOBAL_CONFIG.config_dir / "panel_names.txt", "r") as f:
        seg_names = {name.strip(): i + 1 for i, name in enumerate(f.readlines())}
    n_panels = len(seg_names)

    raw_seg = read_segmentation_file(segmentation_file_path)
    # Panel "colour": normalized panel-name label per 3D vertex.
    panel_label = np.array([seg_names.get(s, 0) for s in raw_seg]) / n_panels
    # Stitching tag per 3D vertex (same encoding as create_gim_from_ply_texture).
    is_multi_seg = np.array(
        [True if s.startswith("stitch") and "," in s else False for s in raw_seg]
    )
    stitch_seg = np.array(
        [
            (
                int(s.split(",")[0].split("_")[-1]) + _NON_STITCHED_BOUNDARY + 1
                if s.startswith("stitch")
                else _NON_STITCHED_BOUNDARY
            )
            for s in raw_seg
        ]
    )

    assert 0 < island_padding_percent <= 100, "Invalid padding percentage"
    _max_ori_uv_size = max(uv_coords.max(axis=0) - uv_coords.min(axis=0))
    padding = _max_ori_uv_size * (island_padding_percent / 100)

    all_uvs, _ = unwarp_UV(uv_coords, texture_faces, padding=padding)
    uv_list, _ = normalize_squared_UVs(all_uvs, padding=padding)

    img_array = interpolate_seg_stitch(
        uv_list,
        panel_label,
        vertex_faces,
        texture_faces,
        (is_multi_seg, stitch_seg),
        **kwargs,
    )

    if plot:
        import matplotlib.pyplot as plt

        plt.imshow(img_array[:, :, 2])
        plt.show()

    if gim_path is not None:
        np.savez_compressed(gim_path, img_array)
    return img_array
