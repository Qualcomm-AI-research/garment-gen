# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Turn a predicted geometry image into a stitched, watertight garment mesh."""

import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch

from .algo_utils import DSU, dtw

logger = logging.getLogger(__name__)

MAX_BUCKET = 160

_STITCH_ID_OFFSET = 2


def meshing_uv_map(occupancy, pixel_index=None, use_4_faces=True):
    """Build up to four triangle sets from occupied pixels using both diagonals."""
    occ = occupancy.astype(bool)

    if pixel_index is None:
        pixel_index = np.arange(occ.size).reshape(occ.shape)

    # 0_
    # |/
    # Determine triangles' vertices
    is_tri_vert = occ & np.roll(occ, shift=-1, axis=0) & np.roll(occ, shift=-1, axis=1)
    verta = pixel_index
    vertb = np.roll(pixel_index, shift=-1, axis=1)
    vertc = np.roll(pixel_index, shift=-1, axis=0)
    face0 = np.stack([verta[is_tri_vert], vertb[is_tri_vert], vertc[is_tri_vert]], axis=1)

    #  /|
    # /_0
    # Determine the second set of triangles' vertices
    is_tri_vert = occ & np.roll(occ, shift=1, axis=0) & np.roll(occ, shift=1, axis=1)
    verta = pixel_index
    vertb = np.roll(pixel_index, shift=1, axis=1)
    vertc = np.roll(pixel_index, shift=1, axis=0)
    face1 = np.stack([verta[is_tri_vert], vertb[is_tri_vert], vertc[is_tri_vert]], axis=1)

    # Face 2
    is_tri_vert = occ & np.roll(occ, shift=1, axis=0) & np.roll(occ, shift=-1, axis=1)
    existing_tri = np.roll(occ, shift=1, axis=0)
    existing_tri = np.roll(existing_tri, shift=-1, axis=1)
    is_tri_vert = is_tri_vert & ~existing_tri

    verta = pixel_index
    vertb = np.roll(pixel_index, shift=1, axis=0)
    vertc = np.roll(pixel_index, shift=-1, axis=1)

    face2 = np.stack([verta[is_tri_vert], vertb[is_tri_vert], vertc[is_tri_vert]], axis=1)

    # Face 3
    is_tri_vert = occ & np.roll(occ, shift=-1, axis=0) & np.roll(occ, shift=1, axis=1)
    existing_tri = np.roll(occ, shift=-1, axis=0)
    existing_tri = np.roll(existing_tri, shift=1, axis=1)
    is_tri_vert = is_tri_vert & ~existing_tri

    verta = pixel_index
    vertb = np.roll(pixel_index, shift=-1, axis=0)
    vertc = np.roll(pixel_index, shift=1, axis=1)

    face3 = np.stack([verta[is_tri_vert], vertb[is_tri_vert], vertc[is_tri_vert]], axis=1)

    # Combine the two sets of faces
    if use_4_faces:
        face = np.concatenate([face0, face1, face2, face3], axis=0)
    else:
        face = np.concatenate(
            [
                face0,
                face1,
            ],
            axis=0,
        )

    return face


def create_buckets(n_buckets):
    """Return ``n_buckets`` evenly spaced ``(start, end)`` intervals over ``[0, 1)``."""
    bucket_st = [i / n_buckets for i in range(n_buckets)]
    bucket_en = [i / n_buckets for i in range(1, n_buckets + 1)]
    buckets = list(zip(bucket_st, bucket_en))
    return np.array(buckets)


def extract_stitching_boundaries(stitching_map, occupancy, buckets):
    """Extract boundary coordinates, stitch bucket IDs, and panel IDs."""
    assert stitching_map.shape == occupancy.shape, (
        "Stitching map and occupancy must have the same shape.\n"
        f"Got stitching_map.shape: {stitching_map.shape} and "
        f"occupancy.shape: {occupancy.shape}"
    )
    contours, _ = cv2.findContours(
        occupancy.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    list_boundaries = []
    gim_panel_ids = []
    stitching_values = []
    for panel_id, raw_contour in enumerate(contours):
        contour = raw_contour.squeeze()
        # swap x and y due to cv2
        contour[:, [0, 1]] = contour[:, [1, 0]]

        # Rotate contour until finding a head of a stitch (contour is a closed loop)
        contour_stitching = stitching_map[contour[:, 0], contour[:, 1]]
        shift_contour_stitching = np.roll(contour_stitching, shift=1)

        i = np.where(contour_stitching != shift_contour_stitching)[0][0]
        contour = np.roll(contour, shift=-i, axis=0)

        # Derive boundaries from the contours
        contour_stitching = np.roll(contour_stitching, shift=-i)
        shift_contour_stitching = np.roll(contour_stitching, shift=1)
        starts = np.where(contour_stitching != shift_contour_stitching)[0].tolist()
        starts.append(len(contour_stitching))
        for start, end in zip(starts[:-1], starts[1:]):
            boundary = np.concatenate([contour[start:end], [contour[end % len(contour_stitching)]]])

            list_boundaries.append(boundary)
            gim_panel_ids.append([panel_id] * len(boundary))
            stitching_values.append([stitching_map[boundary[0, 0], boundary[0, 1]]] * len(boundary))
    boundaries = np.concatenate(list_boundaries, axis=0)
    gim_panel_ids = np.concatenate(gim_panel_ids, axis=0)
    stitching_values = np.concatenate(stitching_values, axis=0)

    # Tuple of (array x, array y), each array is (n,)
    uv_coords = (boundaries[:, 0], boundaries[:, 1])

    stitching_values_expanded = stitching_values[:, np.newaxis]
    mask = (buckets[:, 0] <= stitching_values_expanded) & (
        stitching_values_expanded < buckets[:, 1]
    )
    # (n,)
    stitching_ids = np.where(mask)[1]

    return uv_coords, stitching_ids, gim_panel_ids


def _resample_polyline(coords: np.ndarray, num_samples: int) -> np.ndarray:
    """Resample a polyline to ``num_samples`` points evenly spaced by arc length.

    The two stitched boundaries are rasterized independently, so they generally
    have different point counts and non-uniform spacing along the seam.
    Re-parametrizing both onto a common normalized arc-length grid makes point
    ``k`` of one boundary correspond to the same fractional position along the
    other, which is what the direction test in
    :func:`_align_boundaries_geometric` relies on. Degenerate inputs (a single
    point or a zero-length curve) collapse to a repeated point.
    """
    coords = np.asarray(coords, dtype=float)
    num_samples = max(int(num_samples), 2)
    if len(coords) < 2:
        return np.repeat(coords[:1], num_samples, axis=0)

    seg_len = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    arclen = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = arclen[-1]
    if total == 0:
        return np.repeat(coords[:1], num_samples, axis=0)

    targets = np.linspace(0.0, total, num_samples)
    resampled = np.stack(
        [np.interp(targets, arclen, coords[:, d]) for d in range(coords.shape[1])],
        axis=1,
    )
    return resampled


def _align_boundaries_geometric(panel1_3d_coords, panel2_3d_coords, panel2_pixel_ids):
    """Orient ``panel2`` to match ``panel1`` using predicted 3D geometry only.

    Fallback used when no JSON stitch direction is available (see
    :func:`_align_boundaries`). The two boundaries are meant to be sewn
    together, so the correct orientation is the one whose point-to-point 3D
    distance is smaller; the opposite orientation pairs each point with the far
    end of the other boundary and inflates the distance.

    Only the direction *decision* uses the resampled curves; the returned arrays
    are the original ``panel2`` points/pixel ids (optionally reversed), so the
    downstream ``dtw`` matching still operates on the real boundary vertices.
    """
    num_samples = max(len(panel1_3d_coords), len(panel2_3d_coords))
    resampled1 = _resample_polyline(panel1_3d_coords, num_samples)
    resampled2 = _resample_polyline(panel2_3d_coords, num_samples)

    forward_dist = np.linalg.norm(resampled1 - resampled2, axis=1).sum()
    reverse_dist = np.linalg.norm(resampled1 - resampled2[::-1], axis=1).sum()

    if reverse_dist < forward_dist:
        panel2_3d_coords = panel2_3d_coords[::-1]
        panel2_pixel_ids = panel2_pixel_ids[::-1]

    return panel2_3d_coords, panel2_pixel_ids


def _similarity_transfer(points, src_cloud, dst_cloud):
    """Map ``points`` from the ``src_cloud`` frame into the ``dst_cloud`` frame.

    Uses only a translation + uniform scale (centroids and RMS radii matched).
    The GarmentSewer training normalization maps mesh coordinates by a single
    per-axis-uniform affine (``(v + max_dim) / (2 * max_dim)``), so the predicted
    GIM shares the *orientation* of the design pattern and no rotation is needed
    here. This alignment only has to be coarse enough to tell the two ends of a
    seam apart, which are separated by ~the seam length.
    """
    points = np.asarray(points, dtype=float)
    src_cloud = np.asarray(src_cloud, dtype=float)
    dst_cloud = np.asarray(dst_cloud, dtype=float)
    src_center = src_cloud.mean(axis=0)
    dst_center = dst_cloud.mean(axis=0)
    src_scale = np.sqrt(((src_cloud - src_center) ** 2).sum(axis=1).mean())
    dst_scale = np.sqrt(((dst_cloud - dst_center) ** 2).sum(axis=1).mean())
    if src_scale < 1e-12:
        return points - src_center + dst_center
    return (points - src_center) * (dst_scale / src_scale) + dst_center


def _seam_end_sign(boundary_3d_coords, seam_end0, seam_end1):
    """+1 if the boundary runs ``seam_end0 -> seam_end1``, -1 if reversed.

    Decided from the boundary's two endpoints only (robust to interior
    sampling), by whichever assignment of (start, end) to (seam_end0, seam_end1)
    has the smaller total distance.
    """
    start, end = boundary_3d_coords[0], boundary_3d_coords[-1]
    forward = np.linalg.norm(start - seam_end0) + np.linalg.norm(end - seam_end1)
    reverse = np.linalg.norm(start - seam_end1) + np.linalg.norm(end - seam_end0)
    return 1 if forward <= reverse else -1


def _align_boundaries(panel1_3d_coords, panel2_3d_coords, panel2_pixel_ids, stitch_dir=None):
    """Orient ``panel2`` relative to ``panel1`` using the JSON pattern direction.

    ``stitch_dir`` is ``(ref_a, ref_b)`` for this seam: each is a ``(2, 3)`` array of
    stitched-edge endpoints in the design 3D frame, ordered so ``ref_a[k]`` is sewn to
    ``ref_b[k]`` (the JSON ``swap`` flag has already been applied; see
    :func:`_build_stitch_directions`). The orientation decision compares each predicted
    boundary against the clean design endpoints (transferred via translation + scale) rather
    than against each other, which is fragile for straight/symmetric seams. ``panel2`` is
    flipped when the two boundaries run in opposite seam-parameter directions.

    Falls back to :func:`_align_boundaries_geometric` when ``stitch_dir`` is ``None`` or
    the reference geometry is degenerate.
    """
    if stitch_dir is None:
        return _align_boundaries_geometric(panel1_3d_coords, panel2_3d_coords, panel2_pixel_ids)
    try:
        ref_a, ref_b = stitch_dir
        ref_a = np.asarray(ref_a, dtype=float)
        ref_b = np.asarray(ref_b, dtype=float)
        p1 = np.asarray(panel1_3d_coords, dtype=float)
        p2 = np.asarray(panel2_3d_coords, dtype=float)

        design_pts = np.concatenate([ref_a, ref_b], axis=0)
        pred_pts = np.concatenate([p1, p2], axis=0)
        ref_a_pred = _similarity_transfer(ref_a, design_pts, pred_pts)
        ref_b_pred = _similarity_transfer(ref_b, design_pts, pred_pts)

        # Authoritative seam ends in the predicted frame: ref_a[k] <-> ref_b[k],
        # so each pair of matched endpoints defines one physical end of the seam.
        seam_end0 = (ref_a_pred[0] + ref_b_pred[0]) / 2.0
        seam_end1 = (ref_a_pred[1] + ref_b_pred[1]) / 2.0

        # Degenerate reference (the two seam ends collapse to ~one point) cannot
        # resolve a direction -- defer to geometry.
        pred_scale = np.sqrt(((pred_pts - pred_pts.mean(axis=0)) ** 2).sum(axis=1).mean())
        if np.linalg.norm(seam_end1 - seam_end0) < 1e-3 * max(pred_scale, 1e-12):
            return _align_boundaries_geometric(panel1_3d_coords, panel2_3d_coords, panel2_pixel_ids)

        sign1 = _seam_end_sign(p1, seam_end0, seam_end1)
        sign2 = _seam_end_sign(p2, seam_end0, seam_end1)

        if sign1 != sign2:
            panel2_3d_coords = panel2_3d_coords[::-1]
            panel2_pixel_ids = panel2_pixel_ids[::-1]
        return panel2_3d_coords, panel2_pixel_ids
    except Exception:
        logger.debug("Design-guided boundary alignment failed; using geometry", exc_info=True)
        return _align_boundaries_geometric(panel1_3d_coords, panel2_3d_coords, panel2_pixel_ids)


def _edge_endpoints_3d(pattern, panel_name, edge_id):
    """Design-frame 3D positions of an edge's two endpoints, ``[start, end]``.

    ``pattern`` is a loaded :class:`swifttailor.data.patterns.core.BasicPattern`;
    its edge loops are already normalized on load, so the ``endpoints``
    orientation matches the representation the ``swap`` flag is defined against.
    """
    from swifttailor.data.patterns.core import BasicPattern

    panel = pattern.pattern["panels"][panel_name]
    vertices = np.asarray(panel["vertices"], dtype=float)
    v_start, v_end = panel["edges"][edge_id]["endpoints"]
    rotation = panel["rotation"]
    translation = np.asarray(panel["translation"], dtype=float)
    start_3d = BasicPattern._point_in_3D(vertices[v_start], rotation, translation)
    end_3d = BasicPattern._point_in_3D(vertices[v_end], rotation, translation)
    return np.stack([start_3d, end_3d], axis=0)


def _build_stitch_directions(spec_path):
    """Parse a sewing-pattern JSON into per-stitch design-frame directions.

    Returns ``{stitch_index: (ref_a, ref_b)}`` where each ref is a ``(2, 3)``
    array of the stitched edge's endpoints in design 3D, ordered so that
    ``ref_a[k]`` sews to ``ref_b[k]``. The stitch index matches the raster stitch
    id used by :func:`grouping_vertices` (``bucket_id - _STITCH_ID_OFFSET``).

    Direction follows GarmentCode's rule (pygarment ``BoxMesh.read_stitches`` /
    ``_swap_stitch_ranges``): the default manifold ``swap`` connects the *end* of
    side-0's edge to the *start* of side-1's edge; a ``"right_wrong"`` tag as a
    third stitch element disables the swap. Returns ``{}`` (geometry-only
    fallback) if the pattern cannot be parsed.
    """
    try:
        from swifttailor.data.patterns.core import BasicPattern

        pattern = BasicPattern(str(spec_path))
    except Exception:
        logger.debug("Could not parse stitch directions from %s", spec_path, exc_info=True)
        return {}

    stitches = pattern.pattern.get("stitches", []) or []
    directions = {}
    for idx, stitch_spec in enumerate(stitches):
        try:
            side0, side1 = stitch_spec[0], stitch_spec[1]
            edge_a = _edge_endpoints_3d(pattern, side0["panel"], side0["edge"])
            edge_b = _edge_endpoints_3d(pattern, side1["panel"], side1["edge"])
            swap = not (len(stitch_spec) == 3 and stitch_spec[-1] == "right_wrong")
            # swap connects side0.end <-> side1.start, so reverse edge_b to make
            # edge_a[k] <-> edge_b[k] hold index-for-index.
            ref_b = edge_b[::-1] if swap else edge_b
            directions[idx] = (edge_a, ref_b)
        except Exception:
            logger.debug("Skipping invalid stitch specification %d", idx, exc_info=True)
            continue
    return directions


def _extract_coords_and_index(gim, x_, y_, point_ids, pixel_index):
    """Return the 3D coords and pixel ids of the given boundary points."""
    x = x_[point_ids]
    y = y_[point_ids]
    gim_coords = gim[x, y, :3]
    pixel_ids = pixel_index[x, y]
    return gim_coords, pixel_ids


def sewing_panels(gim, occupancy, stitching_map, max_bucket=MAX_BUCKET, stitch_directions=None):
    """Stitch panel boundaries by grouping, unioning, and merging matched vertices.

    Returns:
        The updated geometry image and its per-pixel vertex index.
    """
    if stitching_map.max() > 1:
        stitching_map /= max_bucket
    buckets = create_buckets(max_bucket)
    uv_coords, stitching_ids, gim_panel_ids = extract_stitching_boundaries(
        stitching_map, occupancy, buckets
    )

    pixel_index = np.arange(occupancy.size).reshape(occupancy.shape)

    # Step 1: Grouping vertices
    start_time = time.perf_counter()
    all_pairs = grouping_vertices(
        gim, pixel_index, stitching_ids, uv_coords, gim_panel_ids, max_bucket, stitch_directions
    )
    end_time = time.perf_counter()
    logger.debug("grouping_vertices completed in %.3f seconds", end_time - start_time)

    # Step 2: Union disjoint sets
    start_time = time.perf_counter()
    max_pixel_id = occupancy.shape[0] * occupancy.shape[1]
    vertex_mapping = union_vertices(all_pairs, max_pixel_id)
    end_time = time.perf_counter()
    logger.debug("union_vertices completed in %.3f seconds", end_time - start_time)

    # Step 3: Merge vertices & update gim
    start_time = time.perf_counter()
    gim, pixel_index = merge_vertices(gim, vertex_mapping, pixel_index)
    end_time = time.perf_counter()
    logger.debug("merge_vertices completed in %.3f seconds", end_time - start_time)

    return gim, pixel_index


def merge_vertices(gim, vertex_mapping, pixel_index):
    """Collapse merged vertex groups to their averaged 3D position in the GIM.

    Returns:
        The updated geometry image and pixel index.
    """
    reversed_vertex_mapping = {}
    for k, v in vertex_mapping.items():
        if v not in reversed_vertex_mapping:
            reversed_vertex_mapping[v] = [k]
        else:
            reversed_vertex_mapping[v].append(k)

    merged_vertices = {}
    for k, v in reversed_vertex_mapping.items():
        ids = np.array(v)
        coords = (ids // pixel_index.shape[1], ids % pixel_index.shape[1])
        merged_vertices[k] = np.mean(gim[coords[0], coords[1]], axis=0)
        gim[coords[0], coords[1]] = merged_vertices[k]
        pixel_index[coords[0], coords[1]] = k

    return gim, pixel_index


def union_vertices(all_pairs, max_pixel_id):
    """Union stitched vertex pairs into groups, assigning each a fresh merged id.

    Returns:
        Mapping from each original pixel id to its group representative id.
    """
    dsu = DSU()
    new_id = max_pixel_id
    for pair in all_pairs:
        dsu.union(pair[0], pair[1])
    mapping = {}
    ids = list(dsu.parent.keys())
    for child in ids:
        parent = dsu.find(child)
        if parent < max_pixel_id:
            dsu.force_parent(parent, new_id)
            new_id += 1
        parent = dsu.find(child)

        mapping[child] = parent

    return mapping


def grouping_vertices(
    gim, pixel_index, stitching_ids, uv_coords, gim_panel_ids, max_bucket, stitch_directions=None
):
    """Match boundary vertices across the two panels of each stitch bucket.

    Returns:
        Vertex-id pairs to be unioned, aligned via ``stitch_directions`` when given.
    """
    all_pairs = []
    for bucket_id in range(2, max_bucket):
        pos = np.where(stitching_ids == bucket_id)
        if len(pos[0]) == 0:
            continue
        x_, y_ = uv_coords[0][pos[0]], uv_coords[1][pos[0]]
        component_ids = gim_panel_ids[pos[0]].astype(np.int32)
        if len(np.unique(component_ids)) == 1:
            half_len = len(component_ids) // 2
            component_ids[half_len:] = component_ids[half_len:] + 1
        if len(np.unique(component_ids)) != 2:
            # TODO: handle stitch with more than 2 components
            logger.warning(
                "Each stitch should have 2 components; found %d for bucket %d",
                len(np.unique(component_ids)),
                bucket_id,
            )

        # Sort component_ids with bincount
        panel1, panel2 = np.argsort(np.bincount(component_ids))[-2:]
        point_ids_on_panel1 = np.where(component_ids == panel1)[0]
        point_ids_on_panel2 = np.where(component_ids == panel2)[0]

        # There are not enough points to form triangles
        if len(point_ids_on_panel1) < 1 or len(point_ids_on_panel2) < 2:
            continue

        # Extract 3D and pixel index of boundary points on panel1 and panel2
        panel1_3d_coords, panel1_pixel_ids = _extract_coords_and_index(
            gim, x_, y_, point_ids_on_panel1, pixel_index
        )
        panel2_3d_coords, panel2_pixel_ids = _extract_coords_and_index(
            gim, x_, y_, point_ids_on_panel2, pixel_index
        )
        # Align boundaries
        stitch_dir = None
        if stitch_directions is not None:
            stitch_dir = stitch_directions.get(bucket_id - _STITCH_ID_OFFSET)
        panel2_3d_coords, panel2_pixel_ids = _align_boundaries(
            panel1_3d_coords, panel2_3d_coords, panel2_pixel_ids, stitch_dir=stitch_dir
        )
        # Sew the two panels
        pairs = dtw(
            panel1_3d_coords,
            panel2_3d_coords,
            panel1_pixel_ids,
            panel2_pixel_ids,
        )
        all_pairs.extend(pairs)

    return all_pairs


def calc_normal(gim, occupancy):
    """Compute per-vertex normals from the grid mesh (NumPy).

    Returns:
        Unit normals shaped like the geometry image's coordinate channels.
    """
    gim = gim[:, :, :3]
    occ = occupancy.astype(bool)
    normals = np.zeros_like(gim)

    # 0_
    # |/
    # Determine triangles' vertices
    is_tri_vert = occ & np.roll(occ, shift=-1, axis=0) & np.roll(occ, shift=-1, axis=1)
    verta = gim
    vertb = np.roll(gim, shift=-1, axis=1)
    vertc = np.roll(gim, shift=-1, axis=0)
    cur_normals = np.zeros_like(gim)
    cur_normals[is_tri_vert] = np.cross(
        vertb[is_tri_vert] - verta[is_tri_vert], vertc[is_tri_vert] - verta[is_tri_vert]
    )
    cur_normals = cur_normals + np.roll(cur_normals, shift=1, axis=1)
    cur_normals = cur_normals + np.roll(cur_normals, shift=1, axis=0)
    normals = normals + cur_normals

    #  /|
    # /_0
    # Determine the second set of triangles' vertices
    is_tri_vert = occ & np.roll(occ, shift=1, axis=0) & np.roll(occ, shift=1, axis=1)
    verta = gim
    vertb = np.roll(gim, shift=1, axis=1)
    vertc = np.roll(gim, shift=1, axis=0)
    cur_normals = np.zeros_like(gim)
    cur_normals[is_tri_vert] = np.cross(
        vertb[is_tri_vert] - verta[is_tri_vert], vertc[is_tri_vert] - verta[is_tri_vert]
    )
    cur_normals = cur_normals + np.roll(cur_normals, shift=-1, axis=1)
    cur_normals = cur_normals + np.roll(cur_normals, shift=-1, axis=0)
    normals = normals + cur_normals

    # Face 2
    is_tri_vert = occ & np.roll(occ, shift=1, axis=0) & np.roll(occ, shift=-1, axis=1)
    existing_tri = np.roll(occ, shift=1, axis=0)
    existing_tri = np.roll(existing_tri, shift=-1, axis=1)
    is_tri_vert = is_tri_vert & ~existing_tri

    verta = gim
    vertb = np.roll(gim, shift=1, axis=0)
    vertc = np.roll(gim, shift=-1, axis=1)
    cur_normals = np.zeros_like(gim)
    cur_normals[is_tri_vert] = np.cross(
        vertb[is_tri_vert] - verta[is_tri_vert], vertc[is_tri_vert] - verta[is_tri_vert]
    )
    cur_normals = cur_normals + np.roll(cur_normals, shift=1, axis=1)
    cur_normals = cur_normals + np.roll(cur_normals, shift=-1, axis=0)
    normals = normals + cur_normals

    # Face 3
    is_tri_vert = occ & np.roll(occ, shift=-1, axis=0) & np.roll(occ, shift=1, axis=1)
    existing_tri = np.roll(occ, shift=-1, axis=0)
    existing_tri = np.roll(existing_tri, shift=1, axis=1)
    is_tri_vert = is_tri_vert & ~existing_tri

    verta = gim
    vertb = np.roll(gim, shift=-1, axis=0)
    vertc = np.roll(gim, shift=1, axis=1)
    cur_normals = np.zeros_like(gim)
    cur_normals[is_tri_vert] = np.cross(
        vertb[is_tri_vert] - verta[is_tri_vert], vertc[is_tri_vert] - verta[is_tri_vert]
    )
    cur_normals = cur_normals + np.roll(cur_normals, shift=1, axis=0)
    cur_normals = cur_normals + np.roll(cur_normals, shift=-1, axis=1)
    normals = normals + cur_normals

    # Normalize the normals
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    # Replace zero norms with 1 to avoid division by zero
    norm = np.where(norm == 0, 1.0, norm)
    normals = normals / norm
    return normals


def torch_calc_normal(gim: torch.Tensor, occupancy: torch.Tensor):
    """Compute per-vertex normals from the grid mesh (Torch), supporting batches.

    Returns:
        Unit normals, ``[B, 3, H, W]`` for batched input else ``[H, W, 3]``.
    """
    # Handle batched inputs
    if gim.ndim == 4:  # [B, C, H, W]
        gim = gim.permute(0, 2, 3, 1)  # [B, H, W, C]
        batch_normals = []
        for b_gim, b_occ in zip(gim, occupancy):
            b_normal = torch_calc_normal(b_gim, b_occ)  # [H, W, 3]
            batch_normals.append(b_normal)
        normals = torch.stack(batch_normals, dim=0)  # [B, H, W, 3]
        return normals.permute(0, 3, 1, 2)  # [B, 3, H, W]

    # Original single image processing
    gim = gim[:, :, :3]
    occ = occupancy.bool()
    normals = torch.zeros_like(gim)

    # 0_
    # |/
    # Determine triangles' vertices
    is_tri_vert = occ & torch.roll(occ, shifts=-1, dims=0) & torch.roll(occ, shifts=-1, dims=1)
    verta = gim
    vertb = torch.roll(gim, shifts=-1, dims=1)
    vertc = torch.roll(gim, shifts=-1, dims=0)
    cur_normals = torch.zeros_like(gim)
    cur_normals[is_tri_vert] = torch.linalg.cross(
        vertb[is_tri_vert] - verta[is_tri_vert], vertc[is_tri_vert] - verta[is_tri_vert]
    )
    cur_normals = cur_normals + torch.roll(cur_normals, shifts=1, dims=1)
    cur_normals = cur_normals + torch.roll(cur_normals, shifts=1, dims=0)
    normals = normals + cur_normals

    #  /|
    # /_0
    # Determine the second set of triangles' vertices
    is_tri_vert = occ & torch.roll(occ, shifts=1, dims=0) & torch.roll(occ, shifts=1, dims=1)
    verta = gim
    vertb = torch.roll(gim, shifts=1, dims=1)
    vertc = torch.roll(gim, shifts=1, dims=0)
    cur_normals = torch.zeros_like(gim)
    cur_normals[is_tri_vert] = torch.linalg.cross(
        vertb[is_tri_vert] - verta[is_tri_vert], vertc[is_tri_vert] - verta[is_tri_vert]
    )
    cur_normals = cur_normals + torch.roll(cur_normals, shifts=-1, dims=1)
    cur_normals = cur_normals + torch.roll(cur_normals, shifts=-1, dims=0)
    normals = normals + cur_normals

    # Face 2
    is_tri_vert = occ & torch.roll(occ, shifts=1, dims=0) & torch.roll(occ, shifts=-1, dims=1)
    existing_tri = torch.roll(occ, shifts=1, dims=0)
    existing_tri = torch.roll(existing_tri, shifts=-1, dims=1)
    is_tri_vert = is_tri_vert & ~existing_tri

    verta = gim
    vertb = torch.roll(gim, shifts=1, dims=0)
    vertc = torch.roll(gim, shifts=-1, dims=1)
    cur_normals = torch.zeros_like(gim)
    cur_normals[is_tri_vert] = torch.linalg.cross(
        vertb[is_tri_vert] - verta[is_tri_vert], vertc[is_tri_vert] - verta[is_tri_vert]
    )
    cur_normals = cur_normals + torch.roll(cur_normals, shifts=1, dims=1)
    cur_normals = cur_normals + torch.roll(cur_normals, shifts=-1, dims=0)
    normals = normals + cur_normals

    # Face 3
    is_tri_vert = occ & torch.roll(occ, shifts=-1, dims=0) & torch.roll(occ, shifts=1, dims=1)
    existing_tri = torch.roll(occ, shifts=-1, dims=0)
    existing_tri = torch.roll(existing_tri, shifts=1, dims=1)
    is_tri_vert = is_tri_vert & ~existing_tri

    verta = gim
    vertb = torch.roll(gim, shifts=-1, dims=0)
    vertc = torch.roll(gim, shifts=1, dims=1)
    cur_normals = torch.zeros_like(gim)
    cur_normals[is_tri_vert] = torch.linalg.cross(
        vertb[is_tri_vert] - verta[is_tri_vert], vertc[is_tri_vert] - verta[is_tri_vert]
    )
    cur_normals = cur_normals + torch.roll(cur_normals, shifts=1, dims=0)
    cur_normals = cur_normals + torch.roll(cur_normals, shifts=-1, dims=1)
    normals = normals + cur_normals

    # Normalize the normals
    norm = torch.linalg.norm(normals, dim=-1, keepdim=True)
    # Replace zero norms with 1 to avoid division by zero
    norm = torch.where(norm == 0, torch.ones_like(norm), norm)
    normals = normals / norm
    return normals


def _clean_occupancy(occupancy):
    """Clean the occupancy map by removing isolated pixels."""
    # TODO: speed up this

    face = meshing_uv_map(occupancy)
    updated_occupancy = np.zeros_like(occupancy)

    uv_coords = np.unique(face.flatten())
    uv_coords = np.array([uv_coords // occupancy.shape[1], uv_coords % occupancy.shape[1]])
    updated_occupancy[uv_coords[0], uv_coords[1]] = 1

    return updated_occupancy


def _reindex_faces(faces):
    """Reindex faces to start from 1 and be consecutive."""
    idx = np.unique(faces.flatten())
    new_idx = {old: new + 1 for new, old in enumerate(idx)}
    new_faces = np.vectorize(new_idx.get)(faces)
    return new_faces


def _extract_vertices_from_gim(
    gim: np.ndarray, occupancy: np.ndarray, vertex_index: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Gather unique occupied-pixel vertex ids and their coordinates from the GIM.

    Returns:
        Sorted vertex ids and their matching 3D coordinates.
    """
    flatten_vertex_ids = vertex_index[np.where(occupancy)]
    flatten_vertex_coords = gim[np.where(occupancy)]

    sorted_vertex_ids, sorted_order = np.unique(flatten_vertex_ids, return_index=True)
    sorted_vertex_coords = flatten_vertex_coords[sorted_order]

    return (sorted_vertex_ids, sorted_vertex_coords)


def _clean_faces(faces: np.ndarray, texture_faces: Optional[np.ndarray] = None):
    """Clean faces by removing invalid faces."""
    valid_face = (
        (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    )

    if texture_faces is not None:
        return faces[valid_face.squeeze()], texture_faces[valid_face.squeeze()]
    return faces[valid_face.squeeze()], None


def create_obj_from_gim(
    gim: np.ndarray,
    stitching_map: Optional[np.ndarray] = None,
    occupancy: Optional[np.ndarray] = None,
    saving_path: Optional[Union[str, Path]] = None,
    max_bucket: int = MAX_BUCKET,
    use_texture: bool = False,
    spec_path: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct a garment mesh from a geometry image, optionally stitching and saving.

    Occupancy defaults to non-background pixels when omitted; ``stitching_map``
    and ``spec_path`` enable seam stitching.

    Returns:
        The mesh vertices and faces (plus texture faces when ``use_texture``).
    """
    assert len(gim.shape) == 3, "GIM must be a 3D array"
    assert gim.shape[-1] == 3, "GIM must have 3 channels"

    if occupancy is None:
        logger.warning("Occupancy not provided; assuming a background value of 0")
        occupancy = np.any(~np.isclose(gim, b=0, atol=1e-6), axis=-1)
    # Preprocessing
    occupancy = _clean_occupancy(occupancy)

    # Per-stitch seam directions from the JSON pattern (empty dict on any issue,
    # which makes _align_boundaries fall back to geometry).
    stitch_directions = _build_stitch_directions(spec_path) if spec_path is not None else None

    # Stitching
    backup_gim = deepcopy(gim)
    try:
        gim, vertex_index = sewing_panels(
            gim,
            occupancy,
            stitching_map,
            max_bucket=max_bucket,
            stitch_directions=stitch_directions,
        )
    except Exception:
        logger.warning("Panel sewing failed; returning the unstitched mesh", exc_info=True)
        gim = backup_gim
        vertex_index = np.arange(occupancy.size).reshape(occupancy.shape)

    # Remeshing
    faces = meshing_uv_map(occupancy, pixel_index=vertex_index, use_4_faces=True)

    vertex_ids, vertices = _extract_vertices_from_gim(gim, occupancy, vertex_index)

    # Sanity check
    sorted_face_vertex_ids = np.unique(faces.flatten())
    assert np.array_equal(vertex_ids, sorted_face_vertex_ids), (
        "Vertex ids extracted from gim and faces do not match.\n"
        f"Got vertex_ids: {vertex_ids}\n"
        f"Got sorted_face_vertex_ids: {sorted_face_vertex_ids}"
    )
    if len(vertices) != occupancy.sum():
        logger.warning(
            "Vertex count %d differs from occupied-pixel count %d",
            len(vertices),
            occupancy.sum(),
        )

    # Reindex faces to start from 1 and be consecutive
    reindex_faces = _reindex_faces(faces)

    # Texture
    reindex_texture_faces = None
    if use_texture:
        texture_index = np.arange(occupancy.size).reshape(occupancy.shape)
        texture_x = texture_index // occupancy.shape[0]
        texture_y = texture_index % occupancy.shape[0]
        texture_gim = np.stack([texture_x, texture_y], axis=-1)
        texture_faces = meshing_uv_map(occupancy, pixel_index=texture_index, use_4_faces=True)
        _, texture_coords = _extract_vertices_from_gim(texture_gim, occupancy, texture_index)
        reindex_texture_faces = _reindex_faces(texture_faces)

    # Clean invalid faces
    reindex_faces, reindex_texture_faces = _clean_faces(reindex_faces, reindex_texture_faces)

    vertices = vertices * 400 - 200
    if saving_path is not None:
        if use_texture:
            create_obj(
                vertices,
                reindex_faces,
                texture_faces=reindex_texture_faces,
                texture_coords=texture_coords,
                saving_path=saving_path,
            )
        else:
            create_obj(vertices, reindex_faces, saving_path=saving_path)

    if use_texture:
        return vertices, reindex_faces, reindex_texture_faces
    else:
        return vertices, reindex_faces


def create_obj(
    vertices: np.ndarray,
    faces: np.ndarray,
    texture_faces: Optional[np.ndarray] = None,
    texture_coords: Optional[np.ndarray] = None,
    saving_path: Union[str, Path] = "output.obj",
) -> None:
    """Write vertices and faces (with optional UV texture faces) to an OBJ file."""
    with open(saving_path, "w") as f:
        for vert in vertices:
            f.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")

        if texture_faces is None:
            for face in faces:
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")
        else:
            assert texture_coords is not None, "Texture coords should be provided to export OBJ."
            for coord in texture_coords:
                f.write(f"vt {coord[0]/512} {coord[1]/512}\n")

            assert (
                faces.shape == texture_faces.shape
            ), "Faces and Texture Faces should be in same shape."
            for face, texture_face in zip(faces, texture_faces):
                f.write(
                    f"f {face[0]}/{texture_face[0]} {face[1]}/{texture_face[1]} {face[2]}/{texture_face[2]}\n"
                )
