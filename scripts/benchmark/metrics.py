# Copyright (c) 2019 Guandao Yang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Changes from Qualcomm Technologies, Inc. are provided under the following license:
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# Based on: https://github.com/stevenygd/PointFlow/blob/master/metrics/evaluation_metrics.py

import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import trimesh
from numpy.linalg import norm
from pytorch3d.loss import chamfer_distance
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm


def distChamfer(a, b):
    """
    Compute Chamfer Distance between two point clouds using PyTorch3D,
    returning per-point squared distances in both directions (a→b, b→a).

    Args:
        a (torch.Tensor): (B, N, D) tensor of points.
        b (torch.Tensor): (B, M, D) tensor of points.

    Returns:
        dists1 (torch.Tensor): (B, N) squared distances from a to b.
        dists2 (torch.Tensor): (B, M) squared distances from b to a.
    """
    a = a.to("cuda")
    b = b.to("cuda")

    # Compute Chamfer Distance and get per-point distances
    (dists_a_to_b, dists_b_to_a), _ = chamfer_distance(
        a, b, batch_reduction=None, point_reduction=None, norm=2, single_directional=False
    )

    return dists_a_to_b, dists_b_to_a


def _pairwise_pc_dist(
    sample_pcs: torch.Tensor, ref_pcs: torch.Tensor, batch_size: int, verbose: bool = True
):
    assert sample_pcs.ndim == 3 and ref_pcs.ndim == 3, (
        "`sample_pcs` and `ref_pcs` must be 3D tensors [N, P, 3]. "
        f"Got {sample_pcs.shape} and {ref_pcs.shape}."
    )

    assert sample_pcs.shape[0] == ref_pcs.shape[0], (
        "`sample_pcs` and `ref_pcs` must have the same number of samples. "
        f"Got {sample_pcs.shape[0]} and {ref_pcs.shape[0]}."
    )

    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    all_cd = []
    iterator = range(N_sample)
    if verbose:
        iterator = tqdm(iterator, desc="Pairwise CD")
    for sample_b_start in iterator:
        sample_batch = sample_pcs[sample_b_start]

        cd_lst = []
        sub_iterator = range(0, N_ref, batch_size)
        # if verbose:
        #     sub_iterator = tqdm(sub_iterator, leave=False)
        for ref_b_start in sub_iterator:
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]

            batch_size_ref = ref_batch.size(0)
            point_dim = ref_batch.size(2)
            sample_batch_exp = sample_batch.view(1, -1, point_dim).expand(batch_size_ref, -1, -1)
            sample_batch_exp = sample_batch_exp.contiguous()

            dl, dr = distChamfer(sample_batch_exp, ref_batch)
            cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1))

        cd_lst = torch.cat(tensors=cd_lst, dim=1)
        all_cd.append(cd_lst)

    all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref

    return all_cd


def lgan_mmd_cov(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    _, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        "lgan_mmd": mmd,
        "lgan_cov": cov,
    }


def compute_all_pc_metrics(sample_pcs, ref_pcs, batch_size, cache_dir=None):
    results = {}

    print("Pairwise CD")
    M_rs_cd = _pairwise_pc_dist(ref_pcs, sample_pcs, batch_size)

    # CD
    res_cd = lgan_mmd_cov(M_rs_cd.t())
    results.update({"%s-CD" % k: v for k, v in res_cd.items()})

    for k, v in results.items():
        print("[%s] %.8f" % (k, v.item()))

    return results


SAMPLING_SEED = 42


def _load_and_sample_surfaces(path, n_sampling_points: int, seed=SAMPLING_SEED) -> np.ndarray:
    # Load the GIM and sample points
    if path is None:
        return np.zeros((n_sampling_points, 3), dtype=np.float32)
    surface = trimesh.load(path)
    points, _ = trimesh.sample.sample_surface(surface, n_sampling_points, seed=seed)
    sampling_pc: np.ndarray = np.asarray(points)

    return sampling_pc.astype(np.float32)


def _compute_metrics(
    gt_paths: List[Path],
    pred_paths: List[Path],
    n_sampling_points: int = 25_000,
    device="cpu",
):
    assert len(gt_paths) == len(
        pred_paths
    ), "Number of groundtruth paths should match that of prediction paths."

    with ProcessPoolExecutor(max_workers=32) as executor:
        gts = np.array(
            list(
                tqdm(
                    executor.map(
                        _load_and_sample_surfaces,
                        gt_paths,
                        [n_sampling_points] * len(gt_paths),
                    ),
                    total=len(gt_paths),
                    desc="Loading and sampling target",
                )
            )
        )

    with ProcessPoolExecutor(max_workers=32) as executor:
        preds = np.array(
            list(
                tqdm(
                    executor.map(
                        _load_and_sample_surfaces,
                        pred_paths,
                        [n_sampling_points] * len(pred_paths),
                    ),
                    total=len(pred_paths),
                    desc="Loading and sampling reconstruction",
                )
            )
        )

    torch_gts = torch.from_numpy(gts).to(device)
    torch_preds = torch.from_numpy(preds).to(device)

    logs = compute_all_pc_metrics(
        torch_preds,
        torch_gts,
        batch_size=128,
    )
    for key, value in logs.items():
        logs[key] = value.item()

    return logs
