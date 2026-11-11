# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Generic algorithms used by remeshing: dynamic time warping and a union-find (DSU)."""

from collections import deque

import numpy as np


# DWT implementation
def dtw(A, B, value_A, value_B):
    """Align two 3D point sequences via dynamic time warping.

    Returns:
        Paired ``(value_A, value_B)`` items along the optimal warp path.
    """
    assert len(A) == len(value_A) and len(B) == len(value_B), (
        "A and value_A must have the same length. "
        f"Got {len(A)} and {len(value_A)}. "
        f"B and value_B must have the same length. "
        f"Got {len(B)} and {len(value_B)}"
    )

    assert len(A) <= len(B), (
        "Len(A) must be less than or equal to Len(B). " f"Got {len(A)} and {len(B)}"
    )

    A = np.asarray(A, float)
    B = np.asarray(B, float)
    assert A.shape[1] == 3 and B.shape[1] == 3
    n, m = len(A), len(B)

    # Cost DP
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    back = np.zeros((n + 1, m + 1, 2), dtype=int)
    back_choices = np.array([(-1, 0), (0, -1), (-1, -1)])

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(A[i - 1] - B[j - 1])
            choices = (D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
            argmin = np.argmin(choices)
            D[i, j] = cost + choices[argmin]
            back[i, j] = np.array([i, j]) + back_choices[argmin]

    # Traceback
    i, j = n, m
    pairs = []
    while i > 0 and j > 0:
        pairs.append((value_A[i - 1], value_B[j - 1]))
        i, j = back[i, j]

    return pairs


# DSU implementation
class DSU:
    """Disjoint-set union (union-find) with path compression and union by rank."""

    def __init__(self):
        """Initialize empty parent and rank maps."""
        self.parent = {}
        self.rank = {}

    def make_set(self, x):
        """Create a singleton set for ``x`` if not already present."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        """Return the representative of ``x``'s set, compressing the path."""
        self.make_set(x)
        stack = deque([x])
        while True:
            x = stack[-1]
            if self.parent[x] != x:
                stack.append(self.parent[x])
            else:
                break

        parent = stack.pop()
        while stack:
            self.parent[stack.pop()] = parent

        return parent

    def union(self, x, y):
        """Merge the sets containing ``x`` and ``y`` by rank."""
        x = self.find(x)
        y = self.find(y)

        if x != y:
            if self.rank[x] < self.rank[y]:
                x, y = y, x
            self.parent[y] = x
            if self.rank[x] == self.rank[y]:
                self.rank[x] += 1

    def force_parent(self, child, parent):
        """Attach ``child``'s set under a new, previously unseen ``parent`` node."""
        assert parent not in self.parent, (
            f"Parent {parent} is already in the DSU. " f"Got {self.parent}."
        )
        child_parent = self.find(child)
        # Add new vertex as parent
        self.parent[child_parent] = parent
        self.parent[parent] = parent

        # Update rank for new vertex
        self.rank[parent] = self.rank[child_parent] + 1
