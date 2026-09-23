"""Connected-component labeling for change masks.

Pure-numpy union-find labeling (no scipy dependency), 4- or 8-connectivity.
Each region is reported with pixel count, bounding box, and centroid so
callers can filter, rank, and vectorize change patches without extra passes.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}

    def find(self, x: int) -> int:
        parent = self.parent
        if x not in parent:
            parent[x] = x
            return x
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def label_regions(mask: np.ndarray, connectivity: int = 8) -> Tuple[np.ndarray, List[Dict]]:
    """Label connected components of a boolean mask.

    Returns ``(labels, regions)`` where ``labels`` is an int32 grid (0 =
    background, 1..N = region ids) and ``regions`` is a list of dicts with
    keys ``label``, ``pixels``, ``bbox`` (row_min, row_max, col_min,
    col_max, inclusive), and ``centroid`` (row, col as floats).
    """
    mask = np.asarray(mask, dtype=bool)
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    rows, cols = mask.shape
    labels = np.zeros((rows, cols), dtype=np.int32)
    uf = _UnionFind()
    nxt = 1

    if connectivity == 8:
        neighbors = ((-1, -1), (-1, 0), (-1, 1), (0, -1))
    else:
        neighbors = ((-1, 0), (0, -1))

    # First pass: provisional labels + equivalences.
    for r in range(rows):
        mrow = mask[r]
        lrow = labels[r]
        for c in range(cols):
            if not mrow[c]:
                continue
            seen = set()
            for dr, dc in neighbors:
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols and labels[rr, cc]:
                    seen.add(uf.find(int(labels[rr, cc])))
            if not seen:
                lrow[c] = nxt
                uf.find(nxt)
                nxt += 1
            else:
                lab = min(seen)
                lrow[c] = lab
                for other in seen:
                    uf.union(lab, other)

    # Second pass: resolve to canonical labels, compact to 1..N.
    remap: Dict[int, int] = {}
    regions_acc: Dict[int, Dict] = {}
    for r in range(rows):
        lrow = labels[r]
        mrow = mask[r]
        for c in range(cols):
            if not mrow[c]:
                continue
            root = uf.find(int(lrow[c]))
            if root not in remap:
                remap[root] = len(remap) + 1
                regions_acc[remap[root]] = {
                    "label": remap[root],
                    "pixels": 0,
                    "row_min": r, "row_max": r,
                    "col_min": c, "col_max": c,
                    "row_sum": 0.0, "col_sum": 0.0,
                }
            lab = remap[root]
            lrow[c] = lab
            acc = regions_acc[lab]
            acc["pixels"] += 1
            acc["row_sum"] += r
            acc["col_sum"] += c
            if r < acc["row_min"]:
                acc["row_min"] = r
            if r > acc["row_max"]:
                acc["row_max"] = r
            if c < acc["col_min"]:
                acc["col_min"] = c
            if c > acc["col_max"]:
                acc["col_max"] = c

    regions: List[Dict] = []
    for lab in sorted(regions_acc):
        acc = regions_acc[lab]
        n = acc["pixels"]
        regions.append({
            "label": lab,
            "pixels": n,
            "bbox": (acc["row_min"], acc["row_max"], acc["col_min"], acc["col_max"]),
            "centroid": (acc["row_sum"] / n, acc["col_sum"] / n),
        })
    return labels, regions


def filter_regions(
    labels: np.ndarray,
    regions: List[Dict],
    min_pixels: int = 1,
    max_pixels: int | None = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """Keep regions within a pixel-count range; relabel the survivors 1..N."""
    keep = {r["label"] for r in regions
            if r["pixels"] >= min_pixels and (max_pixels is None or r["pixels"] <= max_pixels)}
    if not keep:
        return np.zeros_like(labels), []
    out = np.zeros_like(labels)
    new_regions: List[Dict] = []
    for new_id, old in enumerate(sorted(keep), start=1):
        out[labels == old] = new_id
        for r in regions:
            if r["label"] == old:
                nr = dict(r)
                nr["label"] = new_id
                new_regions.append(nr)
                break
    return out, new_regions
