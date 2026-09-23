"""Post-classification comparison of two label grids.

Where per-pixel classifiers (or hand-digitized maps) exist for both dates,
comparing the label grids directly gives a *from-to* change matrix: not
just "what changed" but "what it changed from and to". This is the right
tool for land-cover conversion questions ("how much forest became
developed?") that pure spectral differencing cannot answer.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import ChangeError, Grid


def post_classification_compare(
    labels_t1: Grid,
    labels_t2: Grid,
    class_names: Optional[Dict[int, str]] = None,
    check_crs: bool = True,
) -> Dict:
    """Compare two integer label grids.

    Returns a dict with:
    * ``matrix`` — nested dict ``{from: {to: pixel_count}}`` (sparse: only
      nonzero transitions stored),
    * ``labels`` — sorted list of class ids seen in either grid,
    * ``changed_pixels`` / ``total_valid_pixels`` / ``changed_fraction``,
    * ``transitions`` — list of ``(from, to, pixels)`` sorted by pixels
      descending, excluding no-change diagonals,
    * ``class_names`` — echo of the input mapping (or {}).
    """
    labels_t1.assert_aligned(labels_t2, check_crs=check_crs)
    a = labels_t1.data
    b = labels_t2.data
    valid = labels_t1.valid_mask() & labels_t2.valid_mask()
    av = np.rint(a[valid]).astype(np.int64)
    bv = np.rint(b[valid]).astype(np.int64)

    labels = sorted(set(av.tolist()) | set(bv.tolist()))
    index = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    counts = np.zeros((n, n), dtype=np.int64)
    np.add.at(counts, (np.array([index[v] for v in av]), np.array([index[v] for v in bv])), 1)

    matrix: Dict[int, Dict[int, int]] = {}
    transitions: List[Tuple[int, int, int]] = []
    for i, la in enumerate(labels):
        row: Dict[int, int] = {}
        for j, lb in enumerate(labels):
            c = int(counts[i, j])
            if c:
                row[lb] = c
                if la != lb:
                    transitions.append((la, lb, c))
        if row:
            matrix[la] = row
    transitions.sort(key=lambda t: t[2], reverse=True)
    changed = int(sum(c for _, _, c in transitions))
    total = int(valid.sum())
    return {
        "matrix": matrix,
        "labels": labels,
        "changed_pixels": changed,
        "total_valid_pixels": total,
        "changed_fraction": (changed / total) if total else 0.0,
        "transitions": transitions,
        "class_names": dict(class_names or {}),
    }


def transition_grid(labels_t1: Grid, labels_t2: Grid, check_crs: bool = True) -> Grid:
    """Encode each pixel's from-to transition as a single integer code.

    Code ``from * 1000 + to`` supports class ids below 1000; invalid
    pixels become NaN. Handy for mapping "forest -> developed" as one
    raster that QGIS can style by category.
    """
    labels_t1.assert_aligned(labels_t2, check_crs=check_crs)
    valid = labels_t1.valid_mask() & labels_t2.valid_mask()
    out = np.full(labels_t1.shape, np.nan)
    a = np.rint(labels_t1.data[valid]).astype(np.int64)
    b = np.rint(labels_t2.data[valid]).astype(np.int64)
    if (np.abs(a) >= 1000).any() or (np.abs(b) >= 1000).any():
        raise ChangeError("transition_grid supports class ids with |id| < 1000")
    out[valid] = a * 1000 + b
    return Grid(data=out, transform=labels_t1.transform, crs=labels_t1.crs,
                nodata=np.nan, name="transition")


def summarize_transitions(
    comparison: Dict,
    pixel_area: float,
    class_names: Optional[Dict[int, str]] = None,
) -> List[Dict]:
    """Flatten transitions into records with areas, ready for CSV export."""
    names = class_names or comparison.get("class_names", {})
    records = []
    for la, lb, pixels in comparison["transitions"]:
        records.append({
            "from_class": la,
            "to_class": lb,
            "from_name": names.get(la, str(la)),
            "to_name": names.get(lb, str(lb)),
            "pixels": pixels,
            "area": pixels * pixel_area,
        })
    return records


def stable_mask(labels_t1: Grid, labels_t2: Grid, check_crs: bool = True) -> np.ndarray:
    """Boolean mask of pixels whose class did not change."""
    labels_t1.assert_aligned(labels_t2, check_crs=check_crs)
    valid = labels_t1.valid_mask() & labels_t2.valid_mask()
    same = np.zeros(labels_t1.shape, dtype=bool)
    same[valid] = np.rint(labels_t1.data[valid]) == np.rint(labels_t2.data[valid])
    return same
