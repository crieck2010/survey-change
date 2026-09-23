"""Summary statistics for change results.

Small, dependency-free aggregations that turn change grids and masks into
reportable numbers: pixel counts, areas, and per-side distributions. Used
by the CLI, the QGIS layer metadata, and any reporting workflow.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .core import ChangeResult, Grid


def summarize_change(
    change: Grid,
    change_mask: np.ndarray,
    gain_mask: Optional[np.ndarray] = None,
    loss_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Summarize a signed change grid over a boolean change mask."""
    data = change.data
    vals = data[np.asarray(change_mask, dtype=bool)]
    vals = vals[np.isfinite(vals)]
    px = change.pixel_area
    out: Dict[str, Any] = {
        "pixels": int(np.asarray(change_mask, dtype=bool).sum()),
        "area": float(np.asarray(change_mask, dtype=bool).sum() * px),
        "pixel_area": px,
    }
    if vals.size:
        out.update({
            "mean_change": float(vals.mean()),
            "min_change": float(vals.min()),
            "max_change": float(vals.max()),
            "std_change": float(vals.std()),
        })
    else:
        out.update({"mean_change": None, "min_change": None,
                    "max_change": None, "std_change": None})
    for side, m in (("gain", gain_mask), ("loss", loss_mask)):
        if m is None:
            continue
        m = np.asarray(m, dtype=bool)
        side_vals = data[m]
        side_vals = side_vals[np.isfinite(side_vals)]
        rec: Dict[str, Any] = {
            "pixels": int(m.sum()),
            "area": float(m.sum() * px),
        }
        if side_vals.size:
            rec["mean_change"] = float(side_vals.mean())
            rec["min_change"] = float(side_vals.min())
            rec["max_change"] = float(side_vals.max())
        out[side] = rec
    return out


def histogram(
    change: Grid,
    change_mask: np.ndarray,
    bins: int = 50,
) -> Dict[str, List[float]]:
    """Histogram of signed change values inside the change mask."""
    vals = change.data[np.asarray(change_mask, dtype=bool)]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"counts": [], "edges": []}
    counts, edges = np.histogram(vals, bins=bins)
    return {"counts": [int(c) for c in counts], "edges": [float(e) for e in edges]}


def region_table(
    regions: List[Dict],
    pixel_area: float,
    change: Optional[Grid] = None,
    labels: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Flatten region dicts into CSV-ready records with areas."""
    rows: List[Dict[str, Any]] = []
    for reg in regions:
        rec: Dict[str, Any] = {
            "label": reg["label"],
            "pixels": reg["pixels"],
            "area": reg["pixels"] * pixel_area,
            "centroid_row": reg["centroid"][0],
            "centroid_col": reg["centroid"][1],
        }
        if change is not None and labels is not None:
            vals = change.data[labels == reg["label"]]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                rec["mean_change"] = float(vals.mean())
                rec["min_change"] = float(vals.min())
                rec["max_change"] = float(vals.max())
        rows.append(rec)
    return rows


def attach_stats(result: ChangeResult) -> ChangeResult:
    """Fill ``result.stats`` with summary + gain/loss breakdowns."""
    result.stats = summarize_change(
        result.change_grid, result.change_mask, result.gain_mask, result.loss_mask
    )
    return result
