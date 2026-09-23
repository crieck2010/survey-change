"""Multi-epoch change tracking.

While :func:`change.detection.detect` compares exactly two dates, real
monitoring asks "when did it change, and how much in total?" This module
chains sequential pairs, accumulates net change, and flags the epoch of
first significant change per pixel — the inputs a per-pass site monitor
needs to turn a stack of index rasters into a disturbance timeline.

It also reads the ``timeseries.csv`` files written by survey-imagery's
site monitor, so a change analysis can start from an existing monitoring
archive instead of raw scenes.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import ChangeConfig, ChangeError, Grid
from . import detection as _det


@dataclass
class Epoch:
    """One dated observation: a label plus its grid."""
    date: str  # ISO date, e.g. "2026-07-16"
    grid: Grid


def _sorted_epochs(epochs: Sequence[Epoch]) -> List[Epoch]:
    return sorted(epochs, key=lambda e: e.date)


def sequential_changes(
    epochs: Sequence[Epoch],
    config: Optional[ChangeConfig] = None,
) -> List[Dict[str, Any]]:
    """Detect change between each consecutive epoch pair.

    Returns a list of dicts with ``from_date``, ``to_date``, ``change``
    (Grid), ``change_mask``, ``gain_mask``, ``loss_mask``,
    ``threshold_used``, and ``stats``.
    """
    eps = _sorted_epochs(epochs)
    if len(eps) < 2:
        raise ChangeError("sequential_changes needs at least 2 epochs")
    cfg = config or ChangeConfig()
    out = []
    for e1, e2 in zip(eps[:-1], eps[1:]):
        res = _det.detect(e1.grid, e2.grid, cfg)
        out.append({
            "from_date": e1.date,
            "to_date": e2.date,
            "change": res.change_grid,
            "change_mask": res.change_mask,
            "gain_mask": res.gain_mask,
            "loss_mask": res.loss_mask,
            "threshold_used": res.threshold_used,
            "stats": res.stats,
        })
    return out


def cumulative_change(epochs: Sequence[Epoch]) -> Grid:
    """Net signed change from the first to the last epoch (``last - first``)."""
    eps = _sorted_epochs(epochs)
    if len(eps) < 2:
        raise ChangeError("cumulative_change needs at least 2 epochs")
    return _det.difference(eps[0].grid, eps[-1].grid)


def change_onset(
    epochs: Sequence[Epoch],
    threshold: float,
) -> Grid:
    """Epoch index of first significant absolute change per pixel.

    Compares every epoch against the *first* (baseline) epoch. Pixels are
    coded 0 = never exceeded ``threshold``, else the 1-based index of the
    first epoch whose ``|value - baseline|`` exceeded it. Useful for
    disturbance-date mapping.
    """
    eps = _sorted_epochs(epochs)
    if len(eps) < 2:
        raise ChangeError("change_onset needs at least 2 epochs")
    base = eps[0].grid
    out = np.zeros(base.shape, dtype=np.float64)
    for i, ep in enumerate(eps[1:], start=1):
        base.assert_aligned(ep.grid)
        valid = base.valid_mask() & ep.grid.valid_mask()
        changed = valid & (np.abs(ep.grid.data - base.data) >= threshold) & (out == 0)
        out[changed] = i
    return Grid(data=out, transform=base.transform, crs=base.crs,
                nodata=np.nan, name="change_onset")


def persistent_change_mask(
    epochs: Sequence[Epoch],
    threshold: float,
    min_epochs: int = 2,
) -> np.ndarray:
    """Pixels whose absolute change vs. baseline exceeded ``threshold`` in at
    least ``min_epochs`` epochs. Filters one-off noise (a cloudy pass, a
    mowing event) from durable change."""
    eps = _sorted_epochs(epochs)
    if len(eps) < 2:
        raise ChangeError("persistent_change_mask needs at least 2 epochs")
    base = eps[0].grid
    count = np.zeros(base.shape, dtype=np.int32)
    for ep in eps[1:]:
        base.assert_aligned(ep.grid)
        valid = base.valid_mask() & ep.grid.valid_mask()
        count[valid & (np.abs(ep.grid.data - base.data) >= threshold)] += 1
    return count >= min_epochs


# --- survey-imagery monitor interop --------------------------------------

IMAGERY_TIMESERIES_FIELDS = (
    "scene_id", "date", "index", "mean", "std", "min", "max",
    "valid_pixels", "clear_fraction",
)


def read_imagery_timeseries(path: str) -> List[Dict[str, Any]]:
    """Read a survey-imagery monitor ``timeseries.csv`` into row dicts.

    Numeric fields are converted to float where possible; rows missing a
    date or index are skipped. Raises ChangeError if the header does not
    look like a survey-imagery timeseries file.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or [])
        if not {"date", "index", "mean"}.issubset(fields):
            raise ChangeError(
                f"{path} does not look like a survey-imagery timeseries.csv "
                f"(fields: {sorted(fields)})"
            )
        rows: List[Dict[str, Any]] = []
        for row in reader:
            if not row.get("date") or not row.get("index"):
                continue
            rec = dict(row)
            for key in ("mean", "std", "min", "max", "valid_pixels", "clear_fraction"):
                if key in rec and rec[key] not in (None, ""):
                    try:
                        rec[key] = float(rec[key])
                    except (TypeError, ValueError):
                        pass
            rows.append(rec)
    rows.sort(key=lambda r: (str(r.get("date")), str(r.get("index"))))
    return rows


def index_breaks(
    rows: Sequence[Dict[str, Any]],
    index: str,
    threshold: float,
) -> List[Dict[str, Any]]:
    """Find abrupt breaks in a monitor index's mean-value series.

    For one ``index`` (e.g. ``"NDVI"``), walks the date-sorted means and
    reports steps where ``|mean[t] - mean[t-1]| >= threshold``. Returns
    records with ``index``, ``from_date``, ``to_date``, ``from_mean``,
    ``to_mean``, and ``delta``. This is the lightweight, no-raster way to
    spot *when* something happened before running full change detection.
    """
    series = sorted(
        (r for r in rows if r.get("index") == index and isinstance(r.get("mean"), (int, float))),
        key=lambda r: str(r.get("date")),
    )
    breaks = []
    for prev, cur in zip(series[:-1], series[1:]):
        delta = float(cur["mean"]) - float(prev["mean"])
        if abs(delta) >= threshold:
            breaks.append({
                "index": index,
                "from_date": prev.get("date"),
                "to_date": cur.get("date"),
                "from_mean": float(prev["mean"]),
                "to_mean": float(cur["mean"]),
                "delta": delta,
            })
    return breaks
