"""Change-detection algorithms.

All functions take :class:`~change.core.Grid` objects and return new
``Grid`` objects — the inputs are never mutated. Nodata/invalid pixels
propagate as NaN in outputs so downstream statistics can ignore them.

Conventions
-----------
* ``t1`` is the earlier acquisition, ``t2`` the later one.
* Signed outputs use ``t2 - t1`` semantics: positive values are *gains*,
  negative values are *losses*.
* Thresholding is applied to the absolute change magnitude unless the
  caller supplies separate gain/loss thresholds.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .core import AlignmentError, ChangeConfig, ChangeError, ChangeResult, Grid


def _combined_valid(*grids: Grid) -> np.ndarray:
    mask = np.ones(grids[0].shape, dtype=bool)
    for g in grids:
        mask &= g.valid_mask()
    return mask


def _finish_change(data: np.ndarray, template: Grid, name: str) -> Grid:
    return Grid(data=data, transform=template.transform, crs=template.crs,
               nodata=np.nan, name=name)


def difference(t1: Grid, t2: Grid, check_crs: bool = True) -> Grid:
    """Simple differencing: ``t2 - t1``."""
    t1.assert_aligned(t2, check_crs=check_crs)
    out = np.full(t1.shape, np.nan)
    valid = _combined_valid(t1, t2)
    out[valid] = t2.data[valid] - t1.data[valid]
    return _finish_change(out, t1, f"difference({t1.name},{t2.name})")


def normalized_difference_change(t1: Grid, t2: Grid, check_crs: bool = True) -> Grid:
    """Normalized change: ``(t2 - t1) / (|t2| + |t1|)``.

    Bounded in ``[-1, 1]``; useful for comparing indices (e.g. NDVI) whose
    absolute scales differ between scenes.
    """
    t1.assert_aligned(t2, check_crs=check_crs)
    out = np.full(t1.shape, np.nan)
    valid = _combined_valid(t1, t2)
    denom = np.abs(t2.data) + np.abs(t1.data)
    ok = valid & (denom > 0)
    out[ok] = (t2.data[ok] - t1.data[ok]) / denom[ok]
    return _finish_change(out, t1, f"nd_change({t1.name},{t2.name})")


def relative_change(t1: Grid, t2: Grid, check_crs: bool = True) -> Grid:
    """Relative change in percent: ``100 * (t2 - t1) / |t1|``."""
    t1.assert_aligned(t2, check_crs=check_crs)
    out = np.full(t1.shape, np.nan)
    valid = _combined_valid(t1, t2)
    ok = valid & (np.abs(t1.data) > 0)
    out[ok] = 100.0 * (t2.data[ok] - t1.data[ok]) / np.abs(t1.data[ok])
    return _finish_change(out, t1, f"relative_change({t1.name},{t2.name})")


def ratio_change(t1: Grid, t2: Grid, check_crs: bool = True) -> Grid:
    """Log-ratio change: ``log(t2 / t1)``.

    Symmetric around zero and robust to multiplicative illumination
    differences; the classic SAR change operator, also handy for optical
    brightness changes.
    """
    t1.assert_aligned(t2, check_crs=check_crs)
    out = np.full(t1.shape, np.nan)
    valid = _combined_valid(t1, t2)
    ok = valid & (t1.data > 0) & (t2.data > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[ok] = np.log(t2.data[ok] / t1.data[ok])
    return _finish_change(out, t1, f"log_ratio({t1.name},{t2.name})")


def change_vector_analysis(
    bands_t1: Sequence[Grid],
    bands_t2: Sequence[Grid],
    check_crs: bool = True,
) -> Tuple[Grid, Grid]:
    """Change Vector Analysis over N spectral bands.

    Returns ``(magnitude, angle)``. Magnitude is the Euclidean length of
    the change vector (always >= 0, NaN where any band is invalid).
    Angle is the direction of the vector in the plane of the first two
    bands, in degrees in ``(-180, 180]`` — useful for separating *kinds*
    of change (e.g. vegetation loss vs. soil exposure).

    Both band lists must be the same length, and every grid must align.
    """
    if len(bands_t1) != len(bands_t2):
        raise ChangeError("bands_t1 and bands_t2 must have the same length")
    if len(bands_t1) < 2:
        raise ChangeError("CVA needs at least 2 bands per date")
    for g in list(bands_t1[1:]) + list(bands_t2):
        bands_t1[0].assert_aligned(g, check_crs=check_crs)

    diffs = []
    valid = np.ones(bands_t1[0].shape, dtype=bool)
    for b1, b2 in zip(bands_t1, bands_t2):
        valid &= _combined_valid(b1, b2)
        d = np.full(b1.shape, np.nan)
        d[valid] = b2.data[valid] - b1.data[valid]
        diffs.append(d)
    stack = np.stack(diffs)  # (n_bands, rows, cols)
    mag = np.full(bands_t1[0].shape, np.nan)
    mag[valid] = np.linalg.norm(stack[:, valid], axis=0)
    ang = np.full(bands_t1[0].shape, np.nan)
    ang[valid] = np.degrees(np.arctan2(diffs[1][valid], diffs[0][valid]))
    template = bands_t1[0]
    return (
        _finish_change(mag, template, "cva_magnitude"),
        _finish_change(ang, template, "cva_angle"),
    )


def otsu_threshold(values: np.ndarray) -> float:
    """Otsu's threshold on the absolute change magnitudes.

    Finds the threshold maximizing between-class variance of the
    ``|change|`` histogram. NaNs are ignored. Raises ChangeError when
    there is nothing to threshold.
    """
    v = np.abs(np.asarray(values, dtype=float).ravel())
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ChangeError("otsu_threshold: no finite values")
    if np.all(v == v[0]):
        return float(v[0])
    hist, edges = np.histogram(v, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    sum_total = (hist * centers).sum()
    best_t = centers[0]
    best_var = -1.0
    w0 = 0.0
    sum0 = 0.0
    for i in range(len(hist) - 1):
        w0 += hist[i]
        sum0 += hist[i] * centers[i]
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        mean0 = sum0 / w0
        mean1 = (sum_total - sum0) / w1
        var = w0 * w1 * (mean0 - mean1) ** 2
        if var > best_var:
            best_var = var
            best_t = centers[i]
    return float(best_t)


def apply_threshold(
    change: Grid,
    method: str = "otsu",
    threshold: Optional[float] = None,
    percentile: float = 95.0,
) -> Tuple[np.ndarray, Optional[float]]:
    """Threshold a signed change grid into a boolean change mask.

    The threshold is applied to ``|change|``. Returns ``(mask, used)``
    where ``used`` is the numeric threshold, or None for ``method="none"``.

    Methods: ``"otsu"``, ``"manual"`` (needs ``threshold``),
    ``"percentile"`` (top ``100 - percentile`` % of magnitudes), ``"none"``.
    """
    data = change.data
    if method == "none":
        return np.isfinite(data) & (data != 0), None
    if method == "manual":
        if threshold is None:
            raise ChangeError("manual thresholding needs a threshold value")
        used = float(threshold)
    elif method == "otsu":
        used = otsu_threshold(data)
    elif method == "percentile":
        v = np.abs(data[np.isfinite(data)])
        if v.size == 0:
            raise ChangeError("percentile thresholding: no finite values")
        used = float(np.percentile(v, percentile))
    else:
        raise ChangeError(f"unknown threshold method: {method!r}")
    mask = np.isfinite(data) & (np.abs(data) >= used)
    return mask, used


def gain_loss_masks(
    change: Grid,
    change_mask: np.ndarray,
    gain_threshold: Optional[float] = None,
    loss_threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a change mask into gain (positive) and loss (negative) masks.

    Optional per-side thresholds override the shared change mask on that
    side: e.g. ``gain_threshold=0.2`` keeps only ``change >= 0.2`` as gain.
    """
    data = change.data
    if gain_threshold is not None:
        gain = np.isfinite(data) & (data >= gain_threshold)
    else:
        gain = change_mask & (data > 0)
    if loss_threshold is not None:
        loss = np.isfinite(data) & (data <= -abs(loss_threshold))
    else:
        loss = change_mask & (data < 0)
    return gain, loss


def detect(
    t1: Grid,
    t2: Grid,
    config: Optional[ChangeConfig] = None,
    bands_t1: Optional[Sequence[Grid]] = None,
    bands_t2: Optional[Sequence[Grid]] = None,
) -> ChangeResult:
    """Run a full change-detection pass and return a :class:`ChangeResult`.

    ``config.method`` selects the operator: ``difference``,
    ``normalized_difference``, ``relative_change``, ``ratio``, or ``cva``
    (which needs ``bands_t1``/``bands_t2``). Thresholding follows
    ``config.threshold_method``; regions smaller than
    ``config.min_region_pixels`` are dropped from the mask.
    """
    from . import segmentation as _seg  # local import: keeps module DAG clean

    cfg = config or ChangeConfig()
    method = cfg.method.lower()
    if method == "cva":
        if bands_t1 is None or bands_t2 is None:
            raise ChangeError("cva method needs bands_t1 and bands_t2")
        magnitude, _angle = change_vector_analysis(bands_t1, bands_t2)
        change = magnitude
    elif method == "difference":
        change = difference(t1, t2)
    elif method == "normalized_difference":
        change = normalized_difference_change(t1, t2)
    elif method == "relative_change":
        change = relative_change(t1, t2)
    elif method == "ratio":
        change = ratio_change(t1, t2)
    else:
        raise ChangeError(f"unknown detection method: {cfg.method!r}")

    mask, used = apply_threshold(
        change,
        method=cfg.threshold_method,
        threshold=cfg.threshold,
        percentile=cfg.percentile,
    )
    gain, loss = gain_loss_masks(change, mask, cfg.gain_threshold, cfg.loss_threshold)

    # Drop tiny regions; keep the per-side masks consistent with the mask.
    if cfg.min_region_pixels > 1 and mask.any():
        labels, regions = _seg.label_regions(mask, connectivity=cfg.connectivity)
        keep = {r["label"] for r in regions if r["pixels"] >= cfg.min_region_pixels}
        if keep:
            keep_mask = np.isin(labels, list(keep))
            mask = mask & keep_mask
            gain = gain & keep_mask
            loss = loss & keep_mask
        else:
            mask[:] = False
            gain[:] = False
            loss[:] = False

    stats = {
        "changed_pixels": int(mask.sum()),
        "gain_pixels": int(gain.sum()),
        "loss_pixels": int(loss.sum()),
        "changed_fraction": float(mask.mean()),
    }
    return ChangeResult(
        change_grid=change,
        change_mask=mask,
        gain_mask=gain,
        loss_mask=loss,
        threshold_used=used,
        stats=stats,
        config=cfg,
    )
