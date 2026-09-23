"""Interoperability with the rest of the survey suite.

Every adapter here is *lazy and duck-typed*: sibling packages are imported
inside the function that needs them, so ``survey-change`` installs and runs
with zero suite dependencies. If a sibling is missing, you get a clear
``ImportError`` naming the package to install — never a broken import at
module load.

Supported inputs
----------------
* **survey-imagery**: index COGs / GeoTIFFs written by the site monitor
  (read from disk via :mod:`change.io`), and ``timeseries.csv`` monitor
  archives (see :mod:`change.timeseries`).
* **survey-raster**: any object exposing ``data`` (2-D array), ``transform``
  (6-tuple), ``crs``, and optionally ``nodata``/``name`` is accepted by
  :func:`grid_from_raster_like`.
* **survey-cogo / survey-gnss**: polygon vertex lists ``[(x, y), ...]`` can
  be rasterized into an AOI mask with :func:`rasterize_polygon`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import ChangeError, Grid


def grid_from_raster_like(obj: Any, name: str = "") -> Grid:
    """Build a :class:`Grid` from a survey-raster-like object.

    Accepts anything with ``data``, ``transform``, and ``crs`` attributes
    (e.g. ``survey_raster.Raster``), or a ``(data, transform, crs)`` tuple.
    """
    if isinstance(obj, (tuple, list)) and len(obj) == 3:
        data, transform, crs = obj
        nodata = None
    else:
        try:
            data = obj.data
            transform = tuple(obj.transform)
            crs = obj.crs
        except AttributeError as exc:
            raise ChangeError(
                "grid_from_raster_like needs an object with .data, .transform, "
                f"and .crs (got {type(obj).__name__})"
            ) from exc
        nodata = getattr(obj, "nodata", None)
        name = name or getattr(obj, "name", "")
    return Grid(data=np.asarray(data), transform=tuple(transform), crs=crs,
                nodata=nodata, name=name)


def grid_from_imagery_band(band: Any, name: str = "") -> Grid:
    """Build a :class:`Grid` from a survey-imagery ``BandData``-like object.

    Accepts anything with ``data`` (2-D array), ``transform`` (6-tuple or
    Affine), ``crs``, and optional ``nodata``/``name``.
    """
    try:
        data = band.data
        transform = band.transform
        crs = band.crs
    except AttributeError as exc:
        raise ChangeError(
            "grid_from_imagery_band needs .data, .transform and .crs "
            f"(got {type(band).__name__})"
        ) from exc
    # Affine (rasterio) -> 6-tuple
    if not isinstance(transform, (tuple, list)) and hasattr(transform, "__len__") is False:
        try:
            transform = (transform.a, transform.b, transform.c,
                         transform.d, transform.e, transform.f)
        except AttributeError:
            pass
    transform = tuple(transform)
    nodata = getattr(band, "nodata", None)
    name = name or getattr(band, "name", "")
    return Grid(data=np.asarray(data), transform=transform, crs=crs,
                nodata=nodata, name=name)


def rasterize_polygon(
    vertices: Sequence[Tuple[float, float]],
    grid: Grid,
) -> np.ndarray:
    """Rasterize a polygon (e.g. from survey-cogo) to a boolean AOI mask.

    ``vertices`` are ``(x, y)`` map coordinates; the polygon is closed
    automatically. Uses a scanline fill on pixel centers — pure Python,
    no rasterio needed.
    """
    verts = [(float(x), float(y)) for x, y in vertices]
    if len(verts) < 3:
        raise ChangeError("rasterize_polygon needs at least 3 vertices")
    if verts[0] != verts[-1]:
        verts = verts + [verts[0]]

    a, b, c, d, e, f = grid.transform
    det = b * f - c * e
    if det == 0:
        raise ChangeError("degenerate transform cannot be inverted")

    minx = min(v[0] for v in verts)
    maxx = max(v[0] for v in verts)
    miny = min(v[1] for v in verts)
    maxy = max(v[1] for v in verts)

    def _to_pixel(x: float, y: float) -> Tuple[float, float]:
        col = (f * (x - a) - c * (y - d)) / det
        row = (-e * (x - a) + b * (y - d)) / det
        return row, col

    corners = [_to_pixel(x, y) for x, y in
               [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]]
    r0 = max(0, int(min(r for r, _ in corners)) - 1)
    r1 = min(grid.rows, int(max(r for r, _ in corners)) + 2)
    c0 = max(0, int(min(cc for _, cc in corners)) - 1)
    c1 = min(grid.cols, int(max(cc for _, cc in corners)) + 2)

    mask = np.zeros(grid.shape, dtype=bool)
    n = len(verts)
    for r in range(r0, r1):
        for col in range(c0, c1):
            x, y = grid.pixel_center(r, col)
            inside = False
            j = n - 1
            for i in range(n):
                xi, yi = verts[i]
                xj, yj = verts[j]
                if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                    inside = not inside
                j = i
            mask[r, col] = inside
    return mask


def clip_to_aoi(grid: Grid, aoi_mask: np.ndarray) -> Grid:
    """Return a copy of ``grid`` with pixels outside ``aoi_mask`` set to NaN."""
    if aoi_mask.shape != grid.shape:
        raise ChangeError("AOI mask shape does not match grid shape")
    out = grid.copy()
    data = out.data.astype(np.float64, copy=True)
    data[~np.asarray(aoi_mask, dtype=bool)] = np.nan
    out.data = data
    return out


def describe_suite_inputs() -> Dict[str, str]:
    """Human-readable summary of accepted sibling-suite inputs."""
    return {
        "survey-imagery": "Index COGs / GeoTIFFs from the site monitor (read from disk), "
                          "timeseries.csv monitor archives, BandData objects (in memory).",
        "survey-raster": "Any object with .data / .transform / .crs (or a (data, transform, crs) tuple).",
        "survey-cogo": "Polygon vertex lists [(x, y), ...] rasterized to AOI masks.",
        "survey-gnss": "Point lists used as polygon vertices for AOI masks.",
    }
