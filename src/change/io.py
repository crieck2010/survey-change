"""File I/O: rasters, vectors, and tabular change outputs.

Raster reads/writes go through rasterio (a hard dependency, like the rest
of the suite's raster modules). Everything tabular or vector is plain
stdlib (csv/json) so reports stay dependency-free.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import ChangeError, Grid

_RASTERIO_MISSING = "rasterio is required for raster I/O"


def _require_rasterio():
    try:
        import rasterio
        from rasterio.transform import Affine
    except ImportError as exc:
        raise ChangeError(_RASTERIO_MISSING) from exc
    return rasterio, Affine


def read_grid(path: str, band: int = 1, name: str = "") -> Grid:
    """Read one raster band into a :class:`Grid`."""
    rasterio, Affine = _require_rasterio()
    with rasterio.open(path) as src:
        data = src.read(band).astype(np.float64)
        transform = src.transform
        if isinstance(transform, Affine):
            transform = (transform.a, transform.b, transform.c,
                         transform.d, transform.e, transform.f)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
            nodata = np.nan
        crs = str(src.crs) if src.crs else None
    return Grid(data=data, transform=tuple(transform), crs=crs,
                nodata=nodata, name=name or os.path.basename(path))


def write_geotiff(
    grid: Grid,
    path: str,
    dtype: str = "float32",
    compress: str = "deflate",
    tiled: bool = True,
) -> str:
    """Write a :class:`Grid` as a single-band GeoTIFF."""
    rasterio, Affine = _require_rasterio()
    a, b, c, d, e, f = grid.transform
    profile = {
        "driver": "GTiff",
        "height": grid.rows,
        "width": grid.cols,
        "count": 1,
        "dtype": dtype,
        "crs": grid.crs,
        "transform": Affine(a, b, c, d, e, f),
        "tiled": tiled,
        "compress": compress,
    }
    arr = grid.data.astype(dtype)
    arr[~np.isfinite(grid.data)] = np.nan if "float" in dtype else 0
    if "float" not in dtype:
        profile["nodata"] = 0
    else:
        profile["nodata"] = np.nan
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
    return path


def write_cog(grid: Grid, path: str, dtype: str = "float32") -> str:
    """Write a :class:`Grid` as a Cloud-Optimized GeoTIFF.

    Falls back to a tiled, overviews-included GTiff when the COG driver
    is unavailable in the local GDAL build.
    """
    rasterio, Affine = _require_rasterio()
    a, b, c, d, e, f = grid.transform
    arr = grid.data.astype(dtype)
    profile = {
        "driver": "COG",
        "height": grid.rows,
        "width": grid.cols,
        "count": 1,
        "dtype": dtype,
        "crs": grid.crs,
        "transform": Affine(a, b, c, d, e, f),
        "compress": "deflate",
        "nodata": np.nan if "float" in dtype else 0,
    }
    try:
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr, 1)
    except Exception:
        # COG driver missing: plain tiled GTiff with overviews instead.
        tmp = write_geotiff(grid, path, dtype=dtype)
        with rasterio.open(tmp, "r+") as dst:
            try:
                dst.build_overviews([2, 4, 8, 16], rasterio.enums.Resampling.average)
                dst.update_tags(ns="rio_overview", resampling="average")
            except Exception:
                pass
    return path


def write_mask_geotiff(mask: np.ndarray, template: Grid, path: str) -> str:
    """Write a boolean mask as a byte GeoTIFF sharing ``template``'s grid."""
    grid = Grid(data=np.asarray(mask, dtype=np.uint8), transform=template.transform,
                crs=template.crs, nodata=0, name="mask", keep_dtype=True)
    return write_geotiff(grid, path, dtype="uint8")


def write_csv(records: Sequence[Dict[str, Any]], path: str,
              fieldnames: Optional[Sequence[str]] = None) -> str:
    """Write a list of dicts to CSV; returns the path."""
    records = list(records)
    if not records:
        raise ChangeError("write_csv: no records to write")
    fields = list(fieldnames) if fieldnames else list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return path


def read_geojson(path: str) -> Dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def change_report(
    result_stats: Dict[str, Any],
    regions: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Assemble a JSON-serializable change report dict."""
    report: Dict[str, Any] = {"summary": result_stats}
    if regions is not None:
        report["regions"] = list(regions)
    return report


def write_report(report: Dict[str, Any], path: str) -> str:
    def _default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=_default)
    return path
