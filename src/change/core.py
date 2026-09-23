"""Core data structures for survey-change.

Everything in this module is pure logic: no file I/O, no UI imports, no
network access. A :class:`Grid` is the atomic unit of work — a 2-D numpy
array plus its georeferencing — so every detection algorithm in this
package operates on plain in-memory data and stays testable in isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class ChangeError(Exception):
    """Base exception for all survey-change errors."""


class AlignmentError(ChangeError):
    """Raised when two grids cannot be compared (shape/transform/CRS mismatch)."""


class GridError(ChangeError):
    """Raised for invalid grid construction or operations."""


@dataclass
class Grid:
    """A single-band raster grid: data plus georeferencing.

    Parameters
    ----------
    data:
        2-D numpy array of pixel values. Converted with ``np.asarray`` and
        cast to float64 unless ``keep_dtype`` is set.
    transform:
        6-element GDAL-style geotransform ``(a, b, c, d, e, f)`` where
        ``x = a + b*col + c*row`` and ``y = d + e*col + f*row``.
    crs:
        Coordinate reference system as an EPSG string (e.g. ``"EPSG:32617"``)
        or WKT. Only compared for equality, never reprojected here.
    nodata:
        Value marking missing data, or None.
    name:
        Optional human label (band name, index name, date, ...).
    """

    data: np.ndarray
    transform: Tuple[float, float, float, float, float, float] = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
    crs: Optional[str] = None
    nodata: Optional[float] = None
    name: str = ""
    keep_dtype: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        arr = self.data if isinstance(self.data, np.ndarray) else np.asarray(self.data)
        if arr.ndim != 2:
            raise GridError(f"Grid data must be 2-D, got shape {arr.shape}")
        if not self.keep_dtype and not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float64)
        object.__setattr__(self, "data", arr)
        if len(self.transform) != 6:
            raise GridError("transform must be a 6-element GDAL geotransform tuple")

    @property
    def rows(self) -> int:
        return self.data.shape[0]

    @property
    def cols(self) -> int:
        return self.data.shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape

    @property
    def pixel_width(self) -> float:
        return abs(self.transform[1])

    @property
    def pixel_height(self) -> float:
        return abs(self.transform[5])

    @property
    def pixel_area(self) -> float:
        """Area of one pixel in CRS units squared."""
        return self.pixel_width * self.pixel_height

    def bounds(self) -> Tuple[float, float, float, float]:
        """(minx, miny, maxx, maxy) in CRS coordinates."""
        a, b, c, d, e, f = self.transform
        xs = [a + b * col + c * row for col, row in ((0, 0), (self.cols, 0), (0, self.rows), (self.cols, self.rows))]
        ys = [d + e * col + f * row for col, row in ((0, 0), (self.cols, 0), (0, self.rows), (self.cols, self.rows))]
        return (min(xs), min(ys), max(xs), max(ys))

    def valid_mask(self) -> np.ndarray:
        """Boolean mask of pixels that are finite and not nodata."""
        mask = np.isfinite(self.data)
        if self.nodata is not None:
            mask &= self.data != self.nodata
        return mask

    def copy(self) -> "Grid":
        return Grid(
            data=self.data.copy(),
            transform=self.transform,
            crs=self.crs,
            nodata=self.nodata,
            name=self.name,
        )

    def aligned_with(self, other: "Grid", check_crs: bool = True) -> bool:
        """True when two grids share shape and geotransform (and CRS)."""
        if self.shape != other.shape:
            return False
        if any(abs(x - y) > 1e-9 for x, y in zip(self.transform, other.transform)):
            return False
        if check_crs and self.crs is not None and other.crs is not None:
            return self.crs == other.crs
        return True

    def assert_aligned(self, other: "Grid", check_crs: bool = True) -> None:
        if not self.aligned_with(other, check_crs=check_crs):
            raise AlignmentError(
                f"Grids are not aligned: self shape={self.shape} transform={self.transform} "
                f"crs={self.crs} vs other shape={other.shape} transform={other.transform} "
                f"crs={other.crs}. Resample to a common grid first."
            )

    def xy(self, row: int, col: int) -> Tuple[float, float]:
        """Map coordinates of a pixel's upper-left corner."""
        a, b, c, d, e, f = self.transform
        return (a + b * col + c * row, d + e * col + f * row)

    def pixel_center(self, row: int, col: int) -> Tuple[float, float]:
        a, b, c, d, e, f = self.transform
        return (a + b * (col + 0.5) + c * (row + 0.5), d + e * (col + 0.5) + f * (row + 0.5))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "transform": list(self.transform),
            "crs": self.crs,
            "nodata": self.nodata,
            "name": self.name,
        }


@dataclass
class ChangeConfig:
    """Configuration for a change-detection run.

    Serializes to plain JSON-compatible dicts so configs round-trip through
    files and the CLI without loss.
    """

    method: str = "difference"  # difference | normalized_difference | relative_change |
    # ratio | cva | post_classification
    threshold_method: str = "otsu"  # otsu | manual | percentile | none
    threshold: Optional[float] = None  # for manual
    percentile: float = 95.0  # for percentile
    gain_threshold: Optional[float] = None  # overrides positive side
    loss_threshold: Optional[float] = None  # overrides negative side (absolute value)
    min_region_pixels: int = 1  # drop change regions smaller than this
    connectivity: int = 8  # 4 or 8
    mask_nodata: bool = True
    name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "threshold_method": self.threshold_method,
            "threshold": self.threshold,
            "percentile": self.percentile,
            "gain_threshold": self.gain_threshold,
            "loss_threshold": self.loss_threshold,
            "min_region_pixels": self.min_region_pixels,
            "connectivity": self.connectivity,
            "mask_nodata": self.mask_nodata,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChangeConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ChangeResult:
    """Outcome of a change-detection run."""

    change_grid: Grid            # signed change magnitude (t2 - t1 semantics)
    change_mask: np.ndarray      # boolean: significant change
    gain_mask: np.ndarray        # boolean: significant positive change
    loss_mask: np.ndarray        # boolean: significant negative change
    threshold_used: Optional[float]
    stats: Dict[str, Any] = field(default_factory=dict)
    regions: List[Dict[str, Any]] = field(default_factory=list)
    config: Optional[ChangeConfig] = None
