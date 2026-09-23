"""Vectorization of change regions into polygons.

Change regions (integer label grids from :mod:`change.segmentation`) are
converted to polygon rings with a marching-squares style edge walk, then
simplified with Douglas-Peucker and exported as GeoJSON features in map
coordinates. No shapely/GDAL dependency — everything is pure Python plus
numpy — so the engine stays dependency-light and the polygons drop
straight into QGIS.
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import Grid

_Coord = Tuple[float, float]


def _cell_edges(mask: np.ndarray) -> List[Tuple[_Coord, _Coord]]:
    """Boundary edge segments of a boolean mask, in pixel-corner coords.

    Corners use (x=col, y=row) with integer coordinates; the top-left
    corner of cell (r, c) is (c, r).
    """
    rows, cols = mask.shape
    # pad so border cells get their outer edges
    padded = np.zeros((rows + 2, cols + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask
    segs: List[Tuple[_Coord, _Coord]] = []
    for r in range(1, rows + 1):
        prow = padded[r]
        for c in range(1, cols + 1):
            if not prow[c]:
                continue
            x0, y0 = float(c - 1), float(r - 1)  # back in unpadded coords
            if not padded[r - 1, c]:
                segs.append(((x0, y0), (x0 + 1.0, y0)))          # top
            if not padded[r + 1, c]:
                segs.append(((x0, y0 + 1.0), (x0 + 1.0, y0 + 1.0)))  # bottom
            if not padded[r, c - 1]:
                segs.append(((x0, y0), (x0, y0 + 1.0)))          # left
            if not padded[r, c + 1]:
                segs.append(((x0 + 1.0, y0), (x0 + 1.0, y0 + 1.0)))  # right
    return segs


def _chain_rings(segs: List[Tuple[_Coord, _Coord]]) -> List[List[_Coord]]:
    """Join edge segments into closed rings.

    Direction-agnostic: segments are flipped as needed so they chain into
    cycles regardless of the orientation each edge was emitted with.
    """
    touch: Dict[_Coord, List[int]] = {}
    for i, (a, b) in enumerate(segs):
        touch.setdefault(a, []).append(i)
        touch.setdefault(b, []).append(i)
    used = [False] * len(segs)
    rings: List[List[_Coord]] = []
    for i, (a, b) in enumerate(segs):
        if used[i]:
            continue
        used[i] = True
        ring = [a, b]
        cur = b
        while True:
            nxt = None
            for j in touch.get(cur, ()):
                if not used[j]:
                    nxt = j
                    break
            if nxt is None:
                break  # open fragment; not a valid ring
            used[nxt] = True
            sa, sb = segs[nxt]
            cur = sb if sa == cur else sa
            if cur == ring[0]:
                ring.append(cur)
                break
            ring.append(cur)
        if len(ring) >= 4 and ring[0] == ring[-1]:
            rings.append(ring)
    return rings


def _signed_area(ring: Sequence[_Coord]) -> float:
    s = 0.0
    n = len(ring)
    for i in range(n - 1):
        x0, y0 = ring[i]
        x1, y1 = ring[i + 1]
        s += x0 * y1 - x1 * y0
    return s / 2.0


def douglas_peucker(ring: Sequence[_Coord], tolerance: float) -> List[_Coord]:
    """Simplify a ring (or line) with the Douglas-Peucker algorithm.

    ``tolerance`` is in the ring's coordinate units. The ring is treated
    as closed: first and last points are preserved.
    """
    pts = list(ring)
    if len(pts) <= 4 or tolerance <= 0:
        return pts
    closed = pts[0] == pts[-1]
    work = pts[:-1] if closed else pts

    def _perp_dist(p: _Coord, a: _Coord, b: _Coord) -> float:
        dx, dy = b[0] - a[0], b[1] - a[1]
        denom = math.hypot(dx, dy)
        if denom == 0:
            return math.hypot(p[0] - a[0], p[1] - a[1])
        return abs(dy * p[0] - dx * p[1] + b[0] * a[1] - b[1] * a[0]) / denom

    keep = [False] * len(work)
    keep[0] = keep[-1] = True
    stack = [(0, len(work) - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 - i0 < 2:
            continue
        a, b = work[i0], work[i1]
        dmax, imax = -1.0, -1
        for i in range(i0 + 1, i1):
            d = _perp_dist(work[i], a, b)
            if d > dmax:
                dmax, imax = d, i
        if dmax > tolerance:
            keep[imax] = True
            stack.append((i0, imax))
            stack.append((imax, i1))
    out = [p for p, k in zip(work, keep) if k]
    if closed:
        out.append(out[0])
    return out


def _to_map_coords(ring: Sequence[_Coord], grid: Grid) -> List[_Coord]:
    a, b, c, d, e, f = grid.transform
    return [(a + b * x + c * y, d + e * x + f * y) for x, y in ring]


def region_polygon(
    labels: np.ndarray,
    label_id: int,
    grid: Grid,
    simplify_tolerance: float = 0.0,
) -> Optional[Dict]:
    """Build a GeoJSON Polygon dict for one labeled region, in map coords.

    Returns None when the region has no traceable boundary.
    """
    mask = labels == label_id
    if not mask.any():
        return None
    segs = _cell_edges(mask)
    if not segs:
        return None
    rings = _chain_rings(segs)
    if not rings:
        return None
    map_rings = [_to_map_coords(r, grid) for r in rings]
    # Exterior = ring with the largest absolute area; the rest are holes.
    areas = [abs(_signed_area(r)) for r in map_rings]
    order = sorted(range(len(map_rings)), key=lambda i: areas[i], reverse=True)
    exterior = map_rings[order[0]]
    holes = [map_rings[i] for i in order[1:]]
    if simplify_tolerance > 0:
        exterior = douglas_peucker(exterior, simplify_tolerance)
        holes = [douglas_peucker(h, simplify_tolerance) for h in holes]
        holes = [h for h in holes if len(h) >= 4]
    if len(exterior) < 4:
        return None
    # GeoJSON winding: exterior CCW, holes CW (in standard math orientation).
    if _signed_area(exterior) < 0:
        exterior = exterior[::-1]
    fixed_holes = []
    for h in holes:
        fixed_holes.append(h[::-1] if _signed_area(h) > 0 else h)
    return {"type": "Polygon", "coordinates": [exterior] + fixed_holes}


def polygon_area(geometry: Dict) -> float:
    """Planar area of a GeoJSON Polygon/MultiPolygon in coordinate units^2."""
    def _ring_area(ring: Sequence[Sequence[float]]) -> float:
        s = 0.0
        for i in range(len(ring) - 1):
            s += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1]
        return abs(s) / 2.0

    geoms = [geometry] if geometry["type"] == "Polygon" else geometry["coordinates"]
    if geometry["type"] == "MultiPolygon":
        return sum(_ring_area(r) - sum(_ring_area(h) for h in poly[1:]) for poly in geoms for r in [poly[0]])
    rings = geometry["coordinates"]
    return _ring_area(rings[0]) - sum(_ring_area(h) for h in rings[1:])


def regions_to_features(
    labels: np.ndarray,
    regions: List[Dict],
    grid: Grid,
    change_grid: Optional[Grid] = None,
    simplify_tolerance: float = 0.0,
    kind: str = "change",
) -> List[Dict]:
    """Convert labeled regions to GeoJSON Feature dicts.

    Properties per feature: ``label``, ``pixels``, ``area`` (map units^2),
    ``centroid_x``/``centroid_y`` (map coords), ``bbox`` (map coords),
    ``kind`` (``"gain"``/``"loss"``/``"change"``), and, when
    ``change_grid`` is given, ``mean_change``, ``min_change``,
    ``max_change`` over the region's valid pixels.
    """
    features: List[Dict] = []
    for reg in regions:
        label_id = reg["label"]
        geom = region_polygon(labels, label_id, grid, simplify_tolerance)
        if geom is None:
            continue
        area = polygon_area(geom)
        r_c, c_c = reg["centroid"]
        cx, cy = grid.pixel_center(int(round(r_c)), int(round(c_c)))
        r0, r1, c0, c1 = reg["bbox"]
        x0, y0 = grid.xy(r0, c0)
        x1, y1 = grid.xy(r1 + 1, c1 + 1)
        props: Dict = {
            "label": label_id,
            "pixels": reg["pixels"],
            "area": area,
            "centroid_x": cx,
            "centroid_y": cy,
            "bbox": [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
            "kind": kind,
        }
        if change_grid is not None:
            vals = change_grid.data[labels == label_id]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                props["mean_change"] = float(vals.mean())
                props["min_change"] = float(vals.min())
                props["max_change"] = float(vals.max())
        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": props,
        })
    return features


def features_to_geojson(features: List[Dict], crs: Optional[str] = None) -> Dict:
    """Wrap features in a GeoJSON FeatureCollection dict."""
    fc: Dict = {"type": "FeatureCollection", "features": features}
    if crs:
        fc["crs"] = {"type": "name", "properties": {"name": crs}}
    return fc


def write_geojson(obj: Dict, path: str) -> str:
    """Write a GeoJSON dict to ``path``; returns the path."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    return path
