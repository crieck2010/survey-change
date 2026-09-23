# Architecture

`survey-change` follows the suite's engine-first convention: a pure-logic
engine with zero UI-framework imports, a thin CLI, and dependency-light
packaging (`numpy` + `rasterio` only).

## Module dependency graph

```
cli ──> io, qgis, licensing
  │
  ├─> detection ──> core, segmentation
  ├─> segmentation ──> (numpy only)
  ├─> polygons ──> core
  ├─> classification ──> core
  ├─> statistics ──> core
  ├─> timeseries ──> core, detection
  ├─> interop ──> core
  └─> io, qgis, licensing ──> core (io also uses rasterio)
```

`core` is imported by everything and imports nothing but numpy. No cycles.

## Key design decisions

**Grid as the atomic unit.** Every algorithm takes `Grid` (2-D array +
6-tuple geotransform + CRS + nodata) and returns new `Grid`s. Inputs are
never mutated, which makes tiling/parallelism safe and tests trivial.

**NaN as the missing-data sentinel.** Nodata and non-finite pixels propagate
as NaN through every operator, so statistics and masks can ignore them with
a single `np.isfinite` check. Byte-mask I/O uses 0 = nodata on disk.

**Signed change with t2 − t1 semantics.** All difference-family operators
return signed grids where positive = gain, negative = loss. CVA magnitude is
the one unsigned operator (direction is carried by the angle grid).

**Thresholding on |change|.** One threshold splits "changed" from "unchanged";
gain/loss is decided by sign. Optional per-side overrides (`gain_threshold`,
`loss_threshold`) handle asymmetric cases (e.g. only clear-cuts matter).

**Pure-Python vectorization.** Boundary tracing + Douglas-Peucker are
implemented in the engine (no shapely) so installs stay light and polygons
are reproducible anywhere. Holes are preserved; the largest ring wins as
exterior.

**Lazy suite interop.** Sibling packages are never imported at module load.
Adapters accept duck-typed objects or file paths, so `survey-change` works
with zero suite dependencies installed and degrades to clear `ImportError`s.

**QGIS as a first-class target.** Every raster product gets a `.qml`
sidecar and every vector product is GeoJSON with a matching categorized
`.qml`, so outputs open in QGIS already symbolized — no per-pass styling.

## Scaling notes

- Operators are single-pass O(pixels) numpy; labeling is two-pass union-find.
- Memory per `detect` call is a small constant multiple of the input grids.
- For rasters larger than RAM: tile the AOI, run `detect` per tile (pure
  functions — no shared state), and merge masks/polygons afterwards.
- Polygon count (and GeoJSON size) is bounded by `min_region_pixels` and the
  threshold choice; prefer percentile/Otsu over `none` on noisy scenes.
- `change_onset` / `persistent_change_mask` stream over epochs one grid at a
  time — epoch count does not multiply memory.

## Extension points

- New operators: add a function in `detection.py` returning a `Grid`, then a
  branch in `detect()`.
- New threshold methods: extend `apply_threshold`.
- New output formats: add writers in `io.py` / `qgis.py`; the CLI stays thin.
