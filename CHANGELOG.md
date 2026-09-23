# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-23

### Added
- Core engine: `Grid` (data + geotransform + CRS + nodata), `ChangeConfig`
  (JSON round-trip), `ChangeResult`, `AlignmentError`/`ChangeError`/`GridError`.
- Change operators: simple differencing, normalized difference, relative
  change (%), log-ratio, and multi-band Change Vector Analysis
  (magnitude + direction angle).
- Thresholding: Otsu (automatic), manual, percentile, none; separate
  gain/loss threshold overrides.
- `detect()`: full two-date detection pass with tiny-region filtering and
  summary stats.
- Connected-component labeling (4/8-connectivity, union-find, no scipy)
  with region pixel counts, bounding boxes, and centroids; `filter_regions`.
- Polygon vectorization: direction-agnostic boundary tracing with hole
  support, Douglas-Peucker simplification, GeoJSON export with
  area/centroid/per-region change statistics.
- Post-classification comparison: sparse from-to change matrix, transition
  raster (`from*1000+to`), transition area tables, stable-pixel mask.
- Multi-epoch tracking: sequential pair detection, cumulative net change,
  per-pixel change-onset dating, persistent-change masks.
- survey-imagery interop: `read_imagery_timeseries` for monitor
  `timeseries.csv` archives and `index_breaks` abrupt-break detection.
- Suite adapters (lazy, duck-typed, zero hard deps): survey-raster-like
  objects, survey-imagery `BandData`, survey-cogo/gnss polygon vertex lists
  rasterized to AOI masks, `clip_to_aoi`.
- QGIS outputs: QML styles for difference rasters (diverging blue-white-red),
  change polygons (categorized gain/loss), binary masks, and transition
  rasters; JSON layer-metadata sidecars.
- I/O: GeoTIFF/COG read/write via rasterio, byte mask writing, CSV and JSON
  report writers (numpy-safe).
- CLI: `survey-change detect|polygons|classify|timeseries|license|update-check`.
- License-key and update-check hook stubs (`change.licensing`).
- Docs: README, API reference, architecture, interoperability, and QGIS guides;
  example config and synthetic-data demo script.
- Test suite: 63 unittest tests, all passing.
