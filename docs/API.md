# survey-change API reference

Pure-logic engine: every function below takes plain data in and returns plain
data out. No file dialogs, no UI imports.

## change.core

- `Grid(data, transform, crs, nodata, name)` — 2-D raster + georeferencing.
  `.rows/.cols/.shape`, `.pixel_area`, `.bounds()`, `.valid_mask()`,
  `.aligned_with(other)`, `.assert_aligned(other)` (raises `AlignmentError`),
  `.xy(row, col)`, `.pixel_center(row, col)`, `.copy()`, `.to_dict()`.
- `ChangeConfig(method, threshold_method, threshold, percentile,
  gain_threshold, loss_threshold, min_region_pixels, connectivity,
  mask_nodata, name)` — JSON-serializable via `.to_dict()`/`.from_dict()`.
- `ChangeResult` — `change_grid`, `change_mask`, `gain_mask`, `loss_mask`,
  `threshold_used`, `stats`, `regions`, `config`.
- Exceptions: `ChangeError`, `AlignmentError`, `GridError`.

## change.detection

- `difference(t1, t2)` — signed `t2 - t1`.
- `normalized_difference_change(t1, t2)` — `(t2-t1)/(|t2|+|t1|)`, in [-1, 1].
- `relative_change(t1, t2)` — percent change vs `|t1|`.
- `ratio_change(t1, t2)` — `log(t2/t1)`, symmetric about 0.
- `change_vector_analysis(bands_t1, bands_t2)` — `(magnitude, angle)`;
  angle in degrees (-180, 180] in the plane of the first two bands.
- `otsu_threshold(values)` — Otsu on `|change|`.
- `apply_threshold(change, method, threshold, percentile)` →
  `(mask, used)`. Methods: `otsu`, `manual`, `percentile`, `none`.
- `gain_loss_masks(change, change_mask, gain_threshold, loss_threshold)`.
- `detect(t1, t2, config, bands_t1, bands_t2)` — full pass → `ChangeResult`.
  `config.method`: `difference` | `normalized_difference` |
  `relative_change` | `ratio` | `cva`.

## change.segmentation

- `label_regions(mask, connectivity=8)` → `(labels, regions)`. Regions carry
  `label`, `pixels`, `bbox` (inclusive row/col), `centroid`.
- `filter_regions(labels, regions, min_pixels, max_pixels)`.

## change.polygons

- `region_polygon(labels, label_id, grid, simplify_tolerance)` → GeoJSON
  Polygon dict (holes preserved) or None.
- `douglas_peucker(ring, tolerance)`.
- `polygon_area(geometry)` — planar area, CRS units².
- `regions_to_features(labels, regions, grid, change_grid, simplify_tolerance, kind)`
  → GeoJSON Feature dicts with `label/pixels/area/centroid_x/centroid_y/bbox/
  kind` (+ `mean/min/max_change` when a change grid is given).
- `features_to_geojson(features, crs)`, `write_geojson(obj, path)`.

## change.classification

- `post_classification_compare(labels_t1, labels_t2, class_names)` →
  `{matrix, labels, changed_pixels, total_valid_pixels, changed_fraction,
  transitions, class_names}`.
- `transition_grid(labels_t1, labels_t2)` — codes `from*1000+to`.
- `summarize_transitions(comparison, pixel_area, class_names)` → CSV-ready rows.
- `stable_mask(labels_t1, labels_t2)`.

## change.statistics

- `summarize_change(change, change_mask, gain_mask, loss_mask)` — counts,
  areas, mean/min/max/std with gain/loss breakdowns.
- `histogram(change, change_mask, bins)`.
- `region_table(regions, pixel_area, change, labels)` — CSV-ready rows.
- `attach_stats(result)`.

## change.timeseries

- `Epoch(date, grid)`.
- `sequential_changes(epochs, config)` — per-pair results, date-sorted.
- `cumulative_change(epochs)` — last − first.
- `change_onset(epochs, threshold)` — per-pixel first-exceedance epoch index.
- `persistent_change_mask(epochs, threshold, min_epochs)`.
- `read_imagery_timeseries(path)` — survey-imagery monitor CSV → row dicts.
- `index_breaks(rows, index, threshold)` — abrupt mean-value steps.

## change.interop

- `grid_from_raster_like(obj)` — `.data/.transform/.crs` duck-typing or
  `(data, transform, crs)` tuple.
- `grid_from_imagery_band(band, name)` — survey-imagery `BandData`-likes
  (tuple or Affine transforms accepted).
- `rasterize_polygon(vertices, grid)` — `[(x, y), ...]` → boolean AOI mask.
- `clip_to_aoi(grid, aoi_mask)` — outside → NaN.
- `describe_suite_inputs()`.

## change.qgis

- `difference_raster_qml(vmin, vmax, no_change=0.0)`
- `change_polygon_qml()` — categorized on `kind`.
- `mask_qml()` — paletted 0/1.
- `transition_qml(class_names, palette)`.
- `write_qml(qml, path)`, `layer_metadata(...)`, `write_layer_metadata(...)`.

## change.io

- `read_grid(path, band, name)` → `Grid` (nodata → NaN).
- `write_geotiff(grid, path, dtype, compress, tiled)`.
- `write_cog(grid, path, dtype)` — COG driver with GTiff+overviews fallback.
- `write_mask_geotiff(mask, template, path)`.
- `write_csv(records, path, fieldnames)`, `read_geojson(path)`.
- `change_report(stats, regions)`, `write_report(report, path)`
  (numpy-safe JSON).

## change.licensing

- `check_license(key=None)` → `{valid, tier, message}` (stub: community tier).
- `check_for_updates(timeout)` → `{available, current, latest, url}`; never raises.

## change.cli

`survey-change detect|polygons|classify|timeseries|license|update-check`
— thin wrappers; see `survey-change --help`.
