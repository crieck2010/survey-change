# survey-change

> **Part of [earthwatch-suite](https://github.com/crieck2010/earthwatch-suite)** —
> the remote-sensing project. Terrestrial surveying lives in
> [survey-suite](https://github.com/crieck2010/survey-suite); the two stay
> compatible through the
> [cross-suite contracts](https://github.com/crieck2010/earthwatch-suite/blob/main/docs/CONTRACTS.md).

Change-detection engine for surveying and remote sensing: compare two (or more)
raster acquisitions and turn the differences into decision-ready products —
change rasters, gain/loss masks, change polygons, transition matrices, and
QGIS-styled outputs.

`survey-change` consumes the outputs of `survey-imagery` (index COGs, monitor
`timeseries.csv` archives) and `survey-raster` (from the surveying side), and
its GeoJSON/QML outputs drop straight into QGIS.

## What it does

- **Raster change operators** — simple differencing, normalized difference,
  relative change (%), log-ratio, and multi-band **Change Vector Analysis**
  (magnitude + direction, so you can separate *kinds* of change)
- **Thresholding** — Otsu (automatic), manual, percentile, or none; separate
  gain/loss thresholds supported
- **Change regions** — connected-component labeling (4/8-connectivity) with
  pixel counts, bounding boxes, and centroids; tiny speckle regions filtered out
- **Change polygons** — regions vectorized to true polygons (holes preserved),
  simplified with Douglas-Peucker, exported as GeoJSON with area/centroid/
  per-region change statistics
- **Post-classification comparison** — from-to change matrix, transition
  raster (`from*1000+to` coding), and per-transition area tables for land-cover
  conversion questions ("how much forest became developed?")
- **Multi-epoch tracking** — sequential pair detection, cumulative net change,
  per-pixel change-onset dating, persistent-change masks, and break detection
  directly on `survey-imagery` monitor `timeseries.csv` files
- **QGIS-ready outputs** — QML styles for difference rasters (diverging
  blue-white-red), change polygons (categorized gain/loss), masks, and
  transition rasters, plus JSON layer sidecars

## Installation

```bash
pip install git+https://github.com/crieck2010/survey-change.git
```

Requires Python ≥ 3.9, `numpy`, and `rasterio`. No other runtime dependencies —
the engine is stdlib + numpy + rasterio, and sibling suite packages are
*optional* (lazy, duck-typed adapters; see Interoperability).

## Quick start

```python
from change import io, detection, ChangeConfig
from change import segmentation, polygons, statistics

t1 = io.read_grid("ndvi_2026-07-16.tif")
t2 = io.read_grid("ndvi_2026-09-20.tif")

cfg = ChangeConfig(method="difference", threshold_method="otsu",
                   min_region_pixels=4)
res = detection.detect(t1, t2, cfg)
print(res.stats)
# {'changed_pixels': 1284, 'gain_pixels': 96, 'loss_pixels': 1188, ...}

labels, regions = segmentation.label_regions(res.change_mask)
feats = polygons.regions_to_features(labels, regions, t1,
                                     change_grid=res.change_grid, kind="change")
polygons.write_geojson(polygons.features_to_geojson(feats, crs=t1.crs),
                       "change_polygons.geojson")

io.write_cog(res.change_grid, "change_difference.tif")
```

## CLI

```bash
# Detect change between two rasters -> COG + mask + QML + JSON report
survey-change detect ndvi_t1.tif ndvi_t2.tif --out-dir change_out \
    --method difference --threshold-method otsu

# Vectorize a change mask -> GeoJSON polygons + QML + region CSV
survey-change polygons change_out/change_mask.tif --out change_polygons.geojson \
    --change change_out/change_difference.tif --simplify 5.0

# Post-classification comparison of two label rasters
survey-change classify landcover_2020.tif landcover_2026.tif \
    --out-dir classify_out --class-names classes.json

# Find abrupt breaks in a survey-imagery monitor archive
survey-change timeseries monitor/timeseries.csv --index NDVI --threshold 0.15
```

## Interoperability

| Sibling | How survey-change consumes it |
|---|---|
| `survey-imagery` | Index COGs from the site monitor as `t1`/`t2`; `timeseries.csv` for break detection (`change.timeseries.read_imagery_timeseries`); in-memory `BandData` via `change.interop.grid_from_imagery_band` |
| `survey-raster` | Any object with `.data`/`.transform`/`.crs` via `change.interop.grid_from_raster_like` |
| `survey-cogo` / `survey-gnss` | Polygon vertex lists rasterized to AOI masks (`change.interop.rasterize_polygon`), then `clip_to_aoi` |
| QGIS | GeoJSON polygons, COGs, and `.qml` styles open directly — no manual styling |

See [docs/INTEROP.md](docs/INTEROP.md) for the full contract.

## Scaling

- All operators are O(pixels) single-pass numpy; memory is ~a few arrays per
  input — a 10k×10k scene processes in constant extra memory beyond the inputs.
- Region labeling is a two-pass union-find (no scipy needed); polygon tracing
  touches only boundary cells.
- For stacks larger than RAM, tile the AOI and run `detect` per tile — every
  function is pure (inputs never mutated), so tiles parallelize trivially with
  `concurrent.futures`.
- `min_region_pixels` and percentile thresholding keep polygon counts (and
  GeoJSON size) bounded on noisy scenes.

## API overview

| Module | Contents |
|---|---|
| `change.core` | `Grid`, `ChangeConfig`, `ChangeResult`, exceptions |
| `change.detection` | `difference`, `normalized_difference_change`, `relative_change`, `ratio_change`, `change_vector_analysis`, `otsu_threshold`, `apply_threshold`, `detect` |
| `change.segmentation` | `label_regions`, `filter_regions` |
| `change.polygons` | `region_polygon`, `douglas_peucker`, `regions_to_features`, `features_to_geojson`, `write_geojson` |
| `change.classification` | `post_classification_compare`, `transition_grid`, `summarize_transitions` |
| `change.statistics` | `summarize_change`, `histogram`, `region_table` |
| `change.timeseries` | `Epoch`, `sequential_changes`, `cumulative_change`, `change_onset`, `persistent_change_mask`, `read_imagery_timeseries`, `index_breaks` |
| `change.interop` | `grid_from_raster_like`, `grid_from_imagery_band`, `rasterize_polygon`, `clip_to_aoi` |
| `change.qgis` | QML generators (`difference_raster_qml`, `change_polygon_qml`, `mask_qml`, `transition_qml`), layer metadata |
| `change.io` | `read_grid`, `write_geotiff`, `write_cog`, `write_mask_geotiff`, `write_csv`, `write_report` |
| `change.licensing` | License-key and update-check hooks (stubs) |
| `change.cli` | `survey-change` command line |

Full reference: [docs/API.md](docs/API.md). Architecture notes:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Typical workflows

**Construction / mining disturbance monitoring.** Run the `survey-imagery`
site monitor per satellite pass, then `survey-change detect` on consecutive
NDVI or brightness COGs. `change_onset` dates each disturbed pixel; polygons
give you per-patch areas for reporting.

**Municipal impervious-surface / MS4 tracking.** `survey-change classify` on
yearly land-cover rasters produces the from-to matrix ("vegetation →
impervious: 12.4 ha") that stormwater reports need.

**Wetland / shoreline change.** CVA on NIR+red bands separates vegetation
loss from soil/water exposure by change direction; threshold the magnitude,
vectorize, and open the GeoJSON in QGIS.

## Versioning

Semantic versioning. See [CHANGELOG.md](CHANGELOG.md).

## License

MIT — see [LICENSE](LICENSE).
