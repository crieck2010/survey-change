# Interoperability

`survey-change` is designed to sit downstream of the suite's acquisition
modules and upstream of QGIS. Nothing here requires any sibling package at
import time — adapters are lazy and duck-typed.

## survey-imagery → survey-change

The primary pipeline: `survey-imagery`'s site monitor writes per-pass index
COGs and a `timeseries.csv` archive; `survey-change` turns consecutive passes
into change products.

```python
from change import io, timeseries, detection, ChangeConfig

# 1. Raster path: index COGs straight into detect()
t1 = io.read_grid("monitor/NDVI/NDVI_2026-07-16.tif")
t2 = io.read_grid("monitor/NDVI/NDVI_2026-09-20.tif")
res = detection.detect(t1, t2, ChangeConfig(threshold_method="otsu"))

# 2. Tabular path: find WHEN something happened before spending time on rasters
rows = timeseries.read_imagery_timeseries("monitor/timeseries.csv")
for b in timeseries.index_breaks(rows, "NDVI", threshold=0.15):
    print(b["from_date"], "->", b["to_date"], f'{b["delta"]:+.3f}')

# 3. In-memory path: BandData objects without touching disk
from change.interop import grid_from_imagery_band
g1 = grid_from_imagery_band(band_t1)  # any .data/.transform/.crs object
```

`read_imagery_timeseries` validates the header (`date`, `index`, `mean`
required) and raises `ChangeError` on anything else.

## survey-raster → survey-change

```python
from change.interop import grid_from_raster_like
grid = grid_from_raster_like(raster_obj)          # .data/.transform/.crs
grid = grid_from_raster_like((data, transform, crs))  # or a plain tuple
```

Any object exposing those three attributes works — no import of
`survey_raster` needed.

## survey-cogo / survey-gnss → survey-change

Boundary or traverse vertices become an analysis mask:

```python
from change.interop import rasterize_polygon, clip_to_aoi
verts = [(x1, y1), (x2, y2), ...]          # from a cogo parcel or GNSS trace
aoi = rasterize_polygon(verts, t1)          # boolean mask on t1's grid
t1c = clip_to_aoi(t1, aoi)                  # outside pixels -> NaN
```

## survey-change → QGIS

| Product | File | Style |
|---|---|---|
| Signed change raster | `change_difference.tif` (COG) | `change_difference.tif.qml` — diverging blue-white-red |
| Change mask | `change_mask.tif` | `change_mask.tif.qml` — transparent/orange paletted |
| Change polygons | `change_polygons.geojson` | `change_polygons.geojson.qml` — categorized on `kind` |
| Transition raster | `transition.tif` (COG) | `transition.tif.qml` — paletted per from→to code |

GeoJSON features carry `label`, `pixels`, `area` (CRS units²), `centroid_x/y`,
`bbox`, `kind` (`gain`/`loss`/`change`), and per-region `mean/min/max_change`
when a change grid is supplied — everything an attribute table needs.

## survey-change → downstream code

All engine functions are pure and return plain numpy/dataclass/GeoJSON-dict
structures, so embedding in larger pipelines (e.g. a future `survey-qgis`
Processing plugin) needs no adaptation layer: call `detection.detect`,
`segmentation.label_regions`, `polygons.regions_to_features` directly.
