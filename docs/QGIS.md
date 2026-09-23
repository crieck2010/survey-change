# QGIS guide

Every `survey-change` product ships with a `.qml` sidecar so it opens in QGIS
already symbolized. This guide covers what you get and how to tune it.

## Opening the outputs

Drag any of these into QGIS — the same-named `.qml` is applied automatically:

- **`change_difference.tif`** — signed change (t2 − t1). Styled with a
  5-stop diverging ramp pinned at 0: deep blue = strong loss, white = no
  change, red = strong gain. The ramp range is set from the observed changed
  pixels, so contrast is always meaningful.
- **`change_mask.tif`** — binary mask. Transparent where unchanged, orange
  where changed (paletted renderer, values 0/1).
- **`change_polygons.geojson`** — vectorized change patches, categorized on
  the `kind` attribute: red = gain, blue = loss, orange = change
  (unsplit). 55% fill opacity with dark outlines.
- **`transition.tif`** — post-classification codes (`from*1000+to`), paletted
  with one color per from→to transition and human labels like
  `forest -> urban`.

## Tuning the difference ramp

`change.qgis.difference_raster_qml(vmin, vmax, no_change=0.0)` regenerates the
style. If QGIS shows the raster washed out, your scene's change range is wider
than interesting — regenerate with tighter bounds:

```python
from change import qgis
qgis.write_qml(qgis.difference_raster_qml(-0.3, 0.3), "change_difference.tif.qml")
```

Then right-click the layer in QGIS → Properties → Symbology → "Load Style".

## Working with the polygons

The GeoJSON attribute table includes `area` in CRS units² (m² for UTM),
`mean_change`/`min_change`/`max_change` per patch, and `centroid_x/y`. Useful
QGIS moves:

- **Label patches by area**: Labels → Single labels, value
  `round("area", 1) || ' m²'`.
- **Filter speckle**: Right-click → Filter… → `"pixels" >= 10`.
- **Select gain vs loss**: the `kind` field is already categorized; toggle
  categories in the Layers panel.
- **Zonal summaries**: the `_regions.csv` written next to the GeoJSON has the
  same rows — join it or load it directly for spreadsheet reporting.

## CRS notes

Outputs inherit the CRS of the inputs (`Grid.crs`, passed through from the
source GeoTIFFs). `survey-change` never reprojects — resample inputs to a
common grid/CRS first (e.g. with `gdalwarp` or `survey-raster`) so `detect()`
sees aligned grids. Areas are planar in CRS units; use a projected CRS (UTM,
State Plane) rather than EPSG:4326 when you need true m²/ha.
