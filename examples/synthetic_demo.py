"""Synthetic end-to-end demo: no satellite data needed.

Builds two fake NDVI rasters (a 100x100 scene with a cleared patch in t2),
runs the full survey-change pipeline, and writes QGIS-ready outputs to
./demo_out. Run with:  python examples/synthetic_demo.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from change import ChangeConfig, Grid, detection, io, polygons, qgis, segmentation, statistics

OUT = os.path.join(os.path.dirname(__file__), "demo_out")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(42)
base = np.clip(rng.normal(0.65, 0.08, (100, 100)), 0, 1)  # healthy vegetation
t2data = base.copy()
t2data[40:60, 30:55] = np.clip(rng.normal(0.18, 0.05, (20, 25)), 0, 1)  # clearing
t2data[70:75, 70:75] = np.clip(rng.normal(0.85, 0.03, (5, 5)), 0, 1)    # regrowth

transform = (500000.0, 10.0, 0.0, 4800000.0, 0.0, -10.0)  # UTM 10 m pixels
t1 = Grid(data=base, transform=transform, crs="EPSG:32617", name="NDVI t1")
t2 = Grid(data=t2data, transform=transform, crs="EPSG:32617", name="NDVI t2")

with open(os.path.join(os.path.dirname(__file__), "change_config.json")) as fh:
    cfg = ChangeConfig.from_dict(json.load(fh))

res = detection.detect(t1, t2, cfg)
print("threshold:", round(res.threshold_used, 4))
print("changed px:", res.stats["changed_pixels"],
      "| gain:", res.stats["gain_pixels"], "| loss:", res.stats["loss_pixels"])

io.write_cog(res.change_grid, os.path.join(OUT, "change_difference.tif"))
io.write_mask_geotiff(res.change_mask, res.change_grid, os.path.join(OUT, "change_mask.tif"))
qgis.write_qml(qgis.difference_raster_qml(-0.6, 0.6),
               os.path.join(OUT, "change_difference.tif.qml"))
qgis.write_qml(qgis.mask_qml(), os.path.join(OUT, "change_mask.tif.qml"))

labels, regions = segmentation.label_regions(res.change_mask)
feats = polygons.regions_to_features(labels, regions, t1, res.change_grid,
                                     simplify_tolerance=5.0, kind="change")
polygons.write_geojson(polygons.features_to_geojson(feats, crs=t1.crs),
                       os.path.join(OUT, "change_polygons.geojson"))
qgis.write_qml(qgis.change_polygon_qml(),
               os.path.join(OUT, "change_polygons.geojson.qml"))
table = statistics.region_table(regions, t1.pixel_area, res.change_grid, labels)
io.write_csv(table, os.path.join(OUT, "change_regions.csv"))
io.write_report(io.change_report(statistics.attach_stats(res).stats, table),
                os.path.join(OUT, "change_report.json"))

areas = sorted((f["properties"]["area"] for f in feats), reverse=True)
print(f"polygons: {len(feats)}, largest patch: {areas[0]:,.0f} m^2" if areas else "no polygons")
print("wrote", OUT)
