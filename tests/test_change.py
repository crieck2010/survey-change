"""Unit tests for survey-change. Run with: python -m unittest discover -s tests"""
import csv
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import change
from change import ChangeConfig, ChangeError, Grid
from change import classification as cls_mod
from change import detection as det_mod
from change import interop as interop_mod
from change import io as io_mod
from change import licensing as lic_mod
from change import polygons as poly_mod
from change import qgis as qgis_mod
from change import segmentation as seg_mod
from change import statistics as stats_mod
from change import timeseries as ts_mod
from change.cli import main as cli_main


def _grid(data, transform=(0.0, 10.0, 0.0, 100.0, 0.0, -10.0),
          crs="EPSG:32617", name=""):
    return Grid(data=np.asarray(data, dtype=float), transform=transform,
                crs=crs, name=name)


def _temp_tif(data, transform=(0.0, 10.0, 0.0, 100.0, 0.0, -10.0), crs="EPSG:32617"):
    import rasterio
    from rasterio.transform import Affine
    tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    tmp.close()
    arr = np.asarray(data, dtype=np.float32)
    with rasterio.open(tmp.name, "w", driver="GTiff", height=arr.shape[0],
                       width=arr.shape[1], count=1, dtype="float32",
                       crs=crs, transform=Affine(*transform)) as dst:
        dst.write(arr, 1)
    return tmp.name


class TestCore(unittest.TestCase):
    def test_grid_basics(self):
        g = _grid(np.ones((4, 5)))
        self.assertEqual(g.shape, (4, 5))
        self.assertEqual(g.rows, 4)
        self.assertEqual(g.cols, 5)
        self.assertEqual(g.pixel_width, 10.0)
        self.assertEqual(g.pixel_height, 10.0)
        self.assertEqual(g.pixel_area, 100.0)

    def test_grid_rejects_3d(self):
        with self.assertRaises(change.GridError):
            _grid(np.ones((2, 2, 2)))

    def test_valid_mask(self):
        g = Grid(data=np.array([[1.0, np.nan], [2.0, -999.0]]), nodata=-999.0)
        np.testing.assert_array_equal(g.valid_mask(), [[True, False], [True, False]])

    def test_alignment(self):
        g1 = _grid(np.ones((3, 3)))
        g2 = _grid(np.ones((3, 3)))
        self.assertTrue(g1.aligned_with(g2))
        g3 = _grid(np.ones((3, 4)))
        self.assertFalse(g1.aligned_with(g3))
        with self.assertRaises(change.AlignmentError):
            g1.assert_aligned(g3)
        g4 = _grid(np.ones((3, 3)), crs="EPSG:4326")
        self.assertFalse(g1.aligned_with(g4))
        self.assertTrue(g1.aligned_with(g4, check_crs=False))

    def test_bounds_and_centers(self):
        g = _grid(np.ones((2, 2)), transform=(0.0, 10.0, 0.0, 100.0, 0.0, -10.0))
        self.assertEqual(g.bounds(), (0.0, 80.0, 20.0, 100.0))
        self.assertEqual(g.xy(0, 0), (0.0, 100.0))
        self.assertEqual(g.pixel_center(0, 0), (5.0, 95.0))

    def test_config_roundtrip(self):
        cfg = ChangeConfig(method="cva", threshold_method="manual", threshold=0.5,
                           min_region_pixels=4, name="x")
        cfg2 = ChangeConfig.from_dict(cfg.to_dict())
        self.assertEqual(cfg, cfg2)
        self.assertEqual(ChangeConfig.from_dict({"method": "ratio", "bogus": 1}).method, "ratio")


class TestDetection(unittest.TestCase):
    def test_difference(self):
        t1 = _grid([[1.0, 2.0], [3.0, np.nan]])
        t2 = _grid([[2.0, 2.0], [1.0, 5.0]])
        d = det_mod.difference(t1, t2)
        np.testing.assert_allclose(d.data, [[1.0, 0.0], [-2.0, np.nan]], equal_nan=True)
        self.assertTrue(np.isnan(d.nodata))

    def test_misaligned_raises(self):
        with self.assertRaises(change.AlignmentError):
            det_mod.difference(_grid(np.ones((2, 2))), _grid(np.ones((3, 3))))

    def test_normalized_difference_bounded(self):
        t1 = _grid([[1.0, 0.0], [2.0, 2.0]])
        t2 = _grid([[3.0, 5.0], [2.0, 0.0]])
        nd = det_mod.normalized_difference_change(t1, t2)
        self.assertTrue(np.nanmax(np.abs(nd.data)) <= 1.0 + 1e-12)
        self.assertAlmostEqual(nd.data[0, 0], 0.5)

    def test_relative_change(self):
        rc = det_mod.relative_change(_grid([[2.0]]), _grid([[3.0]]))
        self.assertAlmostEqual(rc.data[0, 0], 50.0)
        rc0 = det_mod.relative_change(_grid([[0.0]]), _grid([[3.0]]))
        self.assertTrue(np.isnan(rc0.data[0, 0]))

    def test_ratio_change(self):
        r = det_mod.ratio_change(_grid([[1.0]]), _grid([[np.e]]))
        self.assertAlmostEqual(r.data[0, 0], 1.0)
        r0 = det_mod.ratio_change(_grid([[0.0]]), _grid([[1.0]]))
        self.assertTrue(np.isnan(r0.data[0, 0]))

    def test_cva(self):
        b1 = [_grid(np.zeros((2, 2))), _grid(np.zeros((2, 2)))]
        b2 = [_grid(np.full((2, 2), 3.0)), _grid(np.full((2, 2), 4.0))]
        mag, ang = det_mod.change_vector_analysis(b1, b2)
        np.testing.assert_allclose(mag.data, 5.0)
        np.testing.assert_allclose(ang.data, np.degrees(np.arctan2(4, 3)))
        with self.assertRaises(ChangeError):
            det_mod.change_vector_analysis(b1[:1], b2[:1])

    def test_otsu(self):
        v = np.concatenate([np.zeros(90), np.ones(10)])
        t = det_mod.otsu_threshold(v)
        self.assertGreater(t, 0.0)
        self.assertLess(t, 1.0)
        with self.assertRaises(ChangeError):
            det_mod.otsu_threshold(np.array([np.nan]))

    def test_apply_threshold_methods(self):
        g = _grid([[0.1, 0.5], [-0.6, 0.0]])
        m, used = det_mod.apply_threshold(g, method="manual", threshold=0.5)
        np.testing.assert_array_equal(m, [[False, True], [True, False]])
        self.assertEqual(used, 0.5)
        m2, used2 = det_mod.apply_threshold(g, method="percentile", percentile=50)
        self.assertIsNotNone(used2)
        m3, used3 = det_mod.apply_threshold(g, method="none")
        self.assertIsNone(used3)
        self.assertTrue(m3[0, 0])
        with self.assertRaises(ChangeError):
            det_mod.apply_threshold(g, method="manual")
        with self.assertRaises(ChangeError):
            det_mod.apply_threshold(g, method="bogus")

    def test_gain_loss_masks(self):
        g = _grid([[0.5, -0.7], [0.1, 0.0]])
        mask = np.ones((2, 2), dtype=bool)
        gain, loss = det_mod.gain_loss_masks(g, mask)
        np.testing.assert_array_equal(gain, [[True, False], [True, False]])
        np.testing.assert_array_equal(loss, [[False, True], [False, False]])
        gain2, _ = det_mod.gain_loss_masks(g, mask, gain_threshold=0.6)
        self.assertFalse(gain2.any())

    def test_detect_end_to_end(self):
        t1 = _grid(np.zeros((10, 10)))
        t2 = _grid(np.zeros((10, 10)))
        t2.data[4:7, 4:7] = 1.0
        cfg = ChangeConfig(method="difference", threshold_method="otsu")
        res = det_mod.detect(t1, t2, cfg)
        self.assertEqual(res.stats["changed_pixels"], 9)
        self.assertEqual(res.stats["gain_pixels"], 9)
        self.assertEqual(res.stats["loss_pixels"], 0)
        self.assertIsNotNone(res.threshold_used)

    def test_detect_min_region_pixels(self):
        t1 = _grid(np.zeros((10, 10)))
        t2 = _grid(np.zeros((10, 10)))
        t2.data[0, 0] = 5.0  # single-pixel speckle
        t2.data[5:8, 5:8] = 5.0
        cfg = ChangeConfig(threshold_method="manual", threshold=1.0,
                           min_region_pixels=4)
        res = det_mod.detect(t1, t2, cfg)
        self.assertEqual(res.stats["changed_pixels"], 9)

    def test_detect_bad_method(self):
        with self.assertRaises(ChangeError):
            det_mod.detect(_grid(np.zeros((3, 3))), _grid(np.zeros((3, 3))),
                           ChangeConfig(method="bogus"))


class TestSegmentation(unittest.TestCase):
    def test_label_8_vs_4(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 0] = mask[1, 1] = True  # diagonal touch only
        _, r8 = seg_mod.label_regions(mask, connectivity=8)
        _, r4 = seg_mod.label_regions(mask, connectivity=4)
        self.assertEqual(len(r8), 1)
        self.assertEqual(len(r4), 2)

    def test_region_stats(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:3, 2:5] = True  # 2 rows x 3 cols
        labels, regions = seg_mod.label_regions(mask)
        self.assertEqual(len(regions), 1)
        r = regions[0]
        self.assertEqual(r["pixels"], 6)
        self.assertEqual(r["bbox"], (1, 2, 2, 4))
        self.assertAlmostEqual(r["centroid"][0], 1.5)
        self.assertAlmostEqual(r["centroid"][1], 3.0)
        self.assertEqual(labels[1, 2], 1)
        self.assertEqual(labels[0, 0], 0)

    def test_filter_regions(self):
        mask = np.zeros((6, 6), dtype=bool)
        mask[0, 0] = True
        mask[3:6, 3:6] = True
        labels, regions = seg_mod.label_regions(mask)
        out, kept = seg_mod.filter_regions(labels, regions, min_pixels=2)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["pixels"], 9)
        self.assertEqual(out.max(), 1)


class TestPolygons(unittest.TestCase):
    def _square_grid(self):
        return _grid(np.zeros((6, 6)), transform=(0.0, 1.0, 0.0, 6.0, 0.0, -1.0),
                     crs="EPSG:3857")

    def test_square_polygon_area(self):
        labels = np.zeros((6, 6), dtype=np.int32)
        labels[2:4, 2:4] = 1  # 2x2 block
        geom = poly_mod.region_polygon(labels, 1, self._square_grid())
        self.assertEqual(geom["type"], "Polygon")
        self.assertAlmostEqual(poly_mod.polygon_area(geom), 4.0)

    def test_donut_hole(self):
        labels = np.zeros((7, 7), dtype=np.int32)
        labels[1:6, 1:6] = 1
        labels[2:5, 2:5] = 0  # 3x3 hole in 5x5 block
        labels2 = np.zeros((7, 7), dtype=np.int32)
        labels2[labels == 1] = 1
        # relabel ring as one region
        lab, _ = seg_mod.label_regions(labels == 1)
        geom = poly_mod.region_polygon(lab, 1, self._square_grid())
        self.assertEqual(len(geom["coordinates"]), 2)  # exterior + hole
        self.assertAlmostEqual(poly_mod.polygon_area(geom), 25.0 - 9.0)

    def test_douglas_peucker(self):
        ring = [(0.0, 0.0), (1.0, 0.01), (2.0, -0.01), (3.0, 0.0), (3.0, 1.0),
                (0.0, 1.0), (0.0, 0.0)]
        simp = poly_mod.douglas_peucker(ring, 0.5)
        self.assertLess(len(simp), len(ring))
        self.assertEqual(simp[0], simp[-1])

    def test_regions_to_features(self):
        mask = np.zeros((6, 6), dtype=bool)
        mask[1:3, 1:3] = True
        labels, regions = seg_mod.label_regions(mask)
        chg = _grid(np.full((6, 6), 2.0), transform=(0.0, 1.0, 0.0, 6.0, 0.0, -1.0))
        feats = poly_mod.regions_to_features(labels, regions, self._square_grid(),
                                             change_grid=chg, kind="gain")
        self.assertEqual(len(feats), 1)
        props = feats[0]["properties"]
        self.assertEqual(props["pixels"], 4)
        self.assertAlmostEqual(props["area"], 4.0)
        self.assertEqual(props["kind"], "gain")
        self.assertAlmostEqual(props["mean_change"], 2.0)
        fc = poly_mod.features_to_geojson(feats, crs="EPSG:3857")
        self.assertEqual(fc["type"], "FeatureCollection")

    def test_write_geojson_roundtrip(self):
        fc = {"type": "FeatureCollection", "features": []}
        with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
            path = tmp.name
        try:
            poly_mod.write_geojson(fc, path)
            self.assertEqual(io_mod.read_geojson(path), fc)
        finally:
            os.unlink(path)


class TestClassification(unittest.TestCase):
    def test_compare(self):
        t1 = Grid(data=np.array([[1, 1], [2, 2]]), keep_dtype=True)
        t2 = Grid(data=np.array([[1, 2], [2, 3]]), keep_dtype=True)
        comp = cls_mod.post_classification_compare(t1, t2, {1: "forest", 2: "field", 3: "urban"})
        self.assertEqual(comp["changed_pixels"], 2)
        self.assertEqual(comp["total_valid_pixels"], 4)
        self.assertAlmostEqual(comp["changed_fraction"], 0.5)
        self.assertEqual(comp["matrix"][1][2], 1)
        self.assertEqual(comp["matrix"][2][3], 1)
        trans = {(a, b) for a, b, _ in comp["transitions"]}
        self.assertEqual(trans, {(1, 2), (2, 3)})

    def test_transition_grid(self):
        t1 = Grid(data=np.array([[1, 2]]), keep_dtype=True)
        t2 = Grid(data=np.array([[1, 1]]), keep_dtype=True)
        tg = cls_mod.transition_grid(t1, t2)
        np.testing.assert_array_equal(tg.data, [[1001, 2001]])

    def test_summarize_transitions(self):
        t1 = Grid(data=np.array([[1, 1], [1, 1]]), keep_dtype=True,
                  transform=(0.0, 10.0, 0.0, 20.0, 0.0, -10.0))
        t2 = Grid(data=np.array([[1, 2], [1, 2]]), keep_dtype=True,
                  transform=(0.0, 10.0, 0.0, 20.0, 0.0, -10.0))
        comp = cls_mod.post_classification_compare(t1, t2)
        recs = cls_mod.summarize_transitions(comp, t1.pixel_area, {1: "a", 2: "b"})
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["area"], 2 * 100.0)
        self.assertEqual(recs[0]["from_name"], "a")

    def test_stable_mask(self):
        t1 = Grid(data=np.array([[1, 2]]), keep_dtype=True)
        t2 = Grid(data=np.array([[1, 1]]), keep_dtype=True)
        np.testing.assert_array_equal(cls_mod.stable_mask(t1, t2), [[True, False]])


class TestStatistics(unittest.TestCase):
    def test_summarize_change(self):
        chg = _grid([[1.0, -2.0], [0.0, np.nan]])
        mask = np.array([[True, True], [False, False]])
        s = stats_mod.summarize_change(chg, mask, np.array([[True, False], [False, False]]),
                                       np.array([[False, True], [False, False]]))
        self.assertEqual(s["pixels"], 2)
        self.assertEqual(s["area"], 2 * 100.0)
        self.assertAlmostEqual(s["mean_change"], -0.5)
        self.assertEqual(s["gain"]["pixels"], 1)
        self.assertEqual(s["loss"]["pixels"], 1)

    def test_summarize_empty(self):
        chg = _grid(np.zeros((2, 2)))
        s = stats_mod.summarize_change(chg, np.zeros((2, 2), dtype=bool))
        self.assertEqual(s["pixels"], 0)
        self.assertIsNone(s["mean_change"])

    def test_histogram(self):
        chg = _grid([[1.0, 2.0], [3.0, 4.0]])
        h = stats_mod.histogram(chg, np.ones((2, 2), dtype=bool), bins=4)
        self.assertEqual(sum(h["counts"]), 4)
        self.assertEqual(len(h["edges"]), 5)

    def test_region_table(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[0:2, 0:2] = True
        labels, regions = seg_mod.label_regions(mask)
        rows = stats_mod.region_table(regions, 100.0)
        self.assertEqual(rows[0]["area"], 400.0)


class TestTimeseries(unittest.TestCase):
    def _epochs(self):
        g0 = _grid(np.zeros((3, 3)))
        g1 = _grid(np.zeros((3, 3)))
        g1.data[1, 1] = 1.0
        g2 = _grid(np.zeros((3, 3)))
        g2.data[1, 1] = 2.0
        return [ts_mod.Epoch("2026-07-01", g0), ts_mod.Epoch("2026-08-01", g1),
                ts_mod.Epoch("2026-09-01", g2)]

    def test_sequential(self):
        out = ts_mod.sequential_changes(self._epochs(),
                                        ChangeConfig(threshold_method="manual", threshold=0.5))
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["from_date"], "2026-07-01")
        self.assertEqual(out[1]["to_date"], "2026-09-01")
        self.assertEqual(out[0]["stats"]["changed_pixels"], 1)

    def test_cumulative(self):
        cum = ts_mod.cumulative_change(self._epochs())
        self.assertAlmostEqual(cum.data[1, 1], 2.0)

    def test_onset(self):
        onset = ts_mod.change_onset(self._epochs(), threshold=0.5)
        self.assertEqual(onset.data[1, 1], 1)
        self.assertEqual(onset.data[0, 0], 0)

    def test_persistent(self):
        m = ts_mod.persistent_change_mask(self._epochs(), threshold=0.5, min_epochs=2)
        self.assertTrue(m[1, 1])
        self.assertFalse(m[0, 0])

    def test_imagery_timeseries(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w",
                                         newline="") as tmp:
            w = csv.DictWriter(tmp, fieldnames=["scene_id", "date", "index", "mean"])
            w.writeheader()
            w.writerow({"scene_id": "a", "date": "2026-07-01", "index": "NDVI", "mean": "0.6"})
            w.writerow({"scene_id": "b", "date": "2026-08-01", "index": "NDVI", "mean": "0.2"})
            w.writerow({"scene_id": "c", "date": "2026-08-01", "index": "NDWI", "mean": "0.1"})
            path = tmp.name
        try:
            rows = ts_mod.read_imagery_timeseries(path)
            self.assertEqual(len(rows), 3)
            self.assertIsInstance(rows[0]["mean"], float)
            br = ts_mod.index_breaks(rows, "NDVI", threshold=0.15)
            self.assertEqual(len(br), 1)
            self.assertAlmostEqual(br[0]["delta"], -0.4)
            self.assertEqual(ts_mod.index_breaks(rows, "NDWI", 0.15), [])
        finally:
            os.unlink(path)

    def test_bad_csv(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
            tmp.write("foo,bar\n1,2\n")
            path = tmp.name
        try:
            with self.assertRaises(ChangeError):
                ts_mod.read_imagery_timeseries(path)
        finally:
            os.unlink(path)


class TestInterop(unittest.TestCase):
    def test_raster_like_object(self):
        class Fake:
            data = np.ones((2, 2))
            transform = (0.0, 1.0, 0.0, 2.0, 0.0, -1.0)
            crs = "EPSG:4326"
            nodata = -1
            name = "fake"
        g = interop_mod.grid_from_raster_like(Fake())
        self.assertEqual(g.shape, (2, 2))
        self.assertEqual(g.crs, "EPSG:4326")
        self.assertEqual(g.nodata, -1)
        g2 = interop_mod.grid_from_raster_like((np.ones((2, 2)), (0, 1, 0, 0, 0, -1), None))
        self.assertEqual(g2.shape, (2, 2))
        with self.assertRaises(ChangeError):
            interop_mod.grid_from_raster_like(object())

    def test_imagery_band(self):
        class AffineLike:
            a, b, c, d, e, f = 0.0, 10.0, 0.0, 100.0, 0.0, -10.0
        class Band:
            data = np.ones((2, 2))
            transform = AffineLike()
            crs = "EPSG:32617"
        g = interop_mod.grid_from_imagery_band(Band(), name="NDVI")
        self.assertEqual(g.transform, (0.0, 10.0, 0.0, 100.0, 0.0, -10.0))
        self.assertEqual(g.name, "NDVI")

    def test_rasterize_polygon(self):
        grid = _grid(np.zeros((10, 10)), transform=(0.0, 1.0, 0.0, 10.0, 0.0, -1.0))
        verts = [(2.0, 4.0), (5.0, 4.0), (5.0, 7.0), (2.0, 7.0)]
        mask = interop_mod.rasterize_polygon(verts, grid)
        # x in [2,5) -> cols 2,3,4 ; y in (4,7] -> rows 3,4,5
        self.assertEqual(mask.sum(), 9)
        self.assertTrue(mask[4, 3])
        self.assertFalse(mask[0, 0])

    def test_clip_to_aoi(self):
        g = _grid(np.ones((3, 3)))
        aoi = np.zeros((3, 3), dtype=bool)
        aoi[1, 1] = True
        clipped = interop_mod.clip_to_aoi(g, aoi)
        self.assertTrue(np.isnan(clipped.data[0, 0]))
        self.assertEqual(clipped.data[1, 1], 1.0)
        with self.assertRaises(ChangeError):
            interop_mod.clip_to_aoi(g, np.zeros((2, 2), dtype=bool))

    def test_describe(self):
        d = interop_mod.describe_suite_inputs()
        self.assertIn("survey-imagery", d)


class TestQgis(unittest.TestCase):
    def test_difference_qml(self):
        qml = qgis_mod.difference_raster_qml(-0.5, 0.5)
        self.assertIn("singlebandpseudocolor", qml)
        self.assertIn("255,255,255,255", qml)

    def test_polygon_qml(self):
        qml = qgis_mod.change_polygon_qml()
        self.assertIn("categorizedSymbol", qml)
        self.assertIn('attr="kind"', qml)

    def test_mask_qml(self):
        self.assertIn("paletted", qgis_mod.mask_qml())

    def test_transition_qml(self):
        qml = qgis_mod.transition_qml({1: "forest", 2: "urban"})
        self.assertIn("1002", qml)  # forest -> urban code
        self.assertIn("2001", qml)  # urban -> forest code

    def test_write_qml_and_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            p = qgis_mod.write_qml(qgis_mod.mask_qml(), os.path.join(td, "m.qml"))
            self.assertTrue(open(p).read().startswith("<!DOCTYPE qgis"))
            meta = qgis_mod.layer_metadata("chg", "difference_raster", crs="EPSG:32617")
            mp = qgis_mod.write_layer_metadata(meta, os.path.join(td, "m.json"))
            self.assertEqual(json.load(open(mp))["produced_by"], "survey-change")


class TestIO(unittest.TestCase):
    def test_geotiff_roundtrip(self):
        g = _grid(np.arange(12, dtype=float).reshape(3, 4), name="t")
        path = _temp_tif(np.zeros((3, 4)))  # placeholder replaced below
        os.unlink(path)
        try:
            io_mod.write_geotiff(g, path)
            back = io_mod.read_grid(path)
            np.testing.assert_allclose(back.data, g.data)
            self.assertEqual(back.crs, "EPSG:32617")
            self.assertEqual(back.transform, g.transform)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_cog_roundtrip(self):
        g = _grid(np.ones((16, 16)))
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            path = tmp.name
        try:
            io_mod.write_cog(g, path)
            back = io_mod.read_grid(path)
            np.testing.assert_allclose(back.data, 1.0)
        finally:
            os.unlink(path)

    def test_mask_geotiff(self):
        g = _grid(np.zeros((4, 4)))
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            path = tmp.name
        try:
            io_mod.write_mask_geotiff(np.eye(4, dtype=bool), g, path)
            back = io_mod.read_grid(path)
            self.assertEqual(np.nansum(back.data), 4)  # 0 == nodata -> NaN on read
        finally:
            os.unlink(path)

    def test_csv(self):
        recs = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            path = tmp.name
        try:
            io_mod.write_csv(recs, path)
            with open(path) as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(rows[1]["b"], "y")
            with self.assertRaises(ChangeError):
                io_mod.write_csv([], path)
        finally:
            os.unlink(path)

    def test_write_report_numpy(self):
        rep = io_mod.change_report({"v": np.float64(1.5), "n": np.int64(3)})
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            path = tmp.name
        try:
            io_mod.write_report(rep, path)
            loaded = json.load(open(path))
            self.assertEqual(loaded["summary"]["n"], 3)
        finally:
            os.unlink(path)


class TestLicensing(unittest.TestCase):
    def test_check_license(self):
        r = lic_mod.check_license()
        self.assertTrue(r["valid"])
        r2 = lic_mod.check_license("ABC-123")
        self.assertEqual(r2["tier"], "licensed")

    def test_update_check_newer(self):
        fake = mock.Mock()
        fake.read.return_value = json.dumps(
            {"tag_name": "v9.9.9", "html_url": "http://x"}).encode()
        ctx = mock.MagicMock()
        ctx.__enter__.return_value = fake
        with mock.patch.object(lic_mod.urllib.request, "urlopen", return_value=ctx):
            r = lic_mod.check_for_updates()
        self.assertTrue(r["available"])
        self.assertEqual(r["latest"], "9.9.9")

    def test_update_check_offline(self):
        with mock.patch.object(lic_mod.urllib.request, "urlopen",
                               side_effect=OSError("no net")):
            r = lic_mod.check_for_updates()
        self.assertFalse(r["available"])
        self.assertIn("error", r)


class TestCLI(unittest.TestCase):
    def test_detect_cli(self):
        p1 = _temp_tif(np.zeros((8, 8)))
        arr = np.zeros((8, 8), dtype=np.float32)
        arr[2:5, 2:5] = 1.0
        p2 = _temp_tif(arr)
        try:
            with tempfile.TemporaryDirectory() as td:
                rc = cli_main(["detect", p1, p2, "--out-dir", td,
                               "--threshold-method", "manual", "--threshold", "0.5"])
                self.assertEqual(rc, 0)
                self.assertTrue(os.path.exists(os.path.join(td, "change_difference.tif")))
                self.assertTrue(os.path.exists(os.path.join(td, "change_mask.tif")))
                self.assertTrue(os.path.exists(os.path.join(td, "change_difference.tif.qml")))
                self.assertTrue(os.path.exists(os.path.join(td, "change_report.json")))
        finally:
            os.unlink(p1)
            os.unlink(p2)

    def test_polygons_cli(self):
        mask = np.zeros((8, 8), dtype=np.float32)
        mask[2:5, 2:5] = 1.0
        p = _temp_tif(mask)
        try:
            with tempfile.TemporaryDirectory() as td:
                out = os.path.join(td, "poly.geojson")
                rc = cli_main(["polygons", p, "--out", out, "--simplify", "0.5"])
                self.assertEqual(rc, 0)
                fc = json.load(open(out))
                self.assertEqual(len(fc["features"]), 1)
                self.assertTrue(os.path.exists(out + ".qml"))
        finally:
            os.unlink(p)

    def test_classify_cli(self):
        p1 = _temp_tif(np.array([[1, 1], [2, 2]], dtype=np.float32))
        p2 = _temp_tif(np.array([[1, 2], [2, 2]], dtype=np.float32))
        try:
            with tempfile.TemporaryDirectory() as td:
                rc = cli_main(["classify", p1, p2, "--out-dir", td])
                self.assertEqual(rc, 0)
                self.assertTrue(os.path.exists(os.path.join(td, "transitions.csv")))
        finally:
            os.unlink(p1)
            os.unlink(p2)

    def test_timeseries_cli(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w",
                                         newline="") as tmp:
            w = csv.DictWriter(tmp, fieldnames=["date", "index", "mean"])
            w.writeheader()
            w.writerow({"date": "2026-07-01", "index": "NDVI", "mean": "0.6"})
            w.writerow({"date": "2026-08-01", "index": "NDVI", "mean": "0.1"})
            path = tmp.name
        try:
            rc = cli_main(["timeseries", path, "--index", "NDVI", "--threshold", "0.2"])
            self.assertEqual(rc, 0)
        finally:
            os.unlink(path)

    def test_license_cli(self):
        self.assertEqual(cli_main(["license"]), 0)


if __name__ == "__main__":
    unittest.main()
