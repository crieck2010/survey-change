"""Thin command-line interface for survey-change.

Every subcommand is a thin wrapper over the pure engine: argument parsing
lives here, and all computation lives in the engine modules. Run
``survey-change --help`` or ``python -m change --help`` for usage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .core import ChangeConfig, Grid
from . import classification as _cls
from . import detection as _det
from . import io as _io
from . import licensing as _lic
from . import polygons as _poly
from . import qgis as _qgis
from . import segmentation as _seg
from . import statistics as _stats
from . import timeseries as _ts


def _build_common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--method", default="difference",
                    choices=["difference", "normalized_difference",
                             "relative_change", "ratio", "cva"],
                    help="change operator")
    sp.add_argument("--threshold-method", default="otsu",
                    choices=["otsu", "manual", "percentile", "none"],
                    help="how to threshold the change magnitude")
    sp.add_argument("--threshold", type=float, default=None,
                    help="manual threshold value")
    sp.add_argument("--percentile", type=float, default=95.0)
    sp.add_argument("--min-region-pixels", type=int, default=1)
    sp.add_argument("--connectivity", type=int, default=8, choices=[4, 8])


def _config_from_args(a: argparse.Namespace) -> ChangeConfig:
    return ChangeConfig(
        method=a.method,
        threshold_method=a.threshold_method,
        threshold=a.threshold,
        percentile=a.percentile,
        min_region_pixels=a.min_region_pixels,
        connectivity=a.connectivity,
    )


def cmd_detect(a: argparse.Namespace) -> int:
    cfg = _config_from_args(a)
    t1 = _io.read_grid(a.t1, name="t1")
    t2 = _io.read_grid(a.t2, name="t2")
    if cfg.method == "cva":
        if not a.bands_t1 or not a.bands_t2:
            print("cva needs --bands-t1 and --bands-t2 (comma-separated rasters)",
                  file=sys.stderr)
            return 2
        b1 = [_io.read_grid(p.strip()) for p in a.bands_t1.split(",")]
        b2 = [_io.read_grid(p.strip()) for p in a.bands_t2.split(",")]
        res = _det.detect(t1, t2, cfg, bands_t1=b1, bands_t2=b2)
    else:
        res = _det.detect(t1, t2, cfg)
    os.makedirs(a.out_dir, exist_ok=True)
    diff_path = os.path.join(a.out_dir, "change_difference.tif")
    _io.write_cog(res.change_grid, diff_path)
    mask_path = os.path.join(a.out_dir, "change_mask.tif")
    _io.write_mask_geotiff(res.change_mask, res.change_grid, mask_path)
    vmin = float(res.change_grid.data[res.change_mask].min()) if res.change_mask.any() else -1.0
    vmax = float(res.change_grid.data[res.change_mask].max()) if res.change_mask.any() else 1.0
    _qgis.write_qml(_qgis.difference_raster_qml(vmin, vmax), diff_path + ".qml")
    _qgis.write_qml(_qgis.mask_qml(), mask_path + ".qml")
    stats = _stats.summarize_change(res.change_grid, res.change_mask,
                                    res.gain_mask, res.loss_mask)
    stats["threshold_used"] = res.threshold_used
    stats["method"] = cfg.method
    _io.write_report(_io.change_report(stats), os.path.join(a.out_dir, "change_report.json"))
    print(f"changed pixels: {stats['pixels']} "
          f"({stats['area']:.1f} sq units), "
          f"gain: {stats['gain']['pixels']}, loss: {stats['loss']['pixels']}")
    print(f"wrote {a.out_dir}")
    return 0


def cmd_polygons(a: argparse.Namespace) -> int:
    mask_grid = _io.read_grid(a.mask)
    mask = mask_grid.data > 0
    gain = loss = None
    if a.gain_mask:
        gain = _io.read_grid(a.gain_mask).data > 0
    if a.loss_mask:
        loss = _io.read_grid(a.loss_mask).data > 0
    change_grid = _io.read_grid(a.change) if a.change else None
    labels, regions = _seg.label_regions(mask, connectivity=a.connectivity)
    regions = [r for r in regions if r["pixels"] >= a.min_region_pixels]
    feats: list = []
    if gain is not None or loss is not None:
        for kind, m in (("gain", gain), ("loss", loss)):
            if m is None:
                continue
            lab, regs = _seg.label_regions(m & mask, connectivity=a.connectivity)
            regs = [r for r in regs if r["pixels"] >= a.min_region_pixels]
            feats += _poly.regions_to_features(lab, regs, mask_grid, change_grid,
                                               a.simplify, kind=kind)
    else:
        feats = _poly.regions_to_features(labels, regions, mask_grid, change_grid,
                                          a.simplify, kind="change")
    fc = _poly.features_to_geojson(feats, crs=mask_grid.crs)
    _poly.write_geojson(fc, a.out)
    _qgis.write_qml(_qgis.change_polygon_qml(), a.out + ".qml")
    table = _stats.region_table(regions, mask_grid.pixel_area, change_grid, labels)
    if table:
        _io.write_csv(table, os.path.splitext(a.out)[0] + "_regions.csv")
    print(f"wrote {len(feats)} polygons to {a.out}")
    return 0


def cmd_classify(a: argparse.Namespace) -> int:
    t1 = _io.read_grid(a.t1)
    t2 = _io.read_grid(a.t2)
    names = {}
    if a.class_names:
        with open(a.class_names, encoding="utf-8") as fh:
            names = {int(k): v for k, v in json.load(fh).items()}
    comp = _cls.post_classification_compare(t1, t2, names)
    os.makedirs(a.out_dir, exist_ok=True)
    trans = _cls.transition_grid(t1, t2)
    trans_path = os.path.join(a.out_dir, "transition.tif")
    _io.write_cog(trans, trans_path)
    _qgis.write_qml(_qgis.transition_qml(names or {int(k): str(k) for k in comp["labels"]}),
                     trans_path + ".qml")
    records = _cls.summarize_transitions(comp, t1.pixel_area, names)
    if records:
        _io.write_csv(records, os.path.join(a.out_dir, "transitions.csv"))
    report = {
        "changed_pixels": comp["changed_pixels"],
        "total_valid_pixels": comp["total_valid_pixels"],
        "changed_fraction": comp["changed_fraction"],
        "transitions": [
            {"from": la, "to": lb, "pixels": c,
             "from_name": names.get(la, str(la)), "to_name": names.get(lb, str(lb))}
            for la, lb, c in comp["transitions"]
        ],
    }
    _io.write_report(report, os.path.join(a.out_dir, "classification_report.json"))
    print(f"changed: {comp['changed_pixels']}/{comp['total_valid_pixels']} pixels, "
          f"{len(comp['transitions'])} transitions")
    return 0


def cmd_timeseries(a: argparse.Namespace) -> int:
    rows = _ts.read_imagery_timeseries(a.timeseries_csv)
    breaks = _ts.index_breaks(rows, a.index, a.threshold)
    out = [{"index": b["index"], "from_date": b["from_date"], "to_date": b["to_date"],
            "from_mean": b["from_mean"], "to_mean": b["to_mean"], "delta": b["delta"]}
           for b in breaks]
    if a.out:
        _io.write_csv(out, a.out)
        print(f"wrote {len(out)} breaks to {a.out}")
    else:
        for b in out:
            print(f"{b['from_date']} -> {b['to_date']}: {b['from_mean']:.4f} -> "
                  f"{b['to_mean']:.4f} (delta {b['delta']:+.4f})")
    if not out:
        print("no breaks found")
    return 0


def cmd_license(a: argparse.Namespace) -> int:
    res = _lic.check_license(a.key)
    print(f"valid={res['valid']} tier={res['tier']}: {res['message']}")
    return 0


def cmd_update_check(_a: argparse.Namespace) -> int:
    res = _lic.check_for_updates()
    if res.get("error"):
        print(f"update check failed: {res['error']}")
    elif res["available"]:
        print(f"update available: {res['current']} -> {res['latest']} ({res['url']})")
    else:
        print(f"up to date (v{res['current']})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="survey-change",
                                description="Change-detection engine for surveying "
                                            "and remote sensing")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="detect change between two rasters")
    d.add_argument("t1", help="earlier raster")
    d.add_argument("t2", help="later raster")
    d.add_argument("--out-dir", default="change_out")
    d.add_argument("--bands-t1", default=None)
    d.add_argument("--bands-t2", default=None)
    _build_common(d)
    d.set_defaults(func=cmd_detect)

    pg = sub.add_parser("polygons", help="vectorize a change mask to GeoJSON polygons")
    pg.add_argument("mask", help="change mask raster (nonzero = change)")
    pg.add_argument("--out", default="change_polygons.geojson")
    pg.add_argument("--change", default=None, help="signed change raster for stats")
    pg.add_argument("--gain-mask", default=None)
    pg.add_argument("--loss-mask", default=None)
    pg.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    pg.add_argument("--min-region-pixels", type=int, default=1)
    pg.add_argument("--simplify", type=float, default=0.0,
                    help="Douglas-Peucker tolerance in map units")
    pg.set_defaults(func=cmd_polygons)

    cl = sub.add_parser("classify", help="post-classification comparison")
    cl.add_argument("t1", help="earlier label raster")
    cl.add_argument("t2", help="later label raster")
    cl.add_argument("--out-dir", default="classify_out")
    cl.add_argument("--class-names", default=None, help="JSON {id: name}")
    cl.set_defaults(func=cmd_classify)

    ts = sub.add_parser("timeseries",
                        help="find breaks in a survey-imagery monitor timeseries.csv")
    ts.add_argument("timeseries_csv")
    ts.add_argument("--index", default="NDVI")
    ts.add_argument("--threshold", type=float, default=0.15)
    ts.add_argument("--out", default=None)
    ts.set_defaults(func=cmd_timeseries)

    li = sub.add_parser("license", help="validate a license key")
    li.add_argument("--key", default=None)
    li.set_defaults(func=cmd_license)

    up = sub.add_parser("update-check", help="check for a newer release")
    up.set_defaults(func=cmd_update_check)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
