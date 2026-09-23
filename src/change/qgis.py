"""QGIS-ready styling and layer metadata for change outputs.

Generates QML style files so change rasters and change polygons open in
QGIS already symbolized — no manual styling per pass. Also writes small
sidecar JSON files describing each output layer (the same pattern
survey-imagery uses for its COG+QML pairs).
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple


def _qml_header() -> str:
    return (
        '<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd" "SYSTEM">\n'
        '<qgis version="3.28" styleCategories="Symbology">\n'
    )


def _qml_footer() -> str:
    return "</qgis>\n"


def difference_raster_qml(
    vmin: float,
    vmax: float,
    no_change: float = 0.0,
) -> str:
    """QML for a signed change raster: blue (loss) -> white -> red (gain).

    ``vmin``/``vmax`` set the pseudocolor range; the midpoint is pinned to
    ``no_change`` so zero change always renders neutral.
    """
    lo = min(vmin, no_change)
    hi = max(vmax, no_change)
    span = (hi - lo) or 1.0

    def _t(v: float) -> float:
        return (v - lo) / span

    stops = [
        (lo, "0,92,230,255"),                    # strong loss: blue
        (lo + 0.25 * span, "165,207,255,255"),   # weak loss: light blue
        (no_change, "255,255,255,255"),          # no change: white
        (lo + 0.75 * span, "255,190,170,255"),   # weak gain: light red
        (hi, "220,20,20,255"),                   # strong gain: red
    ]
    items = "\n".join(
        f'      <item value="{v:.6g}" label="{v:.4g}" '
        f'color="{c}" alpha="255"/>' for v, c in stops
    )
    return (
        _qml_header()
        + "  <pipe>\n"
        + '    <rasterrenderer type="singlebandpseudocolor" opacity="1" '
        + 'classificationMin="{:.6g}" classificationMax="{:.6g}" band="1">\n'.format(lo, hi)
        + "      <rastershader>\n"
        + '        <colorrampshader colorRampType="INTERPOLATED" clip="0">\n'
        + items + "\n"
        + "        </colorrampshader>\n"
        + "      </rastershader>\n"
        + "    </rasterrenderer>\n"
        + "  </pipe>\n"
        + _qml_footer()
    )


def change_polygon_qml() -> str:
    """QML for change polygons, categorized on the ``kind`` attribute."""
    cats = [
        ("gain", "Gain", "220,20,20,255"),
        ("loss", "Loss", "0,92,230,255"),
        ("change", "Change", "255,170,0,255"),
    ]
    symbols = []
    categories = []
    for i, (value, label, color) in enumerate(cats):
        symbols.append(
            f'    <symbol name="{i}" type="fill" alpha="0.55">\n'
            f'      <layer class="SimpleFill">\n'
            f'        <prop k="color" v="{color}"/>\n'
            '        <prop k="outline_color" v="35,35,35,255"/>\n'
            '        <prop k="outline_width" v="0.4"/>\n'
            "      </layer>\n"
            "    </symbol>"
        )
        categories.append(
            f'    <category value="{value}" label="{label}" symbol="{i}" render="true"/>'
        )
    return (
        _qml_header()
        + '  <renderer-v2 type="categorizedSymbol" attr="kind">\n'
        + "    <categories>\n" + "\n".join(categories) + "\n    </categories>\n"
        + "    <symbols>\n" + "\n".join(symbols) + "\n    </symbols>\n"
        + "  </renderer-v2>\n"
        + _qml_footer()
    )


def mask_qml() -> str:
    """QML for a binary change mask raster: transparent background, orange change."""
    return (
        _qml_header()
        + "  <pipe>\n"
        + '    <rasterrenderer type="paletted" opacity="1" band="1">\n'
        + '      <colorPalette>\n'
        + '        <paletteEntry value="0" label="No change" color="#ffffff" alpha="0"/>\n'
        + '        <paletteEntry value="1" label="Change" color="#ff8c00" alpha="255"/>\n'
        + "      </colorPalette>\n"
        + "    </rasterrenderer>\n"
        + "  </pipe>\n"
        + _qml_footer()
    )


def transition_qml(class_names: Dict[int, str], palette: Optional[Sequence[str]] = None) -> str:
    """QML for a post-classification transition raster, paletted by code.

    ``class_names`` maps class id -> label; transition codes are
    ``from * 1000 + to`` (see :mod:`change.classification`).
    """
    default_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    pal = list(palette or default_palette)
    entries = []
    codes = sorted({f * 1000 + t for f in class_names for t in class_names if f != t})
    for i, code in enumerate(codes):
        f, t = divmod(code, 1000)
        label = f"{class_names.get(f, f)} -> {class_names.get(t, t)}"
        entries.append(
            f'        <paletteEntry value="{code}" label="{label}" '
            f'color="{pal[i % len(pal)]}" alpha="255"/>'
        )
    return (
        _qml_header()
        + "  <pipe>\n"
        + '    <rasterrenderer type="paletted" opacity="1" band="1">\n'
        + "      <colorPalette>\n" + "\n".join(entries) + "\n"
        + "      </colorPalette>\n"
        + "    </rasterrenderer>\n"
        + "  </pipe>\n"
        + _qml_footer()
    )


def write_qml(qml: str, path: str) -> str:
    """Write a QML string to ``path``; returns the path."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(qml)
    return path


def layer_metadata(
    name: str,
    kind: str,
    crs: Optional[str] = None,
    extra: Optional[Dict] = None,
) -> Dict:
    """Sidecar metadata dict describing one QGIS-ready output layer."""
    meta: Dict = {
        "name": name,
        "kind": kind,  # difference_raster | change_mask | change_polygons | transition
        "crs": crs,
        "produced_by": "survey-change",
    }
    if extra:
        meta.update(extra)
    return meta


def write_layer_metadata(meta: Dict, path: str) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return path
