"""survey-change: change-detection engine for surveying and remote sensing.

Pure-logic engine (no UI imports): multi-method raster differencing,
change vector analysis, post-classification comparison, change polygons,
multi-epoch tracking, and QGIS-ready outputs. Thin CLI in
:mod:`change.cli`; run ``python -m change --help``.
"""

from .core import (
    AlignmentError,
    ChangeConfig,
    ChangeError,
    ChangeResult,
    Grid,
    GridError,
)
from . import classification
from . import detection
from . import interop
from . import io
from . import licensing
from . import polygons
from . import qgis
from . import segmentation
from . import statistics
from . import timeseries

__version__ = "0.1.0"

__all__ = [
    "AlignmentError",
    "ChangeConfig",
    "ChangeError",
    "ChangeResult",
    "Grid",
    "GridError",
    "classification",
    "detection",
    "interop",
    "io",
    "licensing",
    "polygons",
    "qgis",
    "segmentation",
    "statistics",
    "timeseries",
    "__version__",
]
