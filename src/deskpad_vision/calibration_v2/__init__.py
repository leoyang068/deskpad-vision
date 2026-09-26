"""Model-assisted, fail-closed calibration for the laptop base plane."""

from .geometry import GeometryError, solve_geometry
from .types import CalibrationGeometry, DetectedLine

__all__ = [
    "CalibrationGeometry",
    "DetectedLine",
    "GeometryError",
    "solve_geometry",
]
