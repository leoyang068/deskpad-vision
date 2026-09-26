from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ..config import Point, Quad

Line = Tuple[Point, Point]


@dataclass(frozen=True)
class DetectedLine:
    """One finite line segment emitted by the calibration-only detector."""

    first: Point
    second: Point
    score: float = 1.0

    @property
    def points(self) -> Line:
        return self.first, self.second


@dataclass(frozen=True)
class CalibrationGeometry:
    """Physical plane geometry selected from learned line evidence."""

    chassis_left: Line
    chassis_right: Line
    chassis_front: Line
    trackpad_quad: Quad
    surface_quad: Quad
    confidence: float
    metrics: Dict[str, float] = field(default_factory=dict)
    evidence_lines: Tuple[DetectedLine, ...] = ()

    def to_dict(self) -> Dict[str, object]:
        def line(value: Line) -> List[List[float]]:
            return [[float(x), float(y)] for x, y in value]

        return {
            "chassis": {
                "left": line(self.chassis_left),
                "right": line(self.chassis_right),
                "front": line(self.chassis_front),
            },
            "trackpad_quad": line(self.trackpad_quad),
            "surface_quad": line(self.surface_quad),
            "confidence": float(self.confidence),
            "metrics": {key: float(value) for key, value in self.metrics.items()},
        }
