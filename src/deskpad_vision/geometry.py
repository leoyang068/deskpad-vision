from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .config import Profile, Rect

Point = Tuple[float, float]


def apply_homography(point: Point, matrix: np.ndarray) -> Optional[Point]:
    """Map a point through a 3x3 homography, rejecting invalid projective results."""

    vector = np.array([point[0], point[1], 1.0], dtype=np.float64)
    mapped = matrix @ vector
    if not np.isfinite(mapped).all() or abs(float(mapped[2])) < 1e-8:
        return None
    return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])


def rect_polygon(rect: Rect) -> np.ndarray:
    x, y, width, height = rect
    return np.array(
        [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class ActiveZones:
    zones: Dict[str, Rect]

    @classmethod
    def from_profile(cls, profile: Profile) -> "ActiveZones":
        tx, ty, tw, th = profile.trackpad_rect
        y0 = max(profile.zone_margin, ty)
        y1 = min(profile.crop_y - profile.zone_margin, ty + th)
        usable_height = max(0, y1 - y0)

        left_x0 = profile.zone_margin
        left_x1 = max(left_x0, tx - profile.trackpad_gap)
        right_x0 = min(profile.image_width - profile.zone_margin, tx + tw + profile.trackpad_gap)
        right_x1 = profile.image_width - profile.zone_margin

        zones: Dict[str, Rect] = {}
        if left_x1 - left_x0 >= 30 and usable_height >= 30:
            zones["left"] = (left_x0, y0, left_x1 - left_x0, usable_height)
        if right_x1 - right_x0 >= 30 and usable_height >= 30:
            zones["right"] = (right_x0, y0, right_x1 - right_x0, usable_height)
        return cls(zones)

    def containing(self, point: Point) -> Optional[str]:
        px, py = point
        for name, (x, y, width, height) in self.zones.items():
            if x <= px <= x + width and y <= py <= y + height:
                return name
        return None
