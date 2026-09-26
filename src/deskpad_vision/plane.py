from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .config import Point, Profile, Quad
from .geometry import apply_homography

SURFACE_PLANE_WIDTH = 1800.0
SURFACE_PLANE_HEIGHT = 625.0
TRACKPAD_GAP = 24.0


def quad_array(quad: Quad) -> np.ndarray:
    return np.asarray(quad, dtype=np.float32)


def bounding_rect_for_quad(quad: Quad) -> Tuple[int, int, int, int]:
    points = quad_array(quad)
    left = int(np.floor(np.min(points[:, 0])))
    top = int(np.floor(np.min(points[:, 1])))
    right = int(np.ceil(np.max(points[:, 0])))
    bottom = int(np.ceil(np.max(points[:, 1])))
    return left, top, right - left, bottom - top


def _point_in_polygon(point: Point, polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon.astype(np.float32), point, False) >= 0


@dataclass(frozen=True)
class PlanarSurface:
    """A perspective-correct coordinate system anchored to the physical trackpad."""

    reference_to_plane: np.ndarray
    plane_to_reference: np.ndarray
    plane_zones: Dict[str, np.ndarray]
    reference_zones: Dict[str, np.ndarray]
    trackpad_reference: np.ndarray
    surface_reference: np.ndarray
    trackpad_plane: np.ndarray

    @classmethod
    def from_profile(cls, profile: Profile) -> "PlanarSurface":
        surface_reference = quad_array(profile.effective_surface_quad)
        surface_plane = np.array(
            [
                [0.0, 0.0],
                [SURFACE_PLANE_WIDTH, 0.0],
                [SURFACE_PLANE_WIDTH, SURFACE_PLANE_HEIGHT],
                [0.0, SURFACE_PLANE_HEIGHT],
            ],
            dtype=np.float32,
        )
        reference_to_plane = cv2.getPerspectiveTransform(surface_reference, surface_plane)
        plane_to_reference = cv2.getPerspectiveTransform(surface_plane, surface_reference)

        trackpad_reference = quad_array(profile.effective_trackpad_quad)
        trackpad_plane = cv2.perspectiveTransform(
            trackpad_reference.reshape(1, -1, 2), reference_to_plane
        ).reshape(-1, 2)

        left_zone = np.array(
            [
                [0.0, 0.0],
                [max(0.0, trackpad_plane[0, 0] - TRACKPAD_GAP), trackpad_plane[0, 1]],
                [max(0.0, trackpad_plane[3, 0] - TRACKPAD_GAP), trackpad_plane[3, 1]],
                [0.0, SURFACE_PLANE_HEIGHT],
            ],
            dtype=np.float32,
        )
        right_zone = np.array(
            [
                [
                    min(SURFACE_PLANE_WIDTH, trackpad_plane[1, 0] + TRACKPAD_GAP),
                    trackpad_plane[1, 1],
                ],
                [SURFACE_PLANE_WIDTH, 0.0],
                [SURFACE_PLANE_WIDTH, SURFACE_PLANE_HEIGHT],
                [
                    min(SURFACE_PLANE_WIDTH, trackpad_plane[2, 0] + TRACKPAD_GAP),
                    trackpad_plane[2, 1],
                ],
            ],
            dtype=np.float32,
        )
        candidates = {"left": left_zone, "right": right_zone}
        plane_zones: Dict[str, np.ndarray] = {}
        reference_zones: Dict[str, np.ndarray] = {}
        for name, polygon in candidates.items():
            if abs(float(cv2.contourArea(polygon))) < 100.0:
                continue
            plane_zones[name] = polygon
            reference_zones[name] = cv2.perspectiveTransform(
                polygon.reshape(1, -1, 2), plane_to_reference
            ).reshape(-1, 2)

        return cls(
            reference_to_plane=reference_to_plane,
            plane_to_reference=plane_to_reference,
            plane_zones=plane_zones,
            reference_zones=reference_zones,
            trackpad_reference=trackpad_reference,
            surface_reference=surface_reference,
            trackpad_plane=trackpad_plane,
        )

    def to_plane(self, reference_point: Point) -> Optional[Point]:
        return apply_homography(reference_point, self.reference_to_plane)

    def to_reference(self, plane_point: Point) -> Optional[Point]:
        return apply_homography(plane_point, self.plane_to_reference)

    def containing_plane(self, plane_point: Point) -> Optional[str]:
        for name, polygon in self.plane_zones.items():
            if _point_in_polygon(plane_point, polygon):
                return name
        return None

    def containing_reference(self, reference_point: Point) -> Optional[str]:
        plane_point = self.to_plane(reference_point)
        return self.containing_plane(plane_point) if plane_point is not None else None
