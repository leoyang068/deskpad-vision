import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from deskpad_vision.config import Profile
from deskpad_vision.plane import (
    SURFACE_PLANE_HEIGHT,
    SURFACE_PLANE_WIDTH,
    PlanarSurface,
    bounding_rect_for_quad,
)


class PlanarSurfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.surface_quad = ((80.0, 180.0), (920.0, 250.0), (970.0, 550.0), (30.0, 470.0))
        canonical_surface = np.array(
            [
                [0.0, 0.0],
                [SURFACE_PLANE_WIDTH, 0.0],
                [SURFACE_PLANE_WIDTH, SURFACE_PLANE_HEIGHT],
                [0.0, SURFACE_PLANE_HEIGHT],
            ],
            dtype=np.float32,
        )
        self.canonical_trackpad = np.array(
            [[520.0, 0.0], [1280.0, 0.0], [1280.0, 625.0], [520.0, 625.0]],
            dtype=np.float32,
        )
        plane_to_reference = cv2.getPerspectiveTransform(
            canonical_surface, np.asarray(self.surface_quad, dtype=np.float32)
        )
        projected_trackpad = cv2.perspectiveTransform(
            self.canonical_trackpad.reshape(1, -1, 2), plane_to_reference
        ).reshape(-1, 2)
        self.trackpad_quad = tuple(
            tuple(float(value) for value in point) for point in projected_trackpad
        )
        self.profile = Profile(
            image_width=1000,
            image_height=600,
            crop_y=560,
            trackpad_rect=bounding_rect_for_quad(self.trackpad_quad),
            keyboard_rect=(0, 0, 1000, 190),
            trackpad_quad=self.trackpad_quad,
            surface_quad=self.surface_quad,
        )

    def test_laptop_surface_defines_canonical_plane(self) -> None:
        surface = PlanarSurface.from_profile(self.profile)
        expected = (
            (0.0, 0.0),
            (SURFACE_PLANE_WIDTH, 0.0),
            (SURFACE_PLANE_WIDTH, SURFACE_PLANE_HEIGHT),
            (0.0, SURFACE_PLANE_HEIGHT),
        )
        for reference_point, expected_point in zip(
            self.profile.effective_surface_quad, expected
        ):
            actual = surface.to_plane(reference_point)
            self.assertIsNotNone(actual)
            np.testing.assert_allclose(actual, expected_point, atol=1e-3)
        np.testing.assert_allclose(surface.trackpad_plane, self.canonical_trackpad, atol=1e-3)

    def test_side_zones_are_perspective_polygons(self) -> None:
        surface = PlanarSurface.from_profile(self.profile)
        left_plane = np.mean(surface.plane_zones["left"], axis=0)
        left_reference = surface.to_reference(tuple(left_plane))
        self.assertIsNotNone(left_reference)
        self.assertEqual(surface.containing_reference(left_reference), "left")

        left_polygon = surface.reference_zones["left"]
        self.assertNotAlmostEqual(float(left_polygon[0, 1]), float(left_polygon[1, 1]))

    def test_profile_round_trip_preserves_quads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.profile.save(path)
            loaded = Profile.load(path)
        self.assertEqual(loaded.trackpad_quad, self.profile.trackpad_quad)
        self.assertEqual(loaded.surface_quad, self.profile.surface_quad)


if __name__ == "__main__":
    unittest.main()
