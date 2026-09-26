import tempfile
import unittest
from pathlib import Path

import numpy as np

from deskpad_vision.app import _clamp_roi, _default_trackpad_roi, _rect_from_points
from deskpad_vision.config import Profile
from deskpad_vision.geometry import ActiveZones, apply_homography


class GeometryTests(unittest.TestCase):
    def test_trackpad_roi_helpers(self) -> None:
        self.assertEqual(_rect_from_points((80, 90), (20, 30)), (20, 30, 60, 60))
        self.assertEqual(_clamp_roi((-5, 90, 60, 40), 100, 100), (0, 60, 60, 40))
        self.assertEqual(_default_trackpad_roi(1000, 600), (290, 357, 420, 228))

    def test_homography_maps_point(self) -> None:
        matrix = np.array([[2.0, 0.0, 10.0], [0.0, 3.0, -5.0], [0.0, 0.0, 1.0]])
        self.assertEqual(apply_homography((4.0, 5.0), matrix), (18.0, 10.0))

    def test_zones_are_derived_around_trackpad(self) -> None:
        profile = Profile(
            image_width=1000,
            image_height=600,
            crop_y=560,
            trackpad_rect=(350, 300, 300, 220),
            keyboard_rect=(0, 0, 1000, 280),
            zone_margin=20,
            trackpad_gap=10,
        )
        zones = ActiveZones.from_profile(profile)
        self.assertEqual(zones.containing((100, 400)), "left")
        self.assertEqual(zones.containing((900, 400)), "right")
        self.assertIsNone(zones.containing((500, 400)))

    def test_profile_round_trip(self) -> None:
        profile = Profile(800, 600, 520, (250, 300, 300, 200), (0, 0, 800, 270))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            profile.save(path)
            self.assertEqual(Profile.load(path), profile)


if __name__ == "__main__":
    unittest.main()
