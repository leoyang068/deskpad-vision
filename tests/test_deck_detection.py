import unittest

import cv2
import numpy as np

from deskpad_vision.deck_detection import (
    detect_deck_side_edges,
    surface_quad_from_edges,
)


class DeckDetectionTests(unittest.TestCase):
    def test_detects_converging_laptop_edges(self) -> None:
        image = np.full((600, 1000, 3), 210, dtype=np.uint8)
        deck = np.array([[20, 20], [980, 20], [690, 550], [310, 550]], dtype=np.int32)
        cv2.fillConvexPoly(image, deck, (35, 35, 35))
        cv2.line(image, (20, 20), (310, 550), (245, 245, 245), 5)
        cv2.line(image, (980, 20), (690, 550), (245, 245, 245), 5)

        edges = detect_deck_side_edges(image, crop_y=560)

        self.assertIsNotNone(edges)
        assert edges is not None
        self.assertLess(abs(edges.left[1][0] - 315), 35)
        self.assertLess(abs(edges.right[1][0] - 685), 35)

    def test_intersects_trackpad_depth_band_with_laptop_edges(self) -> None:
        image = np.full((600, 1000, 3), 210, dtype=np.uint8)
        cv2.line(image, (20, 20), (310, 550), (10, 10, 10), 5)
        cv2.line(image, (980, 20), (690, 550), (10, 10, 10), 5)
        edges = detect_deck_side_edges(image, crop_y=560)
        self.assertIsNotNone(edges)
        assert edges is not None

        trackpad = ((390.0, 260.0), (610.0, 260.0), (650.0, 500.0), (350.0, 500.0))
        surface = surface_quad_from_edges(trackpad, edges)

        self.assertIsNotNone(surface)
        assert surface is not None
        self.assertLess(surface[0][0], trackpad[0][0])
        self.assertGreater(surface[1][0], trackpad[1][0])
        self.assertLess(surface[3][0], trackpad[3][0])
        self.assertGreater(surface[2][0], trackpad[2][0])


if __name__ == "__main__":
    unittest.main()
