from __future__ import annotations

import unittest

import cv2
import numpy as np

from deskpad_vision.calibration_v2.geometry import (
    GeometryError,
    add_constrained_gradient_sides,
    solve_geometry,
)


class CalibrationV2GeometryTests(unittest.TestCase):
    def test_recovers_chassis_and_trackpad_from_learned_line_evidence(self) -> None:
        lines = np.asarray(
            [
                # Physical chassis U: left side, right side and front.
                ((60, 0), (220, 186)),
                ((1158, 0), (1024, 172)),
                ((220, 186), (1024, 172)),
                # Trackpad depth sides share the chassis depth vanishing point.
                ((418, 71), (455, 174)),
                ((829, 41), (789, 168)),
                # The learned detector may return interrupted or doubled edges.
                ((421, 48), (689, 42)),
                ((455, 44), (674, 40)),
                ((461, 176), (556, 174)),
                ((560, 174), (650, 172)),
                ((653, 172), (784, 170)),
                # Keyboard and unrelated scene lines are distractors.
                ((162, 43), (1077, 29)),
                ((286, 35), (803, 31)),
                ((410, 355), (630, 245)),
                ((1104, 431), (1159, 369)),
            ],
            dtype=np.float32,
        )

        result = solve_geometry(lines, (448, 1280, 3))

        expected_trackpad = np.asarray(
            ((409, 46), (829, 38), (789, 169), (455, 176)), dtype=np.float32
        )
        np.testing.assert_allclose(result.trackpad_quad, expected_trackpad, atol=4.0)
        self.assertGreater(result.confidence, 0.80)
        self.assertLess(result.metrics["depth_vanishing_residual_px"], 8.0)

    def test_fails_closed_without_physical_side_evidence(self) -> None:
        horizontal_lines = np.asarray(
            [
                ((100, 40), (1180, 35)),
                ((200, 100), (1080, 95)),
                ((220, 180), (1020, 175)),
                ((400, 60), (850, 55)),
                ((450, 165), (800, 160)),
                ((100, 250), (300, 248)),
                ((800, 300), (1100, 297)),
            ],
            dtype=np.float32,
        )

        with self.assertRaises(GeometryError):
            solve_geometry(horizontal_lines, (448, 1280, 3))

    def test_recovers_faint_sides_along_chassis_vanishing_rays(self) -> None:
        frame = np.full((448, 1280, 3), 45, dtype=np.uint8)
        cv2.line(frame, (406, 46), (454, 176), (190, 190, 190), 3, cv2.LINE_AA)
        cv2.line(frame, (826, 38), (786, 169), (190, 190, 190), 3, cv2.LINE_AA)
        cv2.line(frame, (406, 46), (826, 38), (190, 190, 190), 3, cv2.LINE_AA)
        cv2.line(frame, (454, 176), (786, 169), (190, 190, 190), 3, cv2.LINE_AA)
        lines_without_trackpad_sides = np.asarray(
            [
                ((60, 0), (220, 186)),
                ((1158, 0), (1024, 172)),
                ((220, 186), (1024, 172)),
                ((421, 48), (689, 42)),
                ((455, 44), (674, 40)),
                ((461, 176), (556, 174)),
                ((560, 174), (650, 172)),
                ((653, 172), (784, 170)),
                ((162, 43), (1077, 29)),
            ],
            dtype=np.float32,
        )

        augmented = add_constrained_gradient_sides(frame, lines_without_trackpad_sides)
        result = solve_geometry(augmented, frame.shape)

        self.assertEqual(len(augmented), len(lines_without_trackpad_sides) + 4)
        self.assertGreater(result.confidence, 0.80)
        np.testing.assert_allclose(result.trackpad_quad[0], (406, 46), atol=6.0)
        np.testing.assert_allclose(result.trackpad_quad[2], (786, 169), atol=6.0)


if __name__ == "__main__":
    unittest.main()
