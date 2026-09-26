import unittest

from deskpad_vision.browser_calibration import _validate_selection


class BrowserCalibrationTests(unittest.TestCase):
    def test_accepts_valid_selection(self) -> None:
        trackpad = [[300, 250], [500, 260], [520, 450], [280, 440]]
        surface = [[50, 220], [750, 280], [790, 490], [10, 430]]
        payload = {"cropY": 520, "trackpadQuad": trackpad, "surfaceQuad": surface}
        self.assertEqual(
            _validate_selection(payload, 800, 600),
            (
                520,
                ((300.0, 250.0), (500.0, 260.0), (520.0, 450.0), (280.0, 440.0)),
                ((50.0, 220.0), (750.0, 280.0), (790.0, 490.0), (10.0, 430.0)),
            ),
        )

    def test_rejects_trackpad_outside_surface(self) -> None:
        payload = {
            "cropY": 520,
            "trackpadQuad": [[300, 250], [500, 260], [520, 450], [280, 440]],
            "surfaceQuad": [[350, 220], [750, 280], [790, 490], [330, 430]],
        }
        with self.assertRaises(ValueError):
            _validate_selection(payload, 800, 600)

    def test_accepts_shared_perspective_boundaries_with_rounding_error(self) -> None:
        payload = {
            "cropY": 442,
            "trackpadQuad": [
                [396.2387, 136.6842],
                [851.4824, 132.7364],
                [809.5461, 338.2241],
                [443.1788, 341.1343],
            ],
            "surfaceQuad": [
                [59.2492, 139.6065],
                [1224.5418, 129.5013],
                [1119.4543, 335.7624],
                [185.9867, 343.1772],
            ],
        }

        crop_y, trackpad, surface = _validate_selection(payload, 1280, 720)

        self.assertEqual(crop_y, 442)
        self.assertEqual(len(trackpad), 4)
        self.assertEqual(len(surface), 4)


if __name__ == "__main__":
    unittest.main()
