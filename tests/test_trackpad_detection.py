import unittest

import cv2
import numpy as np

from deskpad_vision.trackpad_detection import detect_trackpad


class TrackpadDetectionTests(unittest.TestCase):
    def test_detects_centered_perspective_trackpad(self) -> None:
        image = np.full((600, 1000, 3), 35, dtype=np.uint8)
        trackpad = np.array([[300, 150], [700, 150], [640, 430], [360, 430]], dtype=np.int32)
        cv2.polylines(image, [trackpad], True, (78, 78, 78), 4, cv2.LINE_AA)
        cv2.line(image, (80, 105), (920, 105), (100, 100, 100), 3)

        detection = detect_trackpad(image, crop_y=540)

        self.assertIsNotNone(detection)
        assert detection is not None
        np.testing.assert_allclose(detection.quad, trackpad, atol=15)

    def test_does_not_invent_trackpad_on_flat_image(self) -> None:
        image = np.full((600, 1000, 3), 35, dtype=np.uint8)

        self.assertIsNone(detect_trackpad(image, crop_y=540))


if __name__ == "__main__":
    unittest.main()
