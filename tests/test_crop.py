import unittest

import cv2
import numpy as np

from deskpad_vision.crop import BoundarySampler, detect_horizontal_boundary


class CropDetectionTests(unittest.TestCase):
    def test_detects_long_lower_boundary(self) -> None:
        height, width = 240, 400
        image = np.full((height, width, 3), 95, dtype=np.uint8)
        image[:70, :] = 45
        for x in range(15, width - 15, 35):
            cv2.rectangle(image, (x, 20), (x + 24, 52), (180, 180, 180), 2)
        expected_y = 194
        image[expected_y - 2 : expected_y + 3, :] = 220
        image[expected_y + 3 :, :] = 25

        actual = detect_horizontal_boundary(image)
        self.assertLessEqual(abs(actual - expected_y), 5)

    def test_sampler_uses_temporal_median(self) -> None:
        height, width = 180, 320
        sampler = BoundarySampler(sample_count=5)
        detected = None
        for index in range(5):
            image = np.full((height, width, 3), 80, dtype=np.uint8)
            image[145:149, :] = 230
            image[149:, :] = 15
            # Moving foreground should be removed by the temporal median.
            cv2.rectangle(image, (index * 35, 95), (index * 35 + 45, 140), (240, 240, 240), -1)
            detected = sampler.add(image)
        self.assertIsNotNone(detected)
        self.assertLessEqual(abs(int(detected) - 147), 6)


if __name__ == "__main__":
    unittest.main()
