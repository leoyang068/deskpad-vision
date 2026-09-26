import unittest

import cv2
import numpy as np

from deskpad_vision.alignment import KeyboardAligner
from deskpad_vision.geometry import apply_homography


class AlignmentTests(unittest.TestCase):
    def test_recovers_reference_mapping(self) -> None:
        height, width = 360, 640
        reference = np.full((height, width, 3), 30, dtype=np.uint8)
        rng = np.random.default_rng(7)
        for index in range(90):
            x = int(rng.integers(15, width - 15))
            y = int(rng.integers(15, 185))
            color = int(rng.integers(90, 255))
            cv2.circle(reference, (x, y), int(rng.integers(2, 6)), (color,) * 3, -1)
            if index % 8 == 0:
                cv2.putText(
                    reference,
                    str(index),
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                )

        reference_to_current = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]),
            np.float32([[24, 16], [width - 35, 7], [width - 12, height - 8], [10, height - 18]]),
        )
        current = cv2.warpPerspective(reference, reference_to_current, (width, height))
        aligner = KeyboardAligner(reference, (0, 0, width, 205), min_matches=12)
        result = aligner.align(current, crop_y=330)

        self.assertIsNotNone(result)
        assert result is not None
        source_point = apply_homography((320.0, 100.0), reference_to_current)
        assert source_point is not None
        recovered = apply_homography(source_point, result.current_to_reference)
        assert recovered is not None
        self.assertLess(np.linalg.norm(np.array(recovered) - np.array((320.0, 100.0))), 3.0)


if __name__ == "__main__":
    unittest.main()
