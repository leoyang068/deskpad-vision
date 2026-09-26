from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from .config import Rect


@dataclass
class AlignmentResult:
    current_to_reference: np.ndarray
    match_count: int
    inlier_count: int
    median_error: float
    confidence: float
    current_inliers: np.ndarray


class KeyboardAligner:
    """Align the live image to a saved view using keyboard texture."""

    def __init__(
        self,
        reference_bgr: np.ndarray,
        keyboard_rect: Rect,
        min_matches: int = 18,
        max_median_error: float = 4.5,
    ) -> None:
        self.reference_bgr = reference_bgr
        self.keyboard_rect = keyboard_rect
        self.min_matches = min_matches
        self.max_median_error = max_median_error
        self.orb = cv2.ORB_create(
            nfeatures=2200,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            patchSize=31,
            fastThreshold=10,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.reference_gray = self._prepare_gray(reference_bgr)
        self.reference_mask = self._rect_mask(self.reference_gray.shape, keyboard_rect)
        self.reference_keypoints, self.reference_descriptors = self.orb.detectAndCompute(
            self.reference_gray, self.reference_mask
        )
        if self.reference_descriptors is None or len(self.reference_keypoints) < self.min_matches:
            raise ValueError(
                "The reference keyboard does not contain enough stable features. "
                "Improve lighting and recalibrate."
            )

    def _prepare_gray(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(gray)

    @staticmethod
    def _rect_mask(shape: Sequence[int], rect: Rect) -> np.ndarray:
        height, width = int(shape[0]), int(shape[1])
        x, y, rect_width, rect_height = rect
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + rect_width), min(height, y + rect_height)
        mask = np.zeros((height, width), dtype=np.uint8)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
        return mask

    def align(self, current_bgr: np.ndarray, crop_y: int) -> Optional[AlignmentResult]:
        if current_bgr.shape[:2] != self.reference_bgr.shape[:2]:
            raise ValueError("Current frame and reference frame must have the same dimensions")

        gray = self._prepare_gray(current_bgr)
        mask = np.zeros_like(gray)
        mask[: max(1, min(gray.shape[0], crop_y)), :] = 255
        current_keypoints, current_descriptors = self.orb.detectAndCompute(gray, mask)
        if current_descriptors is None or len(current_keypoints) < self.min_matches:
            return None

        pairs = self.matcher.knnMatch(current_descriptors, self.reference_descriptors, k=2)
        good = []
        for pair in pairs:
            if len(pair) != 2:
                continue
            best, second = pair
            if best.distance < 0.76 * second.distance:
                good.append(best)
        if len(good) < self.min_matches:
            return None

        current_points = np.float32(
            [current_keypoints[match.queryIdx].pt for match in good]
        ).reshape(-1, 1, 2)
        reference_points = np.float32(
            [self.reference_keypoints[match.trainIdx].pt for match in good]
        ).reshape(-1, 1, 2)
        matrix, inlier_mask = cv2.findHomography(
            current_points,
            reference_points,
            cv2.RANSAC,
            ransacReprojThreshold=4.0,
            maxIters=2500,
            confidence=0.995,
        )
        if matrix is None or inlier_mask is None:
            return None

        inliers = inlier_mask.reshape(-1).astype(bool)
        inlier_count = int(inliers.sum())
        if inlier_count < self.min_matches:
            return None

        projected = cv2.perspectiveTransform(current_points[inliers], matrix)
        errors = np.linalg.norm(
            projected.reshape(-1, 2) - reference_points[inliers].reshape(-1, 2), axis=1
        )
        median_error = float(np.median(errors))
        if not np.isfinite(matrix).all() or median_error > self.max_median_error:
            return None

        matrix = matrix / matrix[2, 2]
        if not self._is_sane(matrix, current_bgr.shape[1], current_bgr.shape[0]):
            return None

        inlier_ratio = inlier_count / max(1, len(good))
        count_score = min(1.0, inlier_count / 45.0)
        error_score = max(0.0, 1.0 - median_error / self.max_median_error)
        confidence = float(inlier_ratio * count_score * (0.35 + 0.65 * error_score))
        return AlignmentResult(
            current_to_reference=matrix,
            match_count=len(good),
            inlier_count=inlier_count,
            median_error=median_error,
            confidence=confidence,
            current_inliers=current_points[inliers].reshape(-1, 2),
        )

    @staticmethod
    def _is_sane(matrix: np.ndarray, width: int, height: int) -> bool:
        sample = np.float32(
            [
                [[width * 0.25, height * 0.25]],
                [[width * 0.75, height * 0.25]],
                [[width * 0.5, height * 0.6]],
            ]
        )
        mapped = cv2.perspectiveTransform(sample, matrix).reshape(-1, 2)
        if not np.isfinite(mapped).all():
            return False
        horizontal_scale = np.linalg.norm(mapped[1] - mapped[0]) / max(1.0, width * 0.5)
        if not 0.30 <= horizontal_scale <= 3.2:
            return False
        triangle_area = abs(float(np.cross(mapped[1] - mapped[0], mapped[2] - mapped[0]))) / 2.0
        return triangle_area > width * height * 0.002


def transform_polygon(polygon: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, matrix).reshape(-1, 2)
