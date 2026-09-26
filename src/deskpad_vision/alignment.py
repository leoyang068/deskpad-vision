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
    stable_current_to_reference: Optional[np.ndarray] = None


class LockedAlignmentGuard:
    """Acquire one stable V2 relock transform, then use matching as a drift alarm.

    A homography estimated from a narrow keyboard band is poorly constrained when
    extrapolated over the palm rest. V2 instead estimates a restricted similarity
    transform, confirms it across several frames, and freezes that transform for
    the session. Several bad or displaced frames release the pointer lock.
    """

    def __init__(
        self,
        width: int,
        keyboard_rect: Rect,
        max_drift: float = 8.0,
        acquire_frames: int = 3,
        release_frames: int = 2,
    ) -> None:
        x, y, rect_width, rect_height = keyboard_rect
        self.sample_points = np.float32(
            [
                [x + rect_width * 0.18, y + rect_height * 0.30],
                [x + rect_width * 0.50, y + rect_height * 0.30],
                [x + rect_width * 0.82, y + rect_height * 0.30],
                [x + rect_width * 0.28, y + rect_height * 0.72],
                [x + rect_width * 0.72, y + rect_height * 0.72],
            ]
        )
        self.width = width
        self.max_drift = max_drift
        self.acquire_frames = acquire_frames
        self.release_frames = release_frames
        self.good_streak = 0
        self.bad_streak = 0
        self.locked = False
        self.last_drift: Optional[float] = None
        self.baseline_matrix: Optional[np.ndarray] = None
        self._acquisition_matrices: list[np.ndarray] = []

    @staticmethod
    def _candidate_matrix(result: AlignmentResult) -> np.ndarray:
        if result.stable_current_to_reference is not None:
            return result.stable_current_to_reference
        return result.current_to_reference

    def _matrix_drift(self, first: np.ndarray, second: np.ndarray) -> float:
        points = self.sample_points.reshape(-1, 1, 2)
        first_mapped = cv2.perspectiveTransform(points, first).reshape(-1, 2)
        second_mapped = cv2.perspectiveTransform(points, second).reshape(-1, 2)
        return float(np.max(np.linalg.norm(first_mapped - second_mapped, axis=1)))

    def update(self, result: Optional[AlignmentResult]) -> bool:
        good = False
        self.last_drift = None
        if result is not None:
            candidate = self._candidate_matrix(result)
            if np.all(np.isfinite(candidate)):
                if self.baseline_matrix is None:
                    if self._acquisition_matrices:
                        reference = self._acquisition_matrices[0]
                        self.last_drift = self._matrix_drift(candidate, reference)
                        good = self.last_drift <= self.max_drift
                    else:
                        self.last_drift = 0.0
                        good = True
                    if good:
                        self._acquisition_matrices.append(candidate.copy())
                else:
                    self.last_drift = self._matrix_drift(candidate, self.baseline_matrix)
                    good = self.last_drift <= self.max_drift

        if good:
            self.good_streak += 1
            self.bad_streak = 0
            if self.baseline_matrix is None and self.good_streak >= self.acquire_frames:
                stacked = np.stack(self._acquisition_matrices[-self.acquire_frames :], axis=0)
                self.baseline_matrix = np.median(stacked, axis=0)
                self.baseline_matrix[2] = (0.0, 0.0, 1.0)
                self.locked = True
            elif self.baseline_matrix is not None:
                self.locked = True
        else:
            self.bad_streak += 1
            self.good_streak = 0
            if self.baseline_matrix is None:
                self._acquisition_matrices.clear()
            if self.bad_streak >= self.release_frames:
                self.locked = False
        return self.locked

    def preview_result(self, result: AlignmentResult) -> AlignmentResult:
        matrix = self.baseline_matrix
        if matrix is None:
            matrix = self._candidate_matrix(result)
        return AlignmentResult(
            current_to_reference=matrix.copy(),
            match_count=result.match_count,
            inlier_count=result.inlier_count,
            median_error=result.median_error,
            confidence=result.confidence,
            current_inliers=result.current_inliers,
            stable_current_to_reference=matrix.copy(),
        )


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

        stable_matrix = None
        affine, _affine_inliers = cv2.estimateAffinePartial2D(
            current_points[inliers].reshape(-1, 2),
            reference_points[inliers].reshape(-1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.995,
            refineIters=10,
        )
        if affine is not None and np.all(np.isfinite(affine)):
            candidate = np.vstack([affine, np.array([0.0, 0.0, 1.0])])
            if self._is_sane(candidate, current_bgr.shape[1], current_bgr.shape[0]):
                stable_matrix = candidate

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
            stable_current_to_reference=stable_matrix,
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
