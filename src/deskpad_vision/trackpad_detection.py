from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import Point, Quad
from .deck_detection import Line, line_intersection


@dataclass(frozen=True)
class TrackpadDetection:
    quad: Quad
    confidence: float


@dataclass(frozen=True)
class _ImageSegment:
    first: Point
    second: Point
    length: float

    @property
    def midpoint(self) -> Point:
        return (
            (self.first[0] + self.second[0]) / 2.0,
            (self.first[1] + self.second[1]) / 2.0,
        )


def _detect_segments(frame: np.ndarray, crop_y: int) -> List[_ImageSegment]:
    lightness = cv2.cvtColor(frame[:crop_y], cv2.COLOR_BGR2LAB)[:, :, 0]
    enhanced = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(lightness)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    raw_lines = detector.detect(enhanced)[0]
    if raw_lines is None:
        return []
    result = []
    for raw in raw_lines:
        x1, y1, x2, y2 = (float(value) for value in raw[0])
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length >= 24:
            result.append(_ImageSegment((x1, y1), (x2, y2), length))
    return result


def _fit_horizontal(
    segments: List[_ImageSegment], minimum_y: float, maximum_y: float
) -> Optional[Tuple[Line, float]]:
    candidates = []
    for segment in segments:
        dx = segment.second[0] - segment.first[0]
        dy = segment.second[1] - segment.first[1]
        if abs(dx) < 1e-6 or abs(dy / dx) > 0.12:
            continue
        middle_y = segment.midpoint[1]
        if minimum_y <= middle_y <= maximum_y:
            candidates.append(segment)
    if not candidates:
        return None

    bins = {}
    for segment in candidates:
        key = int(round(segment.midpoint[1] / 7.0))
        bins.setdefault(key, []).append(segment)
    group = max(bins.values(), key=lambda values: sum(item.length for item in values))
    center_y = float(np.median([segment.midpoint[1] for segment in group]))
    compatible = [
        segment for segment in candidates if abs(segment.midpoint[1] - center_y) <= 8.0
    ]
    points = np.asarray(
        [point for segment in compatible for point in (segment.first, segment.second)],
        dtype=np.float32,
    )
    fitted = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    vx, vy, x0, y0 = (float(value) for value in fitted)
    if abs(vx) < 1e-6:
        return None
    slope = vy / vx
    intercept = y0 - slope * x0
    left_x = float(np.min(points[:, 0]))
    right_x = float(np.max(points[:, 0]))
    support = sum(segment.length for segment in compatible)
    return (
        ((left_x, slope * left_x + intercept), (right_x, slope * right_x + intercept)),
        support,
    )


def _fit_trackpad_side_from_gradient(
    gray: np.ndarray,
    top_y: float,
    bottom_y: float,
    image_width: int,
    side: str,
) -> Optional[Tuple[Line, float]]:
    """Find a faint trackpad side by integrating image gradients along a line.

    A black trackpad often has enough contrast to reveal its long side as a gradient,
    even when that edge is too faint or interrupted for a line-segment detector.
    """

    first_y = max(0, int(round(top_y + 5)))
    last_y = min(gray.shape[0] - 1, int(round(bottom_y - 5)))
    if last_y - first_y < 45:
        return None
    enhanced = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    gradient = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3))
    ys = np.arange(first_y, last_y + 1, dtype=np.int32)

    if side == "left":
        top_range = range(int(image_width * 0.24), int(image_width * 0.43), 4)
        delta_range = range(0, int(image_width * 0.14), 4)
        sign = 1
    else:
        top_range = range(int(image_width * 0.57), int(image_width * 0.76), 4)
        delta_range = range(0, int(image_width * 0.14), 4)
        sign = -1

    def line_score(top_x: int, bottom_x: int) -> float:
        positions = top_x + (bottom_x - top_x) * (ys - first_y) / (
            last_y - first_y
        )
        xs = np.rint(positions).astype(np.int32)
        if np.any(xs < 3) or np.any(xs >= image_width - 3):
            return 0.0
        # The edge can be several pixels wide after reflection and CLAHE. Sampling
        # a narrow band is more stable than assuming the peak is exactly one pixel.
        samples = np.stack([gradient[ys, xs + offset] for offset in range(-2, 3)])
        return float(np.mean(np.max(samples, axis=0)))

    candidates = []
    for top_x in top_range:
        for delta in delta_range:
            bottom_x = top_x + sign * delta
            candidates.append((line_score(top_x, bottom_x), top_x, bottom_x))
    candidates.sort(reverse=True)
    if not candidates:
        return None
    coarse_score, coarse_top, coarse_bottom = candidates[0]

    refined = []
    for top_x in range(coarse_top - 5, coarse_top + 6):
        for bottom_x in range(coarse_bottom - 5, coarse_bottom + 6):
            if side == "left" and bottom_x < top_x:
                continue
            if side == "right" and bottom_x > top_x:
                continue
            refined.append((line_score(top_x, bottom_x), top_x, bottom_x))
    best_score, best_top, best_bottom = max(
        refined or [(coarse_score, coarse_top, coarse_bottom)]
    )
    background_score = float(np.median([item[0] for item in candidates]))
    contrast = best_score / max(1.0, background_score)
    if best_score < 9.0 or contrast < 1.18:
        return None
    confidence = min(1.0, (best_score / 32.0) * min(1.0, contrast / 2.0))
    line = (
        (float(best_top), float(first_y)),
        (float(best_bottom), float(last_y)),
    )
    return line, confidence


def detect_trackpad(frame: np.ndarray, crop_y: int) -> Optional[TrackpadDetection]:
    """Detect a large centered trackpad quadrilateral from faint border segments."""

    image_height, image_width = frame.shape[:2]
    crop_y = max(80, min(int(crop_y), image_height))
    segments = _detect_segments(frame, crop_y)
    if not segments:
        return None

    top_result = _fit_horizontal(segments, crop_y * 0.20, crop_y * 0.48)
    bottom_result = _fit_horizontal(segments, crop_y * 0.58, crop_y * 0.84)
    if top_result is None or bottom_result is None:
        return None
    top_line, top_support = top_result
    bottom_line, bottom_support = bottom_result
    top_y = sum(point[1] for point in top_line) / 2.0
    bottom_y = sum(point[1] for point in bottom_line) / 2.0
    if bottom_y - top_y < crop_y * 0.22:
        return None

    gray = cv2.cvtColor(frame[:crop_y], cv2.COLOR_BGR2GRAY)
    left_result = _fit_trackpad_side_from_gradient(
        gray, top_y, bottom_y, image_width, "left"
    )
    right_result = _fit_trackpad_side_from_gradient(
        gray, top_y, bottom_y, image_width, "right"
    )
    if left_result is None or right_result is None:
        return None
    left_line, left_support = left_result
    right_line, right_support = right_result

    top_left = line_intersection(left_line, top_line)
    top_right = line_intersection(right_line, top_line)
    bottom_right = line_intersection(right_line, bottom_line)
    bottom_left = line_intersection(left_line, bottom_line)
    if None in (top_left, top_right, bottom_right, bottom_left):
        return None
    assert top_left is not None
    assert top_right is not None
    assert bottom_right is not None
    assert bottom_left is not None
    quad = (top_left, top_right, bottom_right, bottom_left)
    points = np.asarray(quad, dtype=np.float32)
    if not cv2.isContourConvex(points):
        return None
    area = abs(float(cv2.contourArea(points)))
    if area < image_width * crop_y * 0.08:
        return None
    widths = (top_right[0] - top_left[0], bottom_right[0] - bottom_left[0])
    if min(widths) < image_width * 0.22 or max(widths) > image_width * 0.70:
        return None
    center_x = float(np.mean(points[:, 0]))
    if not 0.38 * image_width <= center_x <= 0.62 * image_width:
        return None

    side_confidence = min(left_support, right_support)
    horizontal_confidence = min(
        1.0, min(top_support, bottom_support) / max(1.0, image_width * 0.25)
    )
    confidence = 0.55 * side_confidence + 0.45 * horizontal_confidence
    return TrackpadDetection(quad=quad, confidence=confidence)
