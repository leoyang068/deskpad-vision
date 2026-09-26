from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import Point, Quad

Line = Tuple[Point, Point]


@dataclass(frozen=True)
class DeckSideEdges:
    left: Line
    right: Line
    confidence: float


@dataclass(frozen=True)
class _Segment:
    slope: float
    intercept: float
    length: float
    vertical_span: float
    points: Line


def _segments(lines: np.ndarray) -> List[_Segment]:
    result = []
    for raw in lines[:, 0, :]:
        x1, y1, x2, y2 = (float(value) for value in raw)
        if y2 < y1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        vertical_span = y2 - y1
        if vertical_span < 25:
            continue
        slope = (x2 - x1) / vertical_span
        intercept = x1 - slope * y1
        result.append(
            _Segment(
                slope=slope,
                intercept=intercept,
                length=float(np.hypot(x2 - x1, vertical_span)),
                vertical_span=vertical_span,
                points=((x1, y1), (x2, y2)),
            )
        )
    return result


def _fit_side(
    candidates: List[_Segment], image_width: int, crop_y: int, side: str
) -> Optional[Tuple[Line, float]]:
    if not candidates:
        return None
    target_intercept = 0.0 if side == "left" else float(image_width)

    def score(segment: _Segment) -> float:
        edge_distance = abs(segment.intercept - target_intercept)
        return segment.length + 0.7 * segment.vertical_span - 0.25 * edge_distance

    best = max(candidates, key=score)
    compatible = [
        segment
        for segment in candidates
        if abs(segment.slope - best.slope) <= 0.16
        and abs(segment.intercept - best.intercept) <= image_width * 0.045
    ]
    points = np.array(
        [point for segment in compatible for point in segment.points], dtype=np.float32
    )
    fitted = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    vx, vy, x0, y0 = (float(value) for value in fitted)
    if abs(vy) < 1e-6:
        return None
    slope = vx / vy
    intercept = x0 - slope * y0
    top = (intercept, 0.0)
    bottom = (intercept + slope * crop_y, float(crop_y))
    support = sum(segment.vertical_span for segment in compatible)
    confidence = min(1.0, support / max(1.0, crop_y * 0.55))
    return (top, bottom), confidence


def detect_deck_side_edges(frame: np.ndarray, crop_y: int) -> Optional[DeckSideEdges]:
    """Propose long converging side guides for manual laptop-plane calibration."""

    image_height, image_width = frame.shape[:2]
    crop_y = max(40, min(int(crop_y), image_height))
    search_height = max(40, min(crop_y, int(round(crop_y * 0.62))))
    gray = cv2.cvtColor(frame[:search_height], cv2.COLOR_BGR2GRAY)
    equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(cv2.GaussianBlur(equalized, (5, 5), 0), 25, 90)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 720,
        threshold=max(28, image_width // 50),
        minLineLength=max(60, int(image_width * 0.05)),
        maxLineGap=max(20, int(image_width * 0.045)),
    )
    if lines is None:
        return None

    all_segments = _segments(lines)
    left_candidates = []
    right_candidates = []
    for segment in all_segments:
        bottom_x = segment.intercept + segment.slope * crop_y
        if (
            0.10 <= segment.slope <= 1.25
            and -0.18 * image_width <= segment.intercept <= 0.24 * image_width
            and 0.02 * image_width <= bottom_x <= 0.60 * image_width
        ):
            left_candidates.append(segment)
        if (
            -1.25 <= segment.slope <= -0.10
            and 0.76 * image_width <= segment.intercept <= 1.18 * image_width
            and 0.40 * image_width <= bottom_x <= 0.98 * image_width
        ):
            right_candidates.append(segment)

    left = _fit_side(left_candidates, image_width, crop_y, "left")
    right = _fit_side(right_candidates, image_width, crop_y, "right")
    if left is None or right is None:
        return None
    left_line, left_confidence = left
    right_line, right_confidence = right
    if left_line[1][0] >= right_line[1][0]:
        return None
    return DeckSideEdges(
        left=left_line,
        right=right_line,
        confidence=min(left_confidence, right_confidence),
    )


def line_intersection(first: Line, second: Line) -> Optional[Point]:
    (x1, y1), (x2, y2) = first
    (x3, y3), (x4, y4) = second
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-8:
        return None
    first_cross = x1 * y2 - y1 * x2
    second_cross = x3 * y4 - y3 * x4
    x = (first_cross * (x3 - x4) - (x1 - x2) * second_cross) / denominator
    y = (first_cross * (y3 - y4) - (y1 - y2) * second_cross) / denominator
    return float(x), float(y)


def surface_quad_from_edges(trackpad_quad: Quad, edges: DeckSideEdges) -> Optional[Quad]:
    top_edge = (trackpad_quad[0], trackpad_quad[1])
    bottom_edge = (trackpad_quad[3], trackpad_quad[2])
    top_left = line_intersection(edges.left, top_edge)
    top_right = line_intersection(edges.right, top_edge)
    bottom_right = line_intersection(edges.right, bottom_edge)
    bottom_left = line_intersection(edges.left, bottom_edge)
    if None in (top_left, top_right, bottom_right, bottom_left):
        return None
    assert top_left is not None
    assert top_right is not None
    assert bottom_right is not None
    assert bottom_left is not None
    return top_left, top_right, bottom_right, bottom_left
