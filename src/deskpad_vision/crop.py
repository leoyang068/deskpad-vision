from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np


def _row_gradient_candidate(gray: np.ndarray, min_fraction: float, max_fraction: float) -> int:
    height, width = gray.shape
    y0 = int(height * min_fraction)
    y1 = max(y0 + 1, int(height * max_fraction))
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    vertical = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    # A trimmed horizontal response is more robust than the mean when a hand is present.
    abs_vertical = np.abs(vertical[:, int(width * 0.08) : int(width * 0.92)])
    scores = np.percentile(abs_vertical, 70, axis=1)
    scores = cv2.GaussianBlur(scores.reshape(-1, 1), (1, 15), 0).reshape(-1)
    return int(y0 + np.argmax(scores[y0:y1]))


def detect_horizontal_boundary(
    image: np.ndarray,
    min_fraction: float = 0.55,
    max_fraction: float = 0.95,
) -> int:
    """Find the persistent lower mirror boundary in a BGR or grayscale frame."""

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.astype(np.uint8, copy=False)

    height, width = gray.shape
    y0 = int(height * min_fraction)
    y1 = int(height * max_fraction)
    candidate_from_rows = _row_gradient_candidate(gray, min_fraction, max_fraction)

    roi = gray[y0:y1]
    edges = cv2.Canny(cv2.GaussianBlur(roi, (5, 5), 0), 35, 110)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(35, width // 12),
        minLineLength=int(width * 0.42),
        maxLineGap=int(width * 0.10),
    )

    scored = []
    if lines is not None:
        for raw in lines[:, 0, :]:
            x1, local_y1, x2, local_y2 = [int(value) for value in raw]
            dx = x2 - x1
            dy = local_y2 - local_y1
            length = float(np.hypot(dx, dy))
            if length < width * 0.42 or abs(dy) > max(4, abs(dx) * 0.08):
                continue
            absolute_y = y0 + int(round((local_y1 + local_y2) / 2))
            row_distance = abs(absolute_y - candidate_from_rows)
            score = length - 2.0 * row_distance
            scored.append((score, absolute_y))

    if scored:
        scored.sort(reverse=True)
        return int(scored[0][1])
    return candidate_from_rows


@dataclass
class BoundarySampler:
    sample_count: int = 24
    _frames: List[np.ndarray] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return len(self._frames) >= self.sample_count

    @property
    def progress(self) -> int:
        return len(self._frames)

    def add(self, frame: np.ndarray) -> Optional[int]:
        if self.ready:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._frames.append(gray)
        if not self.ready:
            return None
        median = np.median(np.stack(self._frames, axis=0), axis=0).astype(np.uint8)
        return detect_horizontal_boundary(median)
