from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from ..config import Quad
from .types import CalibrationGeometry, DetectedLine, Line


class GeometryError(RuntimeError):
    """Raised when learned evidence does not support a trustworthy calibration."""


@dataclass(frozen=True)
class _Segment:
    index: int
    points: np.ndarray
    length: float
    angle: float

    @property
    def midpoint(self) -> np.ndarray:
        return np.mean(self.points, axis=0)


@dataclass(frozen=True)
class _LineGroup:
    indices: Tuple[int, ...]
    line: np.ndarray
    support: float
    extent_min: float
    extent_max: float


@dataclass(frozen=True)
class _ChassisCandidate:
    front: _Segment
    left: _Segment
    right: _Segment
    left_front: np.ndarray
    right_front: np.ndarray
    depth_vp: np.ndarray
    score: float


@dataclass(frozen=True)
class _TrackpadCandidate:
    left: _Segment
    right: _Segment
    top: _LineGroup
    bottom: _LineGroup
    quad: np.ndarray
    score: float


def _as_segments(lines: Sequence[DetectedLine] | np.ndarray) -> List[_Segment]:
    if isinstance(lines, np.ndarray):
        raw = np.asarray(lines, dtype=np.float64)
    else:
        raw = np.asarray([[line.first, line.second] for line in lines], dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1:] != (2, 2):
        raise ValueError("lines must have shape (N, 2, 2)")

    segments = []
    for index, points in enumerate(raw):
        if not np.all(np.isfinite(points)):
            continue
        delta = points[1] - points[0]
        length = float(np.linalg.norm(delta))
        if length < 1e-6:
            continue
        angle = math.atan2(float(delta[1]), float(delta[0])) % math.pi
        segments.append(_Segment(index, points, length, angle))
    return segments


def _homogeneous_line(points: np.ndarray) -> np.ndarray:
    first = np.array([points[0, 0], points[0, 1], 1.0], dtype=np.float64)
    second = np.array([points[1, 0], points[1, 1], 1.0], dtype=np.float64)
    line = np.cross(first, second)
    normal = float(np.hypot(line[0], line[1]))
    if normal < 1e-9:
        raise GeometryError("Degenerate line evidence")
    return line / normal


def _intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    point = np.cross(_homogeneous_line(first), _homogeneous_line(second))
    if abs(float(point[2])) < 1e-10:
        return None
    result = point[:2] / point[2]
    return result if np.all(np.isfinite(result)) else None


def _angle_difference(first: float, second: float) -> float:
    return abs((first - second + math.pi / 2.0) % math.pi - math.pi / 2.0)


def _point_line_distance(point: np.ndarray, line: np.ndarray) -> float:
    return abs(float(_homogeneous_line(line) @ np.array([point[0], point[1], 1.0])))


def _nearest_endpoint_distance(point: np.ndarray, segment: _Segment) -> float:
    return float(np.min(np.linalg.norm(segment.points - point[None, :], axis=1)))


def _fit_line(points: np.ndarray) -> np.ndarray:
    fitted = cv2.fitLine(
        np.asarray(points, dtype=np.float32), cv2.DIST_L2, 0, 0.01, 0.01
    ).reshape(-1)
    vx, vy, x0, y0 = (float(value) for value in fitted)
    direction = np.array([vx, vy], dtype=np.float64)
    center = np.array([x0, y0], dtype=np.float64)
    return np.stack([center - 2000.0 * direction, center + 2000.0 * direction])


def _line_y(line: np.ndarray, x: float) -> float:
    homogeneous = _homogeneous_line(line)
    if abs(float(homogeneous[1])) < 1e-9:
        return float(np.mean(line[:, 1]))
    return float(-(homogeneous[0] * x + homogeneous[2]) / homogeneous[1])


def _projection_interval(points: np.ndarray, direction: np.ndarray) -> Tuple[float, float]:
    projected = np.asarray(points, dtype=np.float64) @ direction
    return float(np.min(projected)), float(np.max(projected))


def _interval_gap(first: Tuple[float, float], second: Tuple[float, float]) -> float:
    return max(0.0, max(first[0], second[0]) - min(first[1], second[1]))


def _select_chassis(
    segments: Sequence[_Segment], image_width: int, image_height: int
) -> _ChassisCandidate:
    # Prefer the two full outer sides. Their lower endpoints directly identify
    # the physical front corners even when the dark front edge is only visible
    # in short fragments. This prevents a long keyboard boundary from being
    # mistaken for the chassis front.
    left_sides = []
    right_sides = []
    minimum_side_length = max(100.0, image_height * 0.32)
    for segment in segments:
        if segment.length < minimum_side_length:
            continue
        top = segment.points[int(np.argmin(segment.points[:, 1]))]
        bottom = segment.points[int(np.argmax(segment.points[:, 1]))]
        if bottom[1] < image_height * 0.25:
            continue
        if (
            top[0] <= image_width * 0.22
            and bottom[0] < image_width * 0.46
            and bottom[0] > top[0] + 18.0
        ):
            left_sides.append((segment, top, bottom))
        if (
            top[0] >= image_width * 0.78
            and bottom[0] > image_width * 0.54
            and bottom[0] < top[0] - 18.0
        ):
            right_sides.append((segment, top, bottom))

    side_first_candidates = []
    for left, _left_top, left_bottom in left_sides:
        for right, _right_top, right_bottom in right_sides:
            if abs(float(left_bottom[1] - right_bottom[1])) > image_height * 0.12:
                continue
            span = float(right_bottom[0] - left_bottom[0])
            if span < image_width * 0.50:
                continue
            front_points = np.stack([left_bottom, right_bottom]).astype(np.float64)
            front_delta = front_points[1] - front_points[0]
            front_angle = math.atan2(float(front_delta[1]), float(front_delta[0])) % math.pi
            if min(front_angle, math.pi - front_angle) > math.radians(20.0):
                continue
            depth_vp = _intersection(left.points, right.points)
            if depth_vp is None:
                continue
            front_y = float(np.mean(front_points[:, 1]))
            if depth_vp[1] < front_y + image_height * 0.25:
                continue
            front = _Segment(
                index=-1,
                points=front_points,
                length=float(np.linalg.norm(front_delta)),
                angle=front_angle,
            )
            center_error = abs(
                float((left_bottom[0] + right_bottom[0]) / 2.0) - image_width / 2.0
            ) / image_width
            score = (
                1.6 * (left.length + right.length) / image_height
                + 1.8 * span / image_width
                + 0.8 * front_y / image_height
                - center_error
            )
            side_first_candidates.append(
                _ChassisCandidate(
                    front,
                    left,
                    right,
                    left_bottom,
                    right_bottom,
                    depth_vp,
                    score,
                )
            )
    if side_first_candidates:
        return max(side_first_candidates, key=lambda candidate: candidate.score)

    # Fall back to a directly detected long front line when one or both outer
    # sides are clipped by the camera frame.
    diagonal = float(np.hypot(image_width, image_height))
    front_candidates = [
        segment
        for segment in segments
        if segment.length >= image_width * 0.34
        and min(segment.angle, math.pi - segment.angle) <= math.radians(20.0)
        and image_height * 0.10 <= segment.midpoint[1] <= image_height * 0.62
    ]
    if not front_candidates:
        raise GeometryError("没有找到有足够像素支持的电脑前沿")

    candidates: List[_ChassisCandidate] = []
    for front in front_candidates:
        front_y = float(front.midpoint[1])
        sides = []
        for segment in segments:
            if segment.index == front.index or segment.length < image_height * 0.18:
                continue
            if _angle_difference(segment.angle, front.angle) < math.radians(25.0):
                continue
            cross = _intersection(front.points, segment.points)
            if cross is None:
                continue
            if not (-0.08 * image_width <= cross[0] <= 1.08 * image_width):
                continue
            if abs(float(cross[1]) - front_y) > image_height * 0.13:
                continue
            top = segment.points[int(np.argmin(segment.points[:, 1]))]
            if top[1] > front_y - image_height * 0.12:
                continue
            if _nearest_endpoint_distance(cross, segment) > diagonal * 0.13:
                continue
            sides.append((segment, cross, top))

        for left, left_cross, left_top in sides:
            if left_cross[0] >= image_width * 0.48 or left_top[0] >= left_cross[0] - 8.0:
                continue
            for right, right_cross, right_top in sides:
                if right.index == left.index:
                    continue
                if right_cross[0] <= image_width * 0.52 or right_top[0] <= right_cross[0] + 8.0:
                    continue
                span = float(right_cross[0] - left_cross[0])
                if span < image_width * 0.50:
                    continue
                depth_vp = _intersection(left.points, right.points)
                if depth_vp is None:
                    continue
                if depth_vp[1] < front_y + image_height * 0.30:
                    continue
                if not (-0.5 * image_width <= depth_vp[0] <= 1.5 * image_width):
                    continue
                end_error = (
                    _nearest_endpoint_distance(left_cross, left)
                    + _nearest_endpoint_distance(right_cross, right)
                ) / diagonal
                center_error = abs(
                    float((left_cross[0] + right_cross[0]) / 2.0) - image_width / 2.0
                ) / image_width
                score = (
                    2.4 * span / image_width
                    + 0.8 * (left.length + right.length) / image_height
                    + 0.7 * front.length / image_width
                    - 1.7 * end_error
                    - 0.6 * center_error
                )
                candidates.append(
                    _ChassisCandidate(
                        front,
                        left,
                        right,
                        left_cross,
                        right_cross,
                        depth_vp,
                        score,
                    )
                )

    if not candidates:
        raise GeometryError("检测到了直线，但无法组成可信的电脑左边、右边和前沿")
    return max(candidates, key=lambda candidate: candidate.score)


def _horizontal_groups(
    segments: Sequence[_Segment], front: _Segment, image_width: int
) -> List[_LineGroup]:
    minimum_length = max(28.0, image_width * 0.025)
    candidates = [
        segment
        for segment in segments
        if segment.length >= minimum_length
        and _angle_difference(segment.angle, front.angle) <= math.radians(7.0)
    ]
    if not candidates:
        return []

    direction = front.points[1] - front.points[0]
    direction /= np.linalg.norm(direction)
    if direction[0] < 0:
        direction *= -1
    parents = list(range(len(candidates)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    intervals = [_projection_interval(segment.points, direction) for segment in candidates]
    for first_index, first in enumerate(candidates):
        for second_index in range(first_index + 1, len(candidates)):
            second = candidates[second_index]
            if _angle_difference(first.angle, second.angle) > math.radians(2.5):
                continue
            distance = min(
                _point_line_distance(first.midpoint, second.points),
                _point_line_distance(second.midpoint, first.points),
            )
            if distance > 4.2:
                continue
            if _interval_gap(intervals[first_index], intervals[second_index]) > image_width * 0.04:
                continue
            union(first_index, second_index)

    members: dict[int, List[_Segment]] = {}
    for index, segment in enumerate(candidates):
        members.setdefault(find(index), []).append(segment)

    groups = []
    for group_segments in members.values():
        points = np.concatenate([segment.points for segment in group_segments], axis=0)
        fitted = _fit_line(points)
        extent_min = min(
            _projection_interval(segment.points, direction)[0] for segment in group_segments
        )
        extent_max = max(
            _projection_interval(segment.points, direction)[1] for segment in group_segments
        )
        groups.append(
            _LineGroup(
                indices=tuple(segment.index for segment in group_segments),
                line=fitted,
                support=sum(segment.length for segment in group_segments),
                extent_min=extent_min,
                extent_max=extent_max,
            )
        )
    return groups


def _select_trackpad_bands(
    chassis: _ChassisCandidate,
    groups: Sequence[_LineGroup],
    image_width: int,
    image_height: int,
) -> Tuple[_LineGroup, _LineGroup]:
    """Select trackpad top/bottom evidence before its faint side edges are known."""

    outer_span = float(chassis.right_front[0] - chassis.left_front[0])
    chassis_center = float((chassis.right_front[0] + chassis.left_front[0]) / 2.0)
    front_y = _line_y(chassis.front.points, chassis_center)
    direction = chassis.front.points[1] - chassis.front.points[0]
    direction /= np.linalg.norm(direction)
    if direction[0] < 0:
        direction *= -1
    projected_center = float(
        np.array([chassis_center, front_y], dtype=np.float64) @ direction
    )
    candidates = []
    for first_index, first in enumerate(groups):
        if chassis.front.index in first.indices:
            continue
        for second in groups[first_index + 1 :]:
            if chassis.front.index in second.indices:
                continue
            first_y = _line_y(first.line, chassis_center)
            second_y = _line_y(second.line, chassis_center)
            top, bottom = (first, second) if first_y < second_y else (second, first)
            top_y, bottom_y = sorted((first_y, second_y))
            if top_y < -5.0 or bottom_y > front_y + image_height * 0.02:
                continue
            if bottom_y - top_y < image_height * 0.16:
                continue
            if front_y - bottom_y > image_height * 0.09:
                continue
            top_extent = top.extent_max - top.extent_min
            bottom_extent = bottom.extent_max - bottom.extent_min
            if not outer_span * 0.15 <= top_extent <= outer_span * 0.80:
                continue
            if not outer_span * 0.15 <= bottom_extent <= outer_span * 0.80:
                continue
            top_center = (top.extent_min + top.extent_max) / 2.0
            bottom_center = (bottom.extent_min + bottom.extent_max) / 2.0
            center_error = (
                abs(top_center - projected_center) + abs(bottom_center - projected_center)
            ) / outer_span
            if center_error > 0.55:
                continue
            score = (
                min(2.0, top.support / max(1.0, outer_span * 0.30))
                + min(2.0, bottom.support / max(1.0, outer_span * 0.30))
                - 2.0 * center_error
                - 2.0 * max(0.0, front_y - bottom_y) / image_height
            )
            candidates.append((score, top, bottom))
    if not candidates:
        raise GeometryError("没有找到可约束触控板侧边的上下沿")
    _, top, bottom = max(candidates, key=lambda candidate: candidate[0])
    return top, bottom


def add_constrained_gradient_sides(frame: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Recover a faint trackpad rectangle inside model-detected chassis geometry.

    Side candidates must pass through the chassis depth vanishing point. Top and
    bottom candidates must cross those two rays in the chassis-width direction.
    Every selected edge is then backed by an integrated image-gradient peak.
    """

    raw_lines = np.asarray(lines, dtype=np.float32)
    image_height, image_width = frame.shape[:2]
    segments = _as_segments(raw_lines)
    chassis = _select_chassis(segments, image_width, image_height)
    groups = _horizontal_groups(segments, chassis.front, image_width)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    gradient_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)

    outer_span = float(chassis.right_front[0] - chassis.left_front[0])
    chassis_center = float((chassis.right_front[0] + chassis.left_front[0]) / 2.0)
    front_y = _line_y(chassis.front.points, chassis_center)
    left_top = chassis.left.points[int(np.argmin(chassis.left.points[:, 1]))]
    right_top = chassis.right.points[int(np.argmin(chassis.right.points[:, 1]))]
    outer_top_y = float((left_top[1] + right_top[1]) / 2.0)
    deck_depth = max(1.0, front_y - outer_top_y)

    guide_candidates = []
    for group in groups:
        guide_y = _line_y(group.line, chassis_center)
        extent = group.extent_max - group.extent_min
        if not outer_top_y - image_height * 0.04 <= guide_y:
            continue
        if guide_y > front_y - deck_depth * 0.30:
            continue
        if extent < outer_span * 0.55:
            continue
        score = (
            1.8 * (guide_y - outer_top_y) / deck_depth
            + min(2.0, group.support / max(1.0, outer_span))
            + min(1.2, extent / max(1.0, outer_span))
        )
        guide_candidates.append((score, group))
    if not guide_candidates:
        raise GeometryError("没有找到横跨机身的键盘下沿，无法约束触控板搜索范围")
    _, guide = max(guide_candidates, key=lambda candidate: candidate[0])

    def scan_sides(
        first_x: float, last_x: float
    ) -> List[Tuple[float, float, np.ndarray, np.ndarray]]:
        candidates = []
        for front_x in np.linspace(first_x, last_x, 180):
            front_point = np.array(
                [front_x, _line_y(chassis.front.points, float(front_x))], dtype=np.float64
            )
            ray = np.stack([chassis.depth_vp, front_point])
            guide_point = _intersection(ray, guide.line)
            chassis_point = _intersection(ray, chassis.front.points)
            if guide_point is None or chassis_point is None:
                continue
            tangent = chassis_point - guide_point
            length = float(np.linalg.norm(tangent))
            if length < image_height * 0.15:
                continue
            tangent /= length
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
            responses = []
            scan_start = guide_point * 0.92 + chassis_point * 0.08
            scan_end = guide_point * 0.05 + chassis_point * 0.95
            for point in np.linspace(scan_start, scan_end, max(30, int(length * 0.87))):
                local = []
                for offset in range(-3, 4):
                    x, y = np.rint(point + normal * offset).astype(int)
                    if 1 <= x < image_width - 1 and 1 <= y < image_height - 1:
                        response = abs(
                            float(
                                gradient_x[y, x] * normal[0]
                                + gradient_y[y, x] * normal[1]
                            )
                        )
                        local.append(response)
                if local:
                    responses.append(max(local))
            if responses:
                candidates.append(
                    (
                        float(np.mean(responses)),
                        float(front_x),
                        guide_point,
                        chassis_point,
                    )
                )
        return candidates

    left_candidates = scan_sides(
        float(chassis.left_front[0] + outer_span * 0.18),
        chassis_center - outer_span * 0.07,
    )
    right_candidates = scan_sides(
        chassis_center + outer_span * 0.07,
        float(chassis.right_front[0] - outer_span * 0.18),
    )
    if not left_candidates or not right_candidates:
        raise GeometryError("触控板侧边梯度搜索区域为空")

    left_background = max(1.0, float(np.median([item[0] for item in left_candidates])))
    right_background = max(1.0, float(np.median([item[0] for item in right_candidates])))
    pairs = []
    best_left = sorted(left_candidates, key=lambda candidate: candidate[0], reverse=True)[:20]
    best_right = sorted(right_candidates, key=lambda candidate: candidate[0], reverse=True)[:20]
    for left in best_left:
        for right in best_right:
            width_ratio = (right[1] - left[1]) / outer_span
            if not 0.24 <= width_ratio <= 0.66:
                continue
            pair_center = (left[1] + right[1]) / 2.0
            center_error = abs(pair_center - chassis_center) / outer_span
            if center_error > 0.15:
                continue
            score = (
                left[0] / left_background
                + right[0] / right_background
                - 6.0 * center_error
                - abs(width_ratio - 0.43)
            )
            pairs.append((score, left, right))
    if not pairs:
        raise GeometryError("触控板左右侧边的梯度候选无法组成居中的透视四边形")
    _, left, right = max(pairs, key=lambda pair: pair[0])
    left_contrast = left[0] / left_background
    right_contrast = right[0] / right_background
    if left[0] < 10.0 or right[0] < 10.0 or min(left_contrast, right_contrast) < 1.65:
        raise GeometryError(
            "触控板侧边像素证据不足"
            f"（响应 {left[0]:.1f}/{right[0]:.1f}，"
            f"对比度 {left_contrast:.2f}/{right_contrast:.2f}）"
        )

    def cross_edge_score(amount: float) -> Tuple[float, float, np.ndarray, np.ndarray]:
        left_point = left[2] * (1.0 - amount) + left[3] * amount
        right_point = right[2] * (1.0 - amount) + right[3] * amount
        tangent = right_point - left_point
        length = float(np.linalg.norm(tangent))
        tangent /= max(1.0, length)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        responses = []
        for point in np.linspace(left_point, right_point, max(60, int(length))):
            local = []
            for offset in range(-3, 4):
                x, y = np.rint(point + normal * offset).astype(int)
                if 1 <= x < image_width - 1 and 1 <= y < image_height - 1:
                    response = abs(
                        float(
                            gradient_x[y, x] * normal[0]
                            + gradient_y[y, x] * normal[1]
                        )
                    )
                    local.append(response)
            if local:
                responses.append(max(local))
        score = float(np.mean(responses)) if responses else 0.0
        return score, amount, left_point, right_point

    top_candidates = [
        cross_edge_score(amount) for amount in np.linspace(0.035, 0.32, 100)
    ]
    bottom_candidates = [
        cross_edge_score(amount) for amount in np.linspace(0.68, 0.975, 110)
    ]
    top_edge = max(top_candidates, key=lambda candidate: candidate[0])
    bottom_edge = max(bottom_candidates, key=lambda candidate: candidate[0])
    top_background = max(1.0, float(np.median([item[0] for item in top_candidates])))
    bottom_background = max(
        1.0, float(np.median([item[0] for item in bottom_candidates]))
    )
    top_contrast = top_edge[0] / top_background
    bottom_contrast = bottom_edge[0] / bottom_background
    if (
        top_edge[0] < 10.0
        or bottom_edge[0] < 10.0
        or min(top_contrast, bottom_contrast) < 1.55
    ):
        raise GeometryError(
            "触控板上下沿像素证据不足"
            f"（响应 {top_edge[0]:.1f}/{bottom_edge[0]:.1f}，"
            f"对比度 {top_contrast:.2f}/{bottom_contrast:.2f}）"
        )

    top_left, top_right = top_edge[2], top_edge[3]
    bottom_left, bottom_right = bottom_edge[2], bottom_edge[3]
    recovered = np.asarray(
        [
            [top_left, bottom_left],
            [top_right, bottom_right],
            [top_left, top_right],
            [bottom_left, bottom_right],
        ],
        dtype=np.float32,
    )
    return np.concatenate([raw_lines, recovered], axis=0)


def _select_trackpad(
    segments: Sequence[_Segment],
    chassis: _ChassisCandidate,
    groups: Sequence[_LineGroup],
    image_width: int,
    image_height: int,
) -> _TrackpadCandidate:
    diagonal = float(np.hypot(image_width, image_height))
    outer_span = float(chassis.right_front[0] - chassis.left_front[0])
    chassis_center = float((chassis.right_front[0] + chassis.left_front[0]) / 2.0)
    front_y = _line_y(chassis.front.points, chassis_center)

    side_candidates = []
    excluded = {chassis.left.index, chassis.right.index, chassis.front.index}
    for segment in segments:
        if segment.index in excluded or segment.length < max(48.0, front_y * 0.27):
            continue
        if _angle_difference(segment.angle, chassis.front.angle) < math.radians(22.0):
            continue
        vp_error = _point_line_distance(chassis.depth_vp, segment.points)
        if vp_error > max(12.0, diagonal * 0.018):
            continue
        cross = _intersection(chassis.front.points, segment.points)
        if cross is None:
            continue
        if not (
            chassis.left_front[0] + outer_span * 0.10
            < cross[0]
            < chassis.right_front[0] - outer_span * 0.10
        ):
            continue
        if float(np.min(segment.points[:, 1])) > front_y - image_height * 0.12:
            continue
        side_candidates.append((segment, cross, vp_error))

    pairs = []
    for left, left_cross, left_error in side_candidates:
        if left_cross[0] >= chassis_center:
            continue
        for right, right_cross, right_error in side_candidates:
            if right_cross[0] <= chassis_center or right.index == left.index:
                continue
            width = float(right_cross[0] - left_cross[0])
            width_ratio = width / outer_span
            if not 0.24 <= width_ratio <= 0.66:
                continue
            pair_center = float((left_cross[0] + right_cross[0]) / 2.0)
            center_error = abs(pair_center - chassis_center) / outer_span
            score = (
                0.9 * (left.length + right.length) / max(1.0, front_y)
                - 2.5 * (left_error + right_error) / diagonal
                - 1.5 * center_error
                - 0.5 * abs(width_ratio - 0.44)
            )
            pairs.append((score, left, right))
    if not pairs:
        raise GeometryError("机身三边已找到，但没有找到共享同一透视方向的触控板左右边")

    trackpad_candidates: List[_TrackpadCandidate] = []
    for side_score, left, right in pairs:
        for first_index, first in enumerate(groups):
            if chassis.front.index in first.indices:
                continue
            for second in groups[first_index + 1 :]:
                if chassis.front.index in second.indices:
                    continue
                first_y = _line_y(first.line, chassis_center)
                second_y = _line_y(second.line, chassis_center)
                top, bottom = (first, second) if first_y < second_y else (second, first)
                top_y, bottom_y = sorted((first_y, second_y))
                if top_y < -5.0 or bottom_y > front_y + image_height * 0.02:
                    continue
                if bottom_y - top_y < image_height * 0.16:
                    continue
                if front_y - bottom_y > image_height * 0.08:
                    continue

                corners = [
                    _intersection(left.points, top.line),
                    _intersection(right.points, top.line),
                    _intersection(right.points, bottom.line),
                    _intersection(left.points, bottom.line),
                ]
                if any(point is None for point in corners):
                    continue
                quad = np.asarray(corners, dtype=np.float64)
                if not np.all(np.isfinite(quad)):
                    continue
                if not cv2.isContourConvex(quad.astype(np.float32)):
                    continue
                if np.any(quad[:, 0] < -0.03 * image_width) or np.any(
                    quad[:, 0] > 1.03 * image_width
                ):
                    continue

                top_width = float(np.linalg.norm(quad[1] - quad[0]))
                bottom_width = float(np.linalg.norm(quad[2] - quad[3]))
                mean_width = (top_width + bottom_width) / 2.0
                left_height = float(np.linalg.norm(quad[3] - quad[0]))
                right_height = float(np.linalg.norm(quad[2] - quad[1]))
                mean_height = (left_height + right_height) / 2.0
                if not image_width * 0.22 <= mean_width <= image_width * 0.68:
                    continue
                if not 1.7 <= mean_width / max(1.0, mean_height) <= 6.5:
                    continue
                area = abs(float(cv2.contourArea(quad.astype(np.float32))))
                minimum_area = image_width * image_height * 0.035
                maximum_area = image_width * image_height * 0.22
                if not minimum_area <= area <= maximum_area:
                    continue

                direction = chassis.front.points[1] - chassis.front.points[0]
                direction /= np.linalg.norm(direction)
                if direction[0] < 0:
                    direction *= -1
                top_interval = _projection_interval(quad[:2], direction)
                bottom_interval = _projection_interval(quad[2:][::-1], direction)
                top_extent = top.extent_max - top.extent_min
                bottom_extent = bottom.extent_max - bottom.extent_min
                if top_extent > (top_interval[1] - top_interval[0]) * 1.55:
                    continue
                if bottom_extent > (bottom_interval[1] - bottom_interval[0]) * 1.55:
                    continue
                top_gap = _interval_gap((top.extent_min, top.extent_max), top_interval)
                if top_gap > mean_width * 0.18:
                    continue
                if (
                    _interval_gap((bottom.extent_min, bottom.extent_max), bottom_interval)
                    > mean_width * 0.18
                ):
                    continue

                quad_center = float(np.mean(quad[:, 0]))
                center_error = abs(quad_center - chassis_center) / outer_span
                top_coverage = min(1.0, top_extent / max(1.0, top_width))
                bottom_coverage = min(1.0, bottom_extent / max(1.0, bottom_width))
                front_gap = max(0.0, front_y - bottom_y) / image_height
                score = (
                    side_score
                    + 1.5 * top_coverage
                    + 1.5 * bottom_coverage
                    + 0.25 * min(2.0, top.support / max(1.0, top_width))
                    + 0.25 * min(2.0, bottom.support / max(1.0, bottom_width))
                    - 2.0 * center_error
                    - 2.0 * front_gap
                )
                trackpad_candidates.append(
                    _TrackpadCandidate(left, right, top, bottom, quad, score)
                )

    if not trackpad_candidates:
        raise GeometryError("触控板候选没有通过四边形、位置和像素支持检查")
    return max(trackpad_candidates, key=lambda candidate: candidate.score)


def _to_line(first: np.ndarray, second: np.ndarray) -> Line:
    return (float(first[0]), float(first[1])), (float(second[0]), float(second[1]))


def _to_quad(points: np.ndarray) -> Quad:
    return tuple((float(point[0]), float(point[1])) for point in points)  # type: ignore[return-value]


def solve_geometry(
    lines: Sequence[DetectedLine] | np.ndarray,
    image_shape: Tuple[int, int] | Tuple[int, int, int],
    minimum_confidence: float = 0.72,
) -> CalibrationGeometry:
    """Fit chassis and trackpad geometry from learned finite line segments.

    The side edges are not required to be perpendicular in image pixels. Instead,
    the two chassis sides and two trackpad sides must agree on one projective
    vanishing point, which is the correct constraint for parallel directions on
    the physical laptop plane.
    """

    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    if image_width < 160 or image_height < 100:
        raise ValueError("image is too small for calibration")
    segments = _as_segments(lines)
    if len(segments) < 7:
        raise GeometryError("强模型返回的有效长线太少")

    chassis = _select_chassis(segments, image_width, image_height)
    groups = _horizontal_groups(segments, chassis.front, image_width)
    trackpad = _select_trackpad(segments, chassis, groups, image_width, image_height)

    top_line = trackpad.top.line
    bottom_line = trackpad.bottom.line
    surface_points = [
        _intersection(chassis.left.points, top_line),
        _intersection(chassis.right.points, top_line),
        _intersection(chassis.right.points, bottom_line),
        _intersection(chassis.left.points, bottom_line),
    ]
    if any(point is None for point in surface_points):
        raise GeometryError("无法把触控板深度带投影到电脑外框")
    surface = np.asarray(surface_points, dtype=np.float64)

    diagonal = float(np.hypot(image_width, image_height))
    vp_residual_pixels = float(
        np.mean(
            [
                _point_line_distance(chassis.depth_vp, trackpad.left.points),
                _point_line_distance(chassis.depth_vp, trackpad.right.points),
            ]
        )
    )
    outer_span = float(chassis.right_front[0] - chassis.left_front[0])
    chassis_center = float((chassis.right_front[0] + chassis.left_front[0]) / 2.0)
    trackpad_width = float(
        (
            np.linalg.norm(trackpad.quad[1] - trackpad.quad[0])
            + np.linalg.norm(trackpad.quad[2] - trackpad.quad[3])
        )
        / 2.0
    )
    direction = chassis.front.points[1] - chassis.front.points[0]
    direction /= np.linalg.norm(direction)
    if direction[0] < 0:
        direction *= -1
    top_coverage = min(
        1.0, (trackpad.top.extent_max - trackpad.top.extent_min) / max(1.0, trackpad_width)
    )
    bottom_coverage = min(
        1.0,
        (trackpad.bottom.extent_max - trackpad.bottom.extent_min) / max(1.0, trackpad_width),
    )
    center_error = abs(float(np.mean(trackpad.quad[:, 0])) - chassis_center) / outer_span
    chassis_quality = min(1.0, chassis.front.length / (image_width * 0.55)) * min(
        1.0, outer_span / (image_width * 0.62)
    )
    side_quality = min(
        1.0,
        min(chassis.left.length, chassis.right.length) / max(1.0, image_height * 0.40),
    )
    vanishing_quality = math.exp(-vp_residual_pixels / max(2.0, diagonal * 0.008))
    evidence_quality = math.sqrt(max(0.0, top_coverage * bottom_coverage))
    placement_quality = max(0.0, 1.0 - center_error / 0.14)
    confidence = (
        0.25 * chassis_quality
        + 0.15 * side_quality
        + 0.25 * vanishing_quality
        + 0.25 * evidence_quality
        + 0.10 * placement_quality
    )

    metrics = {
        "chassis_front_support_ratio": chassis.front.length / image_width,
        "chassis_front_span_ratio": outer_span / image_width,
        "depth_vanishing_residual_px": vp_residual_pixels,
        "trackpad_top_coverage": top_coverage,
        "trackpad_bottom_coverage": bottom_coverage,
        "trackpad_center_error_ratio": center_error,
        "solver_score": trackpad.score,
    }
    if confidence < minimum_confidence:
        raise GeometryError(
            f"几何解置信度 {confidence:.2f} 低于保存门槛 {minimum_confidence:.2f}"
        )

    left_top = chassis.left.points[int(np.argmin(chassis.left.points[:, 1]))]
    right_top = chassis.right.points[int(np.argmin(chassis.right.points[:, 1]))]
    evidence = tuple(
        DetectedLine(
            (float(segment.points[0, 0]), float(segment.points[0, 1])),
            (float(segment.points[1, 0]), float(segment.points[1, 1])),
        )
        for segment in segments
    )
    return CalibrationGeometry(
        chassis_left=_to_line(left_top, chassis.left_front),
        chassis_right=_to_line(right_top, chassis.right_front),
        chassis_front=_to_line(chassis.left_front, chassis.right_front),
        trackpad_quad=_to_quad(trackpad.quad),
        surface_quad=_to_quad(surface),
        confidence=float(confidence),
        metrics=metrics,
        evidence_lines=evidence,
    )


def median_geometry(geometries: Iterable[CalibrationGeometry]) -> CalibrationGeometry:
    """Aggregate independently solved frames and report their corner stability."""

    values = list(geometries)
    if not values:
        raise GeometryError("没有可聚合的校准结果")
    if len(values) == 1:
        return values[0]

    def median_line(name: str) -> Line:
        array = np.asarray([getattr(value, name) for value in values], dtype=np.float64)
        result = np.median(array, axis=0)
        return _to_line(result[0], result[1])

    trackpads = np.asarray([value.trackpad_quad for value in values], dtype=np.float64)
    surfaces = np.asarray([value.surface_quad for value in values], dtype=np.float64)
    median_trackpad = np.median(trackpads, axis=0)
    median_surface = np.median(surfaces, axis=0)
    deviations = np.linalg.norm(trackpads - median_trackpad[None, :, :], axis=2)
    corner_jitter = float(np.max(np.median(deviations, axis=0)))
    image_scale = max(1.0, float(np.ptp(median_surface[:, 0])))
    stability = math.exp(-corner_jitter / max(2.0, image_scale * 0.008))
    confidence = float(np.median([value.confidence for value in values])) * stability
    if confidence < 0.72:
        raise GeometryError(
            f"多帧角点抖动 {corner_jitter:.1f}px，聚合置信度只有 {confidence:.2f}"
        )
    metrics = dict(values[len(values) // 2].metrics)
    metrics["frame_count"] = float(len(values))
    metrics["corner_jitter_px"] = corner_jitter
    metrics["temporal_stability"] = stability
    return CalibrationGeometry(
        chassis_left=median_line("chassis_left"),
        chassis_right=median_line("chassis_right"),
        chassis_front=median_line("chassis_front"),
        trackpad_quad=_to_quad(median_trackpad),
        surface_quad=_to_quad(median_surface),
        confidence=confidence,
        metrics=metrics,
        evidence_lines=values[len(values) // 2].evidence_lines,
    )
