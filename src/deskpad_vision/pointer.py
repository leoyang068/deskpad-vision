from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

Point = Tuple[float, float]


class PointerSink(Protocol):
    def move_relative(self, dx: float, dy: float) -> None: ...


class NullPointerSink:
    def __init__(self) -> None:
        self.last_delta: Point = (0.0, 0.0)

    def move_relative(self, dx: float, dy: float) -> None:
        self.last_delta = (dx, dy)


class MacPointerSink:
    def __init__(self) -> None:
        try:
            from Quartz import (
                CGEventCreate,
                CGEventCreateMouseEvent,
                CGEventGetLocation,
                CGEventPost,
                kCGEventMouseMoved,
                kCGHIDEventTap,
                kCGMouseButtonLeft,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Quartz bindings are unavailable. Install the project dependencies on macOS."
            ) from exc
        self._create_event = CGEventCreate
        self._create_mouse_event = CGEventCreateMouseEvent
        self._get_location = CGEventGetLocation
        self._post = CGEventPost
        self._mouse_moved = kCGEventMouseMoved
        self._event_tap = kCGHIDEventTap
        self._left_button = kCGMouseButtonLeft

    def move_relative(self, dx: float, dy: float) -> None:
        source_event = self._create_event(None)
        current = self._get_location(source_event)
        target = (float(current.x + dx), float(current.y + dy))
        event = self._create_mouse_event(None, self._mouse_moved, target, self._left_button)
        self._post(self._event_tap, event)


@dataclass
class RelativePointerMapper:
    sink: PointerSink
    gain: float = 1.5
    max_jump: float = 70.0
    smoothing_alpha: float = 0.55
    invert_x: bool = False
    invert_y: bool = False
    _last_point: Optional[Point] = None
    _smoothed_point: Optional[Point] = None
    _zone: Optional[str] = None

    def reset(self) -> None:
        self._last_point = None
        self._smoothed_point = None
        self._zone = None

    def update(self, point: Point, zone: Optional[str], clutch: bool) -> Optional[Point]:
        if not clutch or zone is None:
            self.reset()
            return None

        if zone != self._zone or self._smoothed_point is None:
            self._zone = zone
            self._last_point = point
            self._smoothed_point = point
            return (0.0, 0.0)

        alpha = self.smoothing_alpha
        smoothed = (
            alpha * point[0] + (1.0 - alpha) * self._smoothed_point[0],
            alpha * point[1] + (1.0 - alpha) * self._smoothed_point[1],
        )
        dx = smoothed[0] - self._smoothed_point[0]
        dy = smoothed[1] - self._smoothed_point[1]
        self._last_point = point
        self._smoothed_point = smoothed

        if abs(dx) > self.max_jump or abs(dy) > self.max_jump:
            self.reset()
            return None
        if self.invert_x:
            dx = -dx
        if self.invert_y:
            dy = -dy
        dx *= self.gain
        dy *= self.gain
        self.sink.move_relative(dx, dy)
        return dx, dy
