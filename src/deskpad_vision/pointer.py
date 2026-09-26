from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

Point = Tuple[float, float]


class PointerSink(Protocol):
    def move_relative(self, dx: float, dy: float) -> None: ...

    def left_click(self) -> None: ...

    def left_down(self) -> None: ...

    def left_up(self) -> None: ...


class NullPointerSink:
    def __init__(self) -> None:
        self.last_delta: Point = (0.0, 0.0)
        self.events: list[str] = []
        self.left_is_down = False

    def move_relative(self, dx: float, dy: float) -> None:
        self.last_delta = (dx, dy)

    def left_click(self) -> None:
        self.events.append("click")

    def left_down(self) -> None:
        if not self.left_is_down:
            self.left_is_down = True
            self.events.append("down")

    def left_up(self) -> None:
        if self.left_is_down:
            self.left_is_down = False
            self.events.append("up")


class MacPointerSink:
    def __init__(self) -> None:
        try:
            from Quartz import (
                CGEventCreate,
                CGEventCreateMouseEvent,
                CGEventGetLocation,
                CGEventPost,
                kCGEventLeftMouseDown,
                kCGEventLeftMouseDragged,
                kCGEventLeftMouseUp,
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
        self._left_mouse_down = kCGEventLeftMouseDown
        self._left_mouse_up = kCGEventLeftMouseUp
        self._left_mouse_dragged = kCGEventLeftMouseDragged
        self._event_tap = kCGHIDEventTap
        self._left_button = kCGMouseButtonLeft
        self.left_is_down = False

    def _post_at_current_location(self, event_type: int) -> None:
        source_event = self._create_event(None)
        current = self._get_location(source_event)
        location = (float(current.x), float(current.y))
        event = self._create_mouse_event(None, event_type, location, self._left_button)
        self._post(self._event_tap, event)

    def move_relative(self, dx: float, dy: float) -> None:
        source_event = self._create_event(None)
        current = self._get_location(source_event)
        target = (float(current.x + dx), float(current.y + dy))
        event_type = self._left_mouse_dragged if self.left_is_down else self._mouse_moved
        event = self._create_mouse_event(None, event_type, target, self._left_button)
        self._post(self._event_tap, event)

    def left_click(self) -> None:
        self._post_at_current_location(self._left_mouse_down)
        self._post_at_current_location(self._left_mouse_up)

    def left_down(self) -> None:
        if self.left_is_down:
            return
        self._post_at_current_location(self._left_mouse_down)
        self.left_is_down = True

    def left_up(self) -> None:
        if not self.left_is_down:
            return
        self._post_at_current_location(self._left_mouse_up)
        self.left_is_down = False


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
