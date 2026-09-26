from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from .pointer import PointerSink


@dataclass
class PinchLatch:
    on_threshold: float = 0.34
    off_threshold: float = 0.48
    confirmation_frames: int = 2
    active: bool = False
    _candidate_frames: int = 0

    def update(self, ratio: float) -> bool:
        target = ratio < self.on_threshold if not self.active else ratio < self.off_threshold
        if target == self.active:
            self._candidate_frames = 0
            return self.active

        self._candidate_frames += 1
        if self._candidate_frames >= self.confirmation_frames:
            self.active = target
            self._candidate_frames = 0
        return self.active

    def reset(self) -> None:
        self.active = False
        self._candidate_frames = 0


@dataclass
class TouchMouseGesture:
    """Interpret confirmed plane contact as tap or hold-to-drag gestures."""

    tap_max_seconds: float = 0.42
    tap_max_travel: float = 55.0
    drag_hold_seconds: float = 0.62
    drag_hold_max_travel: float = 32.0
    _active: bool = False
    _dragging: bool = False
    _drag_was_emitted: bool = False
    _started_at: float = 0.0
    _last_point: Optional[Tuple[float, float]] = None
    _zone: Optional[str] = None
    _travel: float = 0.0

    def update(
        self,
        timestamp: float,
        point: Optional[Tuple[float, float]],
        zone: Optional[str],
        touching: bool,
        released: bool,
        valid: bool,
        sink: PointerSink,
        emit_click: bool = False,
        emit_drag: bool = False,
    ) -> Optional[str]:
        if not valid or point is None or zone is None:
            return self.cancel(sink)

        if touching:
            if not self._active:
                self._active = True
                self._started_at = timestamp
                self._last_point = point
                self._zone = zone
                self._travel = 0.0
                self._dragging = False
                self._drag_was_emitted = False
                return "TOUCH START"
            if zone != self._zone:
                self.cancel(sink)
                return None
            if self._last_point is not None:
                self._travel += math.hypot(
                    point[0] - self._last_point[0],
                    point[1] - self._last_point[1],
                )
            self._last_point = point
            held_for = timestamp - self._started_at
            if (
                not self._dragging
                and held_for >= self.drag_hold_seconds
                and self._travel <= self.drag_hold_max_travel
            ):
                self._dragging = True
                if emit_drag:
                    sink.left_down()
                    self._drag_was_emitted = True
                return "DRAG START"
            return None

        if not self._active:
            return None
        duration = timestamp - self._started_at
        if self._dragging:
            if self._drag_was_emitted:
                sink.left_up()
            self._reset()
            return "DRAG END" if released else "DRAG CANCEL"

        is_tap = (
            released
            and duration <= self.tap_max_seconds
            and self._travel <= self.tap_max_travel
        )
        self._reset()
        if is_tap:
            if emit_click:
                sink.left_click()
            return "TAP"
        return None

    def cancel(self, sink: PointerSink) -> Optional[str]:
        action = None
        if self._dragging:
            if self._drag_was_emitted:
                sink.left_up()
            action = "DRAG CANCEL"
        self._reset()
        return action

    def _reset(self) -> None:
        self._active = False
        self._dragging = False
        self._drag_was_emitted = False
        self._started_at = 0.0
        self._last_point = None
        self._zone = None
        self._travel = 0.0
