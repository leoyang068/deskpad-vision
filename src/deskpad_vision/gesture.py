from __future__ import annotations

from dataclasses import dataclass


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
