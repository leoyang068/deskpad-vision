from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

Rect = Tuple[int, int, int, int]
Point = Tuple[float, float]
Quad = Tuple[Point, Point, Point, Point]


@dataclass
class Profile:
    """Persistent geometry captured from a clear reference frame."""

    image_width: int
    image_height: int
    crop_y: int
    trackpad_rect: Rect
    keyboard_rect: Rect
    zone_margin: int = 18
    trackpad_gap: int = 12
    trackpad_quad: Optional[Quad] = None
    surface_quad: Optional[Quad] = None
    calibration_version: int = 1

    @property
    def reference_name(self) -> str:
        return "reference.png"

    @property
    def effective_trackpad_quad(self) -> Quad:
        if self.trackpad_quad is not None:
            return self.trackpad_quad
        x, y, width, height = self.trackpad_rect
        return (
            (float(x), float(y)),
            (float(x + width), float(y)),
            (float(x + width), float(y + height)),
            (float(x), float(y + height)),
        )

    @property
    def effective_surface_quad(self) -> Quad:
        if self.surface_quad is not None:
            return self.surface_quad
        _, y, _, height = self.trackpad_rect
        left = float(self.zone_margin)
        right = float(self.image_width - self.zone_margin)
        return (
            (left, float(y)),
            (right, float(y)),
            (right, float(y + height)),
            (left, float(y + height)),
        )

    def save(self, profile_dir: Path) -> None:
        profile_dir.mkdir(parents=True, exist_ok=True)
        path = profile_dir / "profile.json"
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, profile_dir: Path) -> "Profile":
        path = profile_dir / "profile.json"
        if not path.exists():
            raise FileNotFoundError(f"Profile not found: {path}. Run `deskpad calibrate` first.")
        raw: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        raw["trackpad_rect"] = tuple(raw["trackpad_rect"])
        raw["keyboard_rect"] = tuple(raw["keyboard_rect"])
        for name in ("trackpad_quad", "surface_quad"):
            if raw.get(name) is not None:
                raw[name] = tuple(tuple(float(value) for value in point) for point in raw[name])
        return cls(**raw)


def default_profile_dir() -> Path:
    return Path.home() / ".deskpad-vision"
