from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

Rect = Tuple[int, int, int, int]


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

    @property
    def reference_name(self) -> str:
        return "reference.png"

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
        return cls(**raw)


def default_profile_dir() -> Path:
    return Path.home() / ".deskpad-vision"
