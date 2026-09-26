from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

INDEX_TIP = 8
INDEX_MCP = 5
THUMB_TIP = 4
PINKY_MCP = 17


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class HandObservation:
    landmarks: List[Landmark]
    handedness: str
    handedness_score: float

    @property
    def index_tip(self) -> Tuple[float, float]:
        point = self.landmarks[INDEX_TIP]
        return point.x, point.y

    @property
    def pinch_ratio(self) -> float:
        thumb = self.landmarks[THUMB_TIP]
        index = self.landmarks[INDEX_TIP]
        index_mcp = self.landmarks[INDEX_MCP]
        pinky_mcp = self.landmarks[PINKY_MCP]
        pinch = float(np.hypot(thumb.x - index.x, thumb.y - index.y))
        palm_width = float(np.hypot(index_mcp.x - pinky_mcp.x, index_mcp.y - pinky_mcp.y))
        return pinch / max(palm_width, 1e-6)


class MediaPipeHandTracker:
    def __init__(self) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is not installed. Run `python -m pip install -e .`."
            ) from exc

        try:
            hands_module = mp.solutions.hands
        except AttributeError:
            from mediapipe.python.solutions import hands as hands_module

        self._hands_module = hands_module
        self._hands = hands_module.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )

    def process(self, frame_bgr: np.ndarray) -> Optional[HandObservation]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None

        height, width = frame_bgr.shape[:2]
        raw_landmarks = result.multi_hand_landmarks[0].landmark
        landmarks = [
            Landmark(float(point.x * width), float(point.y * height), float(point.z))
            for point in raw_landmarks
        ]
        handedness = "unknown"
        handedness_score = 0.0
        if result.multi_handedness:
            classification = result.multi_handedness[0].classification[0]
            handedness = str(classification.label)
            handedness_score = float(classification.score)
        return HandObservation(landmarks, handedness, handedness_score)

    def close(self) -> None:
        self._hands.close()
