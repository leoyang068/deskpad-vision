from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .hand_tracking import HandObservation, Landmark

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_TIP = 12
PINKY_MCP = 17
WRIST = 0

# The mirror/camera arrangement used by this project reverses the user's physical
# left/right in the captured image. Keep this semantic conversion separate from
# pointer axis inversion: this describes a place, not a movement direction.
IMAGE_TO_PHYSICAL_SIDE = {"left": "right", "right": "left"}


def physical_side_for_image_zone(image_zone: Optional[str]) -> Optional[str]:
    if image_zone is None:
        return None
    return IMAGE_TO_PHYSICAL_SIDE.get(image_zone, image_zone)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _distance_2d(first: Landmark, second: Landmark) -> float:
    return float(math.hypot(first.x - second.x, first.y - second.y))


def _point_3d(point: Landmark, image_width: int) -> np.ndarray:
    # MediaPipe x/y have already been converted to pixels while z remains
    # normalized to image width. Convert z to pixels before measuring angles.
    return np.asarray((point.x, point.y, point.z * image_width), dtype=np.float64)


def _joint_angle(
    first: Landmark,
    vertex: Landmark,
    third: Landmark,
    image_width: int,
) -> float:
    first_vector = _point_3d(first, image_width) - _point_3d(vertex, image_width)
    second_vector = _point_3d(third, image_width) - _point_3d(vertex, image_width)
    denominator = float(np.linalg.norm(first_vector) * np.linalg.norm(second_vector))
    if denominator <= 1e-8:
        return 0.0
    cosine = float(np.dot(first_vector, second_vector) / denominator)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _safe_patch(gray: np.ndarray, center: Tuple[float, float], radius: int) -> np.ndarray:
    height, width = gray.shape[:2]
    x = int(round(center[0]))
    y = int(round(center[1]))
    left = max(0, x - radius)
    right = min(width, x + radius + 1)
    top = max(0, y - radius)
    bottom = min(height, y + radius + 1)
    if right <= left or bottom <= top:
        return np.empty((0, 0), dtype=np.uint8)
    return gray[top:bottom, left:right]


def _patch_features(
    current_gray: np.ndarray,
    reference_gray: np.ndarray,
    current_tip: Tuple[float, float],
    reference_tip: Tuple[float, float],
    radius: int,
) -> Tuple[float, float, float, float, float]:
    current = _safe_patch(current_gray, current_tip, radius)
    reference = _safe_patch(reference_gray, reference_tip, radius)
    if current.size < 16:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    current_float = current.astype(np.float32)
    gradient_x = cv2.Sobel(current_float, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(current_float, cv2.CV_32F, 0, 1, ksize=3)
    gradient = float(np.mean(cv2.magnitude(gradient_x, gradient_y)))

    mid_y = max(1, current.shape[0] // 2)
    mid_x = max(1, current.shape[1] // 2)
    upper = float(np.mean(current_float[:mid_y]))
    lower = float(np.mean(current_float[mid_y:]))
    left = float(np.mean(current_float[:, :mid_x]))
    right = float(np.mean(current_float[:, mid_x:]))

    reference_difference = 0.0
    if reference.size >= 16:
        current_resized = cv2.resize(current, (32, 32), interpolation=cv2.INTER_AREA)
        reference_resized = cv2.resize(reference, (32, 32), interpolation=cv2.INTER_AREA)
        reference_difference = float(
            np.mean(
                np.abs(
                    current_resized.astype(np.float32)
                    - reference_resized.astype(np.float32)
                )
            )
        )
    return (
        float(np.std(current_float)),
        gradient,
        lower - upper,
        right - left,
        reference_difference,
    )


CONTACT_FEATURE_NAMES = (
    "index_extension",
    "pip_angle",
    "dip_angle",
    "tip_dx_mcp",
    "tip_dy_mcp",
    "tip_dz_mcp",
    "tip_dx_dip",
    "tip_dy_dip",
    "tip_dz_dip",
    "tip_distance_middle",
    "tip_dz_middle",
    "wrist_dz_knuckles",
    "pinch_ratio",
    "patch_std",
    "patch_gradient",
    "patch_upper_lower",
    "patch_left_right",
    "reference_patch_difference",
    "plane_speed",
    "image_vertical_speed",
)


@dataclass(frozen=True)
class ContactFeatures:
    timestamp: float
    image_zone: str
    physical_zone: str
    image_tip: Tuple[float, float]
    reference_tip: Tuple[float, float]
    plane_tip: Tuple[float, float]
    handedness: str
    handedness_score: float
    palm_width: float
    index_extension: float
    pip_angle: float
    dip_angle: float
    tip_dx_mcp: float
    tip_dy_mcp: float
    tip_dz_mcp: float
    tip_dx_dip: float
    tip_dy_dip: float
    tip_dz_dip: float
    tip_distance_middle: float
    tip_dz_middle: float
    wrist_dz_knuckles: float
    pinch_ratio: float
    patch_std: float
    patch_gradient: float
    patch_upper_lower: float
    patch_left_right: float
    reference_patch_difference: float
    plane_speed: float
    image_vertical_speed: float
    landmarks: Tuple[Tuple[float, float, float], ...]

    def vector(self) -> np.ndarray:
        return np.asarray(
            [float(getattr(self, name)) for name in CONTACT_FEATURE_NAMES],
            dtype=np.float64,
        )

    def to_sample(self, label: str) -> Dict[str, object]:
        sample: Dict[str, object] = asdict(self)
        sample["label"] = label
        return sample


class ContactFeatureExtractor:
    """Extract pose, appearance, and temporal evidence around the index fingertip."""

    def __init__(self, reference_bgr: np.ndarray) -> None:
        self.reference_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
        self._last_timestamp: Optional[float] = None
        self._last_plane_tip: Optional[np.ndarray] = None
        self._last_image_y: Optional[float] = None

    def reset_motion(self) -> None:
        self._last_timestamp = None
        self._last_plane_tip = None
        self._last_image_y = None

    def extract(
        self,
        timestamp: float,
        frame_bgr: np.ndarray,
        hand: HandObservation,
        reference_tip: Tuple[float, float],
        plane_tip: Tuple[float, float],
        image_zone: str,
    ) -> ContactFeatures:
        landmarks = hand.landmarks
        image_width = frame_bgr.shape[1]
        tip = landmarks[INDEX_TIP]
        dip = landmarks[INDEX_DIP]
        pip = landmarks[INDEX_PIP]
        mcp = landmarks[INDEX_MCP]
        middle_tip = landmarks[MIDDLE_TIP]
        wrist = landmarks[WRIST]
        knuckles = [landmarks[index] for index in (5, 9, 13, 17)]

        palm_width = max(_distance_2d(mcp, landmarks[PINKY_MCP]), 1.0)
        segment_length = (
            _distance_2d(mcp, pip)
            + _distance_2d(pip, dip)
            + _distance_2d(dip, tip)
        )
        index_extension = _distance_2d(mcp, tip) / max(segment_length, 1.0)

        z_scale = image_width / palm_width
        current_plane = np.asarray(plane_tip, dtype=np.float64)
        plane_speed = 0.0
        image_vertical_speed = 0.0
        if self._last_timestamp is not None:
            delta_time = timestamp - self._last_timestamp
            if 1e-3 <= delta_time <= 0.25:
                if self._last_plane_tip is not None:
                    plane_speed = float(
                        np.linalg.norm(current_plane - self._last_plane_tip) / delta_time
                    )
                if self._last_image_y is not None:
                    image_vertical_speed = float(
                        (tip.y - self._last_image_y) / (palm_width * delta_time)
                    )
        self._last_timestamp = timestamp
        self._last_plane_tip = current_plane
        self._last_image_y = tip.y

        current_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        patch_radius = int(np.clip(round(palm_width * 0.20), 8, 32))
        (
            patch_std,
            patch_gradient,
            patch_upper_lower,
            patch_left_right,
            reference_patch_difference,
        ) = _patch_features(
            current_gray,
            self.reference_gray,
            hand.index_tip,
            reference_tip,
            patch_radius,
        )

        knuckle_z = float(np.mean([point.z for point in knuckles]))
        physical_zone = physical_side_for_image_zone(image_zone) or image_zone
        return ContactFeatures(
            timestamp=timestamp,
            image_zone=image_zone,
            physical_zone=physical_zone,
            image_tip=(float(tip.x), float(tip.y)),
            reference_tip=(float(reference_tip[0]), float(reference_tip[1])),
            plane_tip=(float(plane_tip[0]), float(plane_tip[1])),
            handedness=hand.handedness,
            handedness_score=hand.handedness_score,
            palm_width=palm_width,
            index_extension=index_extension,
            pip_angle=_joint_angle(mcp, pip, dip, image_width) / 180.0,
            dip_angle=_joint_angle(pip, dip, tip, image_width) / 180.0,
            tip_dx_mcp=(tip.x - mcp.x) / palm_width,
            tip_dy_mcp=(tip.y - mcp.y) / palm_width,
            tip_dz_mcp=(tip.z - mcp.z) * z_scale,
            tip_dx_dip=(tip.x - dip.x) / palm_width,
            tip_dy_dip=(tip.y - dip.y) / palm_width,
            tip_dz_dip=(tip.z - dip.z) * z_scale,
            tip_distance_middle=_distance_2d(tip, middle_tip) / palm_width,
            tip_dz_middle=(tip.z - middle_tip.z) * z_scale,
            wrist_dz_knuckles=(wrist.z - knuckle_z) * z_scale,
            pinch_ratio=float(np.clip(hand.pinch_ratio, 0.0, 4.0)),
            patch_std=patch_std / 64.0,
            patch_gradient=patch_gradient / 256.0,
            patch_upper_lower=patch_upper_lower / 128.0,
            patch_left_right=patch_left_right / 128.0,
            reference_patch_difference=reference_patch_difference / 255.0,
            plane_speed=float(np.clip(plane_speed / 2500.0, 0.0, 4.0)),
            image_vertical_speed=float(np.clip(image_vertical_speed / 15.0, -4.0, 4.0)),
            landmarks=tuple((point.x, point.y, point.z) for point in landmarks),
        )


class ContactState(str, Enum):
    AIR = "AIR"
    TOUCH = "TOUCH"
    RELEASE = "RELEASE"


@dataclass(frozen=True)
class ContactCollectionStage:
    physical_side: str
    label: str


@dataclass(frozen=True)
class ContactCollectionStatus:
    stage_number: int
    stage_count: int
    physical_side: Optional[str]
    label: Optional[str]
    mode: str
    remaining_seconds: float


DEFAULT_CONTACT_COLLECTION_STAGES = tuple(
    ContactCollectionStage(physical_side, label)
    for physical_side in ("right", "left")
    for label in ("air", "touch", "slide", "lift")
)


class GuidedContactProtocol:
    """Timed two-side collection whose recording clock pauses on invalid frames."""

    def __init__(
        self,
        prepare_seconds: float = 3.0,
        record_seconds: float = 5.0,
        stages: Sequence[ContactCollectionStage] = DEFAULT_CONTACT_COLLECTION_STAGES,
    ) -> None:
        if prepare_seconds < 0 or record_seconds <= 0:
            raise ValueError("Contact collection durations must be positive")
        if not stages:
            raise ValueError("Contact collection requires at least one stage")
        self.prepare_seconds = float(prepare_seconds)
        self.record_seconds = float(record_seconds)
        self.stages = tuple(stages)
        self.stage_index = 0
        self.mode = "prepare"
        self._phase_started: Optional[float] = None
        self._last_timestamp: Optional[float] = None
        self._recorded_seconds = 0.0

    @property
    def done(self) -> bool:
        return self.stage_index >= len(self.stages)

    def _ensure_started(self, timestamp: float) -> None:
        if self._phase_started is None:
            self._phase_started = timestamp
        if self._last_timestamp is None:
            self._last_timestamp = timestamp

    def status(self, timestamp: float) -> ContactCollectionStatus:
        self._ensure_started(timestamp)
        if self.done:
            return ContactCollectionStatus(
                stage_number=len(self.stages),
                stage_count=len(self.stages),
                physical_side=None,
                label=None,
                mode="done",
                remaining_seconds=0.0,
            )
        stage = self.stages[self.stage_index]
        assert self._phase_started is not None
        if self.mode == "prepare":
            remaining = max(0.0, self.prepare_seconds - (timestamp - self._phase_started))
        else:
            remaining = max(0.0, self.record_seconds - self._recorded_seconds)
        return ContactCollectionStatus(
            stage_number=self.stage_index + 1,
            stage_count=len(self.stages),
            physical_side=stage.physical_side,
            label=stage.label,
            mode=self.mode,
            remaining_seconds=remaining,
        )

    def update(self, timestamp: float, valid_sample: bool) -> ContactCollectionStatus:
        self._ensure_started(timestamp)
        assert self._phase_started is not None
        assert self._last_timestamp is not None
        delta_time = float(np.clip(timestamp - self._last_timestamp, 0.0, 0.10))
        if not self.done and self.mode == "prepare":
            if timestamp - self._phase_started >= self.prepare_seconds:
                self.mode = "record"
                self._recorded_seconds = 0.0
        elif not self.done and self.mode == "record":
            if valid_sample:
                self._recorded_seconds += delta_time
            if self._recorded_seconds + 1e-9 >= self.record_seconds:
                self.stage_index += 1
                self.mode = "prepare"
                self._phase_started = timestamp
                self._recorded_seconds = 0.0
        self._last_timestamp = timestamp
        return self.status(timestamp)


@dataclass
class ContactZoneGrace:
    """Keep the original side briefly while a touching fingertip lifts out of its 2D polygon."""

    max_seconds: float = 0.45
    max_plane_distance: float = 520.0
    _image_zone: Optional[str] = None
    _last_inside_at: Optional[float] = None
    _anchor_plane: Optional[np.ndarray] = None

    def update(
        self,
        timestamp: float,
        detected_image_zone: Optional[str],
        plane_tip: Optional[Tuple[float, float]],
        contact_active: bool,
    ) -> Optional[str]:
        if detected_image_zone is not None and plane_tip is not None:
            self._image_zone = detected_image_zone
            self._last_inside_at = timestamp
            self._anchor_plane = np.asarray(plane_tip, dtype=np.float64)
            return detected_image_zone
        if (
            not contact_active
            or self._image_zone is None
            or self._last_inside_at is None
            or self._anchor_plane is None
            or plane_tip is None
        ):
            return None
        elapsed = timestamp - self._last_inside_at
        distance = float(
            np.linalg.norm(np.asarray(plane_tip, dtype=np.float64) - self._anchor_plane)
        )
        if elapsed <= self.max_seconds and distance <= self.max_plane_distance:
            return self._image_zone
        self.reset()
        return None

    def reset(self) -> None:
        self._image_zone = None
        self._last_inside_at = None
        self._anchor_plane = None


@dataclass
class ContactLatch:
    """Conservative temporal contact latch with asymmetric hysteresis."""

    on_threshold: float = 0.75
    off_threshold: float = 0.45
    touch_frames: int = 4
    release_frames: int = 3
    release_display_frames: int = 3
    active: bool = False
    _candidate_frames: int = 0
    _release_display_remaining: int = 0

    def update(self, probability: float, valid: bool = True) -> ContactState:
        probability = float(np.clip(probability, 0.0, 1.0))
        if not valid:
            self.reset()
            return ContactState.AIR

        if not self.active:
            if probability >= self.on_threshold:
                self._candidate_frames += 1
                if self._candidate_frames >= self.touch_frames:
                    self.active = True
                    self._candidate_frames = 0
                    self._release_display_remaining = 0
                    return ContactState.TOUCH
            else:
                self._candidate_frames = 0
            if self._release_display_remaining > 0:
                self._release_display_remaining -= 1
                return ContactState.RELEASE
            return ContactState.AIR

        if probability <= self.off_threshold:
            self._candidate_frames += 1
            if self._candidate_frames >= self.release_frames:
                self.active = False
                self._candidate_frames = 0
                self._release_display_remaining = max(0, self.release_display_frames - 1)
                return ContactState.RELEASE
        else:
            self._candidate_frames = 0
        return ContactState.TOUCH

    def reset(self) -> None:
        self.active = False
        self._candidate_frames = 0
        self._release_display_remaining = 0


@dataclass(frozen=True)
class ContactTrainingReport:
    sample_count: int
    positive_count: int
    negative_count: int
    validation_count: int
    threshold: float
    false_positive_rate: float
    recall: float
    accuracy: float


@dataclass(frozen=True)
class ZonedContactTrainingReport:
    zone_reports: Dict[str, ContactTrainingReport]
    sample_count: int
    positive_count: int
    negative_count: int
    validation_count: int
    false_positive_rate: float
    recall: float
    accuracy: float


@dataclass(frozen=True)
class LogisticContactModel:
    feature_names: Tuple[str, ...]
    mean: Tuple[float, ...]
    scale: Tuple[float, ...]
    weights: Tuple[float, ...]
    bias: float
    threshold: float
    version: int = 1
    reference_sha256: Optional[str] = None
    physical_hand: str = "right"

    def probability(self, features: ContactFeatures) -> float:
        vector = features.vector()
        if tuple(self.feature_names) != CONTACT_FEATURE_NAMES:
            raise ValueError("Contact model feature schema does not match this app version")
        normalized = (vector - np.asarray(self.mean)) / np.asarray(self.scale)
        score = float(np.dot(normalized, np.asarray(self.weights)) + self.bias)
        score = float(np.clip(score, -40.0, 40.0))
        return 1.0 / (1.0 + math.exp(-score))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, value: Dict[str, object]) -> "LogisticContactModel":
        raw = dict(value)
        for name in ("feature_names", "mean", "scale", "weights"):
            raw[name] = tuple(raw[name])
        model = cls(**raw)
        expected = len(CONTACT_FEATURE_NAMES)
        if not all(len(getattr(model, name)) == expected for name in ("mean", "scale", "weights")):
            raise ValueError("Contact model has an invalid feature vector length")
        return model

    @classmethod
    def load(cls, path: Path) -> "LogisticContactModel":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(raw)


@dataclass(frozen=True)
class ZonedContactModel:
    zone_models: Dict[str, LogisticContactModel]
    reference_sha256: str
    physical_hand: str = "right"
    version: int = 2

    def model_for(self, physical_zone: Optional[str]) -> Optional[LogisticContactModel]:
        if physical_zone is None:
            return None
        return self.zone_models.get(physical_zone)

    def probability(self, features: ContactFeatures) -> float:
        model = self.model_for(features.physical_zone)
        if model is None:
            raise ValueError(f"No contact model for physical zone {features.physical_zone!r}")
        return model.probability(features)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ZonedContactModel":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if "zone_models" not in raw:
            # Backward compatibility with the short-lived single-model preview.
            legacy = LogisticContactModel.from_dict(raw)
            if legacy.reference_sha256 is None:
                raise ValueError("Legacy contact model has no calibration fingerprint")
            return cls(
                zone_models={"left": legacy, "right": legacy},
                reference_sha256=legacy.reference_sha256,
                physical_hand=legacy.physical_hand,
            )
        zone_models = {
            str(name): LogisticContactModel.from_dict(value)
            for name, value in raw["zone_models"].items()
        }
        if set(zone_models) != {"left", "right"}:
            raise ValueError("Contact model must contain physical left and right zone models")
        return cls(
            zone_models=zone_models,
            reference_sha256=str(raw["reference_sha256"]),
            physical_hand=str(raw.get("physical_hand", "right")),
            version=int(raw.get("version", 2)),
        )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _select_threshold(probabilities: np.ndarray, labels: np.ndarray) -> float:
    best_threshold = 0.80
    best_key = (-1.0, -1.0)
    negatives = max(1, int(np.sum(labels == 0)))
    positives = max(1, int(np.sum(labels == 1)))
    for threshold in np.linspace(0.50, 0.98, 49):
        predicted = probabilities >= threshold
        false_positive_rate = float(np.sum(predicted & (labels == 0)) / negatives)
        recall = float(np.sum(predicted & (labels == 1)) / positives)
        # False contact is more harmful than a missed contact. Prefer thresholds
        # below 2% validation FPR, then maximize recall and margin from that limit.
        if false_positive_rate <= 0.02:
            key = (recall, threshold)
            if key > best_key:
                best_key = key
                best_threshold = float(threshold)
    if best_key[0] < 0:
        negative_probabilities = probabilities[labels == 0]
        if negative_probabilities.size:
            best_threshold = float(np.clip(np.max(negative_probabilities) + 0.03, 0.5, 0.99))
    return best_threshold


def train_contact_model(
    vectors: np.ndarray,
    labels: np.ndarray,
    seed: int = 19,
) -> Tuple[LogisticContactModel, ContactTrainingReport]:
    vectors = np.asarray(vectors, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != len(CONTACT_FEATURE_NAMES):
        raise ValueError("Contact samples have an incompatible feature shape")
    if len(vectors) < 60 or np.sum(labels == 1) < 25 or np.sum(labels == 0) < 25:
        raise ValueError("Need at least 25 AIR and 25 TOUCH/SLIDE samples to fit contact")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("Contact samples contain non-finite feature values")

    rng = np.random.default_rng(seed)
    validation_indices: List[int] = []
    training_indices: List[int] = []
    for label in (0.0, 1.0):
        indices = np.flatnonzero(labels == label)
        # Camera frames next to one another are almost duplicates. Split whole
        # temporal blocks rather than scattering individual frames across train
        # and validation, which would report unrealistically good accuracy.
        blocks = [indices[start : start + 12] for start in range(0, len(indices), 12)]
        rng.shuffle(blocks)
        validation_target = max(8, int(round(len(indices) * 0.20)))
        validation_blocks: List[np.ndarray] = []
        training_blocks: List[np.ndarray] = []
        validation_size = 0
        for block_index, block in enumerate(blocks):
            can_hold_out = block_index < len(blocks) - 1
            if can_hold_out and validation_size < validation_target:
                validation_blocks.append(block)
                validation_size += len(block)
            else:
                training_blocks.append(block)
        validation_indices.extend(
            index for block in validation_blocks for index in block.tolist()
        )
        training_indices.extend(index for block in training_blocks for index in block.tolist())
    train = np.asarray(training_indices, dtype=np.int64)
    validation = np.asarray(validation_indices, dtype=np.int64)

    mean = np.mean(vectors[train], axis=0)
    scale = np.std(vectors[train], axis=0)
    scale = np.where(scale < 1e-5, 1.0, scale)
    normalized = (vectors - mean) / scale

    train_x = normalized[train]
    train_y = labels[train]
    weights = np.zeros(train_x.shape[1], dtype=np.float64)
    bias = 0.0
    positive_weight = len(train_y) / max(1.0, 2.0 * float(np.sum(train_y == 1)))
    negative_weight = len(train_y) / max(1.0, 2.0 * float(np.sum(train_y == 0)))
    sample_weights = np.where(train_y == 1, positive_weight, negative_weight)
    learning_rate = 0.055
    regularization = 0.025
    for step in range(1400):
        probabilities = _sigmoid(train_x @ weights + bias)
        error = (probabilities - train_y) * sample_weights
        gradient = train_x.T @ error / len(train_x) + regularization * weights
        bias_gradient = float(np.mean(error))
        decay = 1.0 / (1.0 + step / 900.0)
        weights -= learning_rate * decay * gradient
        bias -= learning_rate * decay * bias_gradient

    validation_probabilities = _sigmoid(normalized[validation] @ weights + bias)
    validation_labels = labels[validation]
    threshold = _select_threshold(validation_probabilities, validation_labels)
    predicted = validation_probabilities >= threshold
    negatives = max(1, int(np.sum(validation_labels == 0)))
    positives = max(1, int(np.sum(validation_labels == 1)))
    false_positive_rate = float(
        np.sum(predicted & (validation_labels == 0)) / negatives
    )
    recall = float(np.sum(predicted & (validation_labels == 1)) / positives)
    accuracy = float(np.mean(predicted == validation_labels))

    model = LogisticContactModel(
        feature_names=CONTACT_FEATURE_NAMES,
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        weights=tuple(float(value) for value in weights),
        bias=bias,
        threshold=threshold,
    )
    report = ContactTrainingReport(
        sample_count=len(vectors),
        positive_count=int(np.sum(labels == 1)),
        negative_count=int(np.sum(labels == 0)),
        validation_count=len(validation),
        threshold=threshold,
        false_positive_rate=false_positive_rate,
        recall=recall,
        accuracy=accuracy,
    )
    return model, report


def samples_to_training_arrays(
    samples: Iterable[Dict[str, object]],
) -> Tuple[np.ndarray, np.ndarray]:
    vectors: List[List[float]] = []
    labels: List[float] = []
    positive_labels = {"touch", "slide"}
    negative_labels = {"air", "lift"}
    for sample in samples:
        label = str(sample.get("label", "")).lower()
        if label not in positive_labels | negative_labels:
            continue
        try:
            vector = [float(sample[name]) for name in CONTACT_FEATURE_NAMES]
        except (KeyError, TypeError, ValueError):
            continue
        vectors.append(vector)
        labels.append(1.0 if label in positive_labels else 0.0)
    if not vectors:
        return (
            np.empty((0, len(CONTACT_FEATURE_NAMES)), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
    return np.asarray(vectors, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def load_jsonl_samples(paths: Sequence[Path]) -> List[Dict[str, object]]:
    samples: List[Dict[str, object]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if isinstance(value, dict):
                    samples.append(value)
    return samples
