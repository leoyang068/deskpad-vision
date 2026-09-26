from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, TextIO, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .alignment import AlignmentResult, KeyboardAligner, LockedAlignmentGuard
from .app import (
    WINDOW_NAME,
    _detect_boundary_automatically,
    _draw_crop,
    _draw_hand,
    _draw_surface,
    _draw_text,
    _fit_frame,
    _open_camera,
    _read_frame,
)
from .config import Profile
from .contact import (
    DEFAULT_CONTACT_COLLECTION_STAGES,
    ContactCollectionStatus,
    ContactFeatureExtractor,
    GuidedContactProtocol,
    ZonedContactModel,
    ZonedContactTrainingReport,
    physical_side_for_image_zone,
    samples_to_training_arrays,
    sha256_file,
    train_contact_model,
)
from .geometry import apply_homography
from .hand_tracking import MediaPipeHandTracker
from .plane import PlanarSurface

LABEL_KEYS = {
    ord("1"): "air",
    ord("2"): "touch",
    ord("3"): "slide",
    ord("4"): "lift",
}
MIN_SAMPLES_PER_CLASS = 25
SUPPORTED_PHYSICAL_HAND = "right"
CONTACT_LABELS = ("air", "touch", "slide", "lift")
ACTION_TITLES_ZH = {
    "air": "悬空移动｜绝对不要碰表面",
    "touch": "接触静止｜碰到后不要移动",
    "slide": "贴面滑动｜始终保持接触",
    "lift": "接触后抬起｜抬起后不要再碰",
}
ACTION_COLORS = {
    "air": (255, 220, 70),
    "touch": (80, 255, 120),
    "slide": (255, 150, 70),
    "lift": (220, 100, 255),
}
LIFT_ZONE_GRACE_DISTANCE = 520.0
CHINESE_FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
)


def _session_paths(profile_dir: Path) -> Tuple[Path, Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir = profile_dir / "contact-data" / timestamp
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir, session_dir / "metadata.json", session_dir / "samples.jsonl"


def _write_metadata(
    path: Path,
    reference_sha256: str,
    profile: Profile,
    counts: Counter[str],
    complete: bool,
    replaces: Optional[Sequence[Tuple[str, str]]] = None,
    requested_label: Optional[str] = None,
) -> None:
    payload = {
        "version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "reference_sha256": reference_sha256,
        "calibration_version": profile.calibration_version,
        "physical_hand": SUPPORTED_PHYSICAL_HAND,
        "requested_label": requested_label,
        "image_to_physical_side": {"left": "right", "right": "left"},
        "labels": {
            "air": "finger hovering above the plane",
            "touch": "stationary fingertip touching the plane",
            "slide": "fingertip moving while remaining in contact",
            "lift": "finger just released and above the plane",
        },
        "counts": dict(counts),
        "complete": complete,
        "replaces": [
            {"physical_zone": physical_zone, "label": label}
            for physical_zone, label in (replaces or ())
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _matching_sample_paths(profile_dir: Path, reference_sha256: str) -> List[Path]:
    paths: List[Path] = []
    data_dir = profile_dir / "contact-data"
    if not data_dir.exists():
        return paths
    for metadata_path in sorted(data_dir.glob("*/metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("reference_sha256") != reference_sha256:
            continue
        # An interrupted or user-rejected recording must never affect a model.
        if metadata.get("complete") is not True:
            continue
        samples_path = metadata_path.with_name("samples.jsonl")
        if samples_path.exists():
            paths.append(samples_path)
    return paths


def archive_contact_data(profile_dir: Path, label: Optional[str] = None) -> Optional[Path]:
    """Remove current contact data from use by moving it to a dated archive."""

    if label is not None and label not in CONTACT_LABELS:
        raise ValueError(f"label must be one of: {', '.join(CONTACT_LABELS)}")
    if label is not None:
        data_dir = profile_dir / "contact-data"
        session_dirs: List[Path] = []
        if data_dir.exists():
            for metadata_path in sorted(data_dir.glob("*/metadata.json")):
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                requested = metadata.get("requested_label")
                counts = metadata.get("counts", {})
                if requested == label or (
                    requested is None
                    and isinstance(counts, dict)
                    and set(counts) == {label}
                ):
                    session_dirs.append(metadata_path.parent)
        targets = session_dirs + [
            path
            for path in (
                profile_dir / "contact-model.json",
                profile_dir / "contact-training-report.json",
            )
            if path.exists()
        ]
    else:
        targets = [
            profile_dir / "contact-data",
            profile_dir / "contact-model.json",
            profile_dir / "contact-training-report.json",
        ]
    existing = [path for path in targets if path.exists()]
    if not existing:
        return None
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive_dir = profile_dir / "contact-archive" / timestamp
    suffix = 1
    while archive_dir.exists():
        archive_dir = profile_dir / "contact-archive" / f"{timestamp}-{suffix}"
        suffix += 1
    archive_dir.mkdir(parents=True)
    moved = []
    for path in existing:
        if label is not None and path.parent.name == "contact-data":
            destination = archive_dir / "contact-data" / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
        else:
            destination = archive_dir / path.name
        shutil.move(str(path), str(destination))
        moved.append(path.name)
    current_data_dir = profile_dir / "contact-data"
    if current_data_dir.exists() and not any(current_data_dir.iterdir()):
        current_data_dir.rmdir()
    (archive_dir / "archive.json").write_text(
        json.dumps(
            {
                "archived_at": datetime.now().astimezone().isoformat(),
                "reason": "contact data reset" if label is None else f"reset label {label}",
                "label": label,
                "moved": moved,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return archive_dir


def _load_training_samples(paths: Sequence[Path]) -> List[dict[str, object]]:
    """Load completed sessions, applying explicit correction-session replacements."""

    samples: List[dict[str, object]] = []
    for path in sorted(paths):
        metadata_path = path.with_name("metadata.json")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        replacement_keys = {
            (str(value.get("physical_zone")), str(value.get("label")))
            for value in metadata.get("replaces", [])
            if isinstance(value, dict)
        }
        if replacement_keys:
            samples = [
                sample
                for sample in samples
                if (str(sample.get("physical_zone")), str(sample.get("label")))
                not in replacement_keys
            ]
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    samples.append(value)
    return samples


def _profile_training_samples(profile_dir: Path, reference_sha256: str) -> List[dict[str, object]]:
    sample_paths = _matching_sample_paths(profile_dir, reference_sha256)
    samples = _load_training_samples(sample_paths)
    return [
        sample
        for sample in samples
        # Sessions created before hand scoping were confirmed to be right-hand-only.
        if str(sample.get("physical_hand", SUPPORTED_PHYSICAL_HAND))
        == SUPPORTED_PHYSICAL_HAND
    ]


def _missing_training_groups(samples: Sequence[dict[str, object]]) -> List[Tuple[str, str]]:
    counts = Counter(
        (str(sample.get("physical_zone")), str(sample.get("label")))
        for sample in samples
    )
    return [
        (physical_zone, label)
        for physical_zone in ("right", "left")
        for label in CONTACT_LABELS
        if counts[(physical_zone, label)] < MIN_SAMPLES_PER_CLASS
    ]


def fit_contact_model_for_profile(profile_dir: Path) -> ZonedContactTrainingReport:
    reference_path = profile_dir / "reference.png"
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference image not found in {profile_dir}")
    reference_sha256 = sha256_file(reference_path)
    samples = _profile_training_samples(profile_dir, reference_sha256)
    if not samples:
        raise RuntimeError("No contact samples match the current calibration")
    missing = _missing_training_groups(samples)
    if missing:
        names = ", ".join(f"physical-{side}/{label}" for side, label in missing)
        raise RuntimeError(f"Contact data is incomplete; collect: {names}")
    zone_models = {}
    zone_reports = {}
    for physical_zone in ("left", "right"):
        zone_samples = [
            sample for sample in samples if sample.get("physical_zone") == physical_zone
        ]
        vectors, labels = samples_to_training_arrays(zone_samples)
        zone_model, zone_report = train_contact_model(vectors, labels)
        zone_model = replace(
            zone_model,
            reference_sha256=reference_sha256,
            physical_hand=SUPPORTED_PHYSICAL_HAND,
        )
        if zone_report.false_positive_rate > 0.08 or zone_report.recall < 0.45:
            raise RuntimeError(
                f"Physical-{physical_zone} contact data is not separable enough: "
                f"false-positive rate {zone_report.false_positive_rate:.1%}, "
                f"recall {zone_report.recall:.1%}. Record more varied samples for that side."
            )
        zone_models[physical_zone] = zone_model
        zone_reports[physical_zone] = zone_report

    model = ZonedContactModel(
        zone_models=zone_models,
        reference_sha256=reference_sha256,
        physical_hand=SUPPORTED_PHYSICAL_HAND,
    )
    report = ZonedContactTrainingReport(
        zone_reports=zone_reports,
        sample_count=sum(value.sample_count for value in zone_reports.values()),
        positive_count=sum(value.positive_count for value in zone_reports.values()),
        negative_count=sum(value.negative_count for value in zone_reports.values()),
        validation_count=sum(value.validation_count for value in zone_reports.values()),
        false_positive_rate=max(
            value.false_positive_rate for value in zone_reports.values()
        ),
        recall=min(value.recall for value in zone_reports.values()),
        accuracy=min(value.accuracy for value in zone_reports.values()),
    )
    model.save(profile_dir / "contact-model.json")
    report_path = profile_dir / "contact-training-report.json"
    report_path.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    return report


def print_contact_training_report(report: ZonedContactTrainingReport) -> None:
    print("Contact model saved with separate physical-side models:")
    for physical_zone in ("right", "left"):
        zone_report = report.zone_reports[physical_zone]
        print(
            f"  physical-{physical_zone}: "
            f"false-positive {zone_report.false_positive_rate:.1%}, "
            f"recall {zone_report.recall:.1%}, accuracy {zone_report.accuracy:.1%}"
        )


def _draw_capture_instructions(
    display: np.ndarray,
    label: Optional[str],
    counts: Counter[str],
    image_zone: Optional[str],
    geometry_trusted: bool,
    settling: bool,
) -> None:
    physical_zone = physical_side_for_image_zone(image_zone)
    zone_text = "-"
    if image_zone is not None:
        zone_text = f"image-{image_zone} / physical-{physical_zone}"
    if label is None:
        label_text = "PAUSED"
        color = (80, 220, 255)
    elif settling:
        label_text = f"{label.upper()} (settling)"
        color = (80, 220, 255)
    else:
        label_text = f"RECORDING {label.upper()}"
        color = (80, 255, 120) if geometry_trusted else (40, 180, 255)
    _draw_text(display, f"CONTACT DATA: {label_text}", 0, color)
    _draw_text(display, f"zone: {zone_text}", 1)
    _draw_text(display, "1 AIR  2 TOUCH  3 SLIDE  4 LIFT  0 PAUSE  q FINISH", 2)
    _draw_text(
        display,
        "samples  air:{air} touch:{touch} slide:{slide} lift:{lift}".format(
            air=counts["air"],
            touch=counts["touch"],
            slide=counts["slide"],
            lift=counts["lift"],
        ),
        3,
    )
    if not geometry_trusted:
        _draw_text(
            display,
            "geometry is not locked - samples are not being saved",
            4,
            (40, 180, 255),
        )
    elif image_zone is None:
        _draw_text(
            display,
            "put the index fingertip inside a colored side plane",
            4,
            (40, 180, 255),
        )


@lru_cache(maxsize=4)
def _chinese_font(size: int) -> ImageFont.FreeTypeFont:
    for path in CHINESE_FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    raise RuntimeError("A Chinese system font was not found")


def _draw_unicode_lines(
    display: np.ndarray,
    lines: Sequence[Tuple[str, Tuple[int, int, int]]],
) -> None:
    height, width = display.shape[:2]
    panel_height = min(height, 16 + len(lines) * max(28, width // 58))
    dark = display.copy()
    cv2.rectangle(dark, (0, 0), (width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(dark, 0.72, display, 0.28, 0, display)

    rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font_size = max(20, min(32, int(round(width / 64))))
    font = _chinese_font(font_size)
    line_height = max(28, int(round(font_size * 1.35)))
    for row, (text, bgr_color) in enumerate(lines):
        rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
        draw.text(
            (13, 8 + row * line_height),
            text,
            font=font,
            fill=rgb_color,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    display[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _guided_action_lines(instruction: ContactCollectionStatus) -> Tuple[str, str, str]:
    side_zh = "右侧" if instruction.physical_side == "right" else "左侧"
    image_zh = "左侧" if instruction.physical_side == "right" else "右侧"
    location = f"电脑物理{side_zh}（画面{image_zh}彩色区域），始终使用右手食指"
    if instruction.label == "air":
        return (
            f"位置：{location}",
            "动作：食指尖悬空约 1–2 厘米，绝对不要碰到电脑表面",
            "采集时：缓慢左右来回移动食指；拇指不要和食指捏合",
        )
    if instruction.label == "touch":
        return (
            f"位置：{location}",
            "动作：食指尖轻轻接触电脑表面，并保持基本不动",
            "保持正常轻触力度；不要滑动，也不要与拇指捏合",
        )
    if instruction.label == "slide":
        return (
            f"位置：{location}，先让食指尖接触表面",
            "动作：保持指尖接触，缓慢左右来回滑动",
            "整个采集期间不要抬起指尖，也不要与拇指捏合",
        )
    return (
        f"位置：{location}；准备倒计时内先让指尖接触表面",
        "动作：一出现“正在采集”，立刻把指尖抬高约 1 厘米",
        "抬起后保持悬空，不要再次接触，也不要与拇指捏合",
    )


def _guided_status_warning(
    instruction: ContactCollectionStatus,
    image_zone: Optional[str],
    geometry_trusted: bool,
    valid_sample: bool,
    lift_off_zone_tracking: bool = False,
) -> str:
    current_physical = physical_side_for_image_zone(image_zone)
    side_zh = "右侧" if instruction.physical_side == "right" else "左侧"
    image_zh = "左侧" if instruction.physical_side == "right" else "右侧"
    if not geometry_trusted:
        return "计时暂停：电脑平面定位尚未锁定，请保持电脑和镜子不动"
    if lift_off_zone_tracking and valid_sample:
        return "LIFT 正常：红点已离开彩色框，仍按抬起前的原区域继续采集"
    if image_zone is None:
        return f"计时暂停：请把食指尖移入画面{image_zh}的彩色区域"
    if current_physical != instruction.physical_side:
        return f"计时暂停：当前在错误一侧，请移到电脑物理{side_zh}"
    if instruction.mode == "prepare":
        return "正在准备：请先摆好上述姿势，倒计时结束后才保存数据"
    if valid_sample:
        return "正在保存有效数据：请持续保持上述动作"
    return "计时暂停：没有获得有效的食指画面"


def _effective_lift_image_zone(
    instruction: Optional[ContactCollectionStatus],
    detected_image_zone: Optional[str],
    plane_tip: Optional[Tuple[float, float]],
    lift_anchor_plane: Optional[np.ndarray],
) -> Tuple[Optional[str], bool]:
    if detected_image_zone is not None:
        return detected_image_zone, False
    if (
        instruction is None
        or instruction.label != "lift"
        or instruction.mode != "record"
        or plane_tip is None
        or lift_anchor_plane is None
    ):
        return None, False
    distance_from_anchor = float(
        np.linalg.norm(np.asarray(plane_tip, dtype=np.float64) - lift_anchor_plane)
    )
    if distance_from_anchor > LIFT_ZONE_GRACE_DISTANCE:
        return None, False
    expected_image_zone = "left" if instruction.physical_side == "right" else "right"
    return expected_image_zone, True


def _announce_guided_instruction(instruction: ContactCollectionStatus) -> None:
    if instruction.mode == "done":
        print("\n接触数据采集完成，正在训练模型…")
        return
    phase = "准备" if instruction.mode == "prepare" else "开始采集"
    label = str(instruction.label)
    print(
        f"\n第 {instruction.stage_number}/{instruction.stage_count} 步 - "
        f"{phase} {label.upper()}"
    )
    print(f"  本次唯一动作：{ACTION_TITLES_ZH[label]}")
    for line in _guided_action_lines(instruction):
        print(f"  {line}")


def _draw_guided_capture_instructions(
    display: np.ndarray,
    instruction: ContactCollectionStatus,
    counts: Counter[str],
    image_zone: Optional[str],
    geometry_trusted: bool,
    valid_sample: bool,
    lift_off_zone_tracking: bool = False,
) -> None:
    if instruction.mode == "done":
        _draw_unicode_lines(display, (("接触数据采集完成，正在训练模型…", (80, 255, 120)),))
        return

    phase = "准备" if instruction.mode == "prepare" else "正在采集"
    label = instruction.label or "air"
    action_color = ACTION_COLORS[label]
    color = (80, 220, 255) if instruction.mode == "prepare" else action_color
    if instruction.mode == "record" and not valid_sample:
        color = (40, 180, 255)
    action_lines = _guided_action_lines(instruction)
    warning = _guided_status_warning(
        instruction,
        image_zone,
        geometry_trusted,
        valid_sample,
        lift_off_zone_tracking,
    )
    label_upper = label.upper()
    count_text = f"本窗口只会写入 {label_upper}｜当前累计 {counts[label]} 帧"
    _draw_unicode_lines(
        display,
        (
            (f"唯一训练动作：{label_upper}｜{ACTION_TITLES_ZH[label]}", action_color),
            (
                f"第 {instruction.stage_number}/{instruction.stage_count} 步｜{phase}｜"
                f"剩余 {instruction.remaining_seconds:.1f} 秒",
                color,
            ),
            (action_lines[0], (245, 245, 245)),
            (action_lines[1], (245, 245, 245)),
            (action_lines[2], (245, 245, 245)),
            (warning, (80, 255, 120) if valid_sample else (40, 180, 255)),
            (count_text, (210, 210, 210)),
        ),
    )


def collect_contact_data(
    profile_dir: Path,
    camera_index: int = 0,
    auto_train: bool = True,
    guided: bool = True,
    prepare_seconds: float = 6.0,
    record_seconds: float = 5.0,
    redo_stage: Optional[int] = None,
    label_filter: Optional[str] = None,
) -> Optional[ZonedContactTrainingReport]:
    if label_filter is not None and label_filter not in CONTACT_LABELS:
        raise ValueError(f"label_filter must be one of: {', '.join(CONTACT_LABELS)}")
    if redo_stage is not None and label_filter is not None:
        raise ValueError("Use either redo_stage or label_filter, not both")
    profile = Profile.load(profile_dir)
    if profile.calibration_version < 2:
        raise RuntimeError("Contact collection requires a successful `deskpad calibrate --v2`")
    reference_path = profile_dir / profile.reference_name
    reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if reference is None:
        raise FileNotFoundError(f"Reference image not found in {profile_dir}")
    reference = _fit_frame(reference, profile.image_width, profile.image_height)
    reference_sha256 = sha256_file(reference_path)
    session_dir, metadata_path, samples_path = _session_paths(profile_dir)
    counts: Counter[str] = Counter()
    replacement_keys: Tuple[Tuple[str, str], ...] = ()
    canonical_stage: Optional[int] = None
    protocol_stages = DEFAULT_CONTACT_COLLECTION_STAGES
    if redo_stage is not None:
        if not 1 <= redo_stage <= len(DEFAULT_CONTACT_COLLECTION_STAGES):
            raise ValueError(
                f"redo_stage must be between 1 and {len(DEFAULT_CONTACT_COLLECTION_STAGES)}"
            )
        selected_stage = DEFAULT_CONTACT_COLLECTION_STAGES[redo_stage - 1]
        protocol_stages = (selected_stage,)
        canonical_stage = redo_stage
        replacement_keys = ((selected_stage.physical_side, selected_stage.label),)
    elif label_filter is not None:
        protocol_stages = tuple(
            stage for stage in DEFAULT_CONTACT_COLLECTION_STAGES if stage.label == label_filter
        )
        if {stage.label for stage in protocol_stages} != {label_filter}:
            raise RuntimeError("Internal label routing error before contact collection")
    _write_metadata(
        metadata_path,
        reference_sha256,
        profile,
        counts,
        complete=False,
        replaces=replacement_keys,
        requested_label=label_filter,
    )

    aligner = KeyboardAligner(reference, profile.keyboard_rect)
    alignment_guard = LockedAlignmentGuard(profile.image_width, profile.keyboard_rect)
    last_stable_alignment: Optional[AlignmentResult] = None
    surface = PlanarSurface.from_profile(profile)
    feature_extractor = ContactFeatureExtractor(reference)
    capture = _open_camera(camera_index, profile.image_width, profile.image_height)
    tracker: Optional[MediaPipeHandTracker] = None
    samples_file: Optional[TextIO] = None
    completed = False
    try:
        crop_y, _ = _detect_boundary_automatically(
            capture,
            target_size=(profile.image_width, profile.image_height),
            sample_count=18,
            countdown_seconds=2,
        )
        tracker = MediaPipeHandTracker()
        samples_file = samples_path.open("a", encoding="utf-8")
        label: Optional[str] = None
        settle_until = 0.0
        last_announcement: Optional[Tuple[int, str]] = None
        last_stage_key: Optional[Tuple[int, Optional[str]]] = None
        lift_anchor_plane: Optional[np.ndarray] = None
        protocol = (
            GuidedContactProtocol(
                prepare_seconds=prepare_seconds,
                record_seconds=record_seconds,
                stages=protocol_stages,
            )
            if guided
            else None
        )
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if label_filter is not None:
            cv2.setWindowTitle(
                WINDOW_NAME,
                f"Deskpad Vision - ONLY {label_filter.upper()} - {ACTION_TITLES_ZH[label_filter]}",
            )

        if guided:
            if redo_stage is None:
                if label_filter is None:
                    print("自动接触采集已就绪，不需要按任何数字键。")
                else:
                    print(
                        f"本次只采集 {label_filter.upper()}，先物理右侧、再物理左侧；"
                        "完成后数据会累加。"
                    )
            else:
                print(f"只重录原第 {redo_stage} 步；完成后会替换旧数据。")
            print(
                f"每个动作先准备 {prepare_seconds:.0f} 秒，再累计采集 "
                f"{record_seconds:.0f} 秒有效画面。"
            )
            print("先做电脑物理右侧（画面左侧），然后做电脑物理左侧（画面右侧）。")
            print("如果食指不在指定区域，采集倒计时会自动暂停。")
        else:
            print("Manual contact collection is ready. Use the number keys in the camera window:")
            print("  1 AIR: hover and move naturally without touching")
            print("  2 TOUCH: rest the index fingertip on the plane")
            print("  3 SLIDE: keep touching while sliding the fingertip")
            print("  4 LIFT: lift just above the plane")
            print("Collect every label on both physical sides, then press q.")

        while True:
            timestamp = time.monotonic()
            frame = _fit_frame(_read_frame(capture), profile.image_width, profile.image_height)
            tracking_frame = frame.copy()
            tracking_frame[crop_y:, :] = 0
            measured_alignment = aligner.align(tracking_frame, crop_y)
            geometry_trusted = alignment_guard.update(measured_alignment)
            if (
                measured_alignment is not None
                and alignment_guard.last_drift is not None
                and alignment_guard.last_drift <= alignment_guard.max_drift
            ):
                last_stable_alignment = measured_alignment
            preview_basis = measured_alignment or last_stable_alignment
            alignment: Optional[AlignmentResult] = None
            if preview_basis is not None:
                alignment = alignment_guard.preview_result(preview_basis)

            hand = tracker.process(tracking_frame)
            image_zone: Optional[str] = None
            reference_tip: Optional[Tuple[float, float]] = None
            plane_tip: Optional[Tuple[float, float]] = None
            if alignment is not None and hand is not None:
                reference_tip = apply_homography(hand.index_tip, alignment.current_to_reference)
                if reference_tip is not None:
                    plane_tip = surface.to_plane(reference_tip)
                if plane_tip is not None:
                    image_zone = surface.containing_plane(plane_tip)

            instruction = protocol.status(timestamp) if protocol is not None else None
            if instruction is not None:
                stage_key = (instruction.stage_number, instruction.label)
                if stage_key != last_stage_key:
                    lift_anchor_plane = None
                    feature_extractor.reset_motion()
                    last_stage_key = stage_key
                if (
                    label_filter is not None
                    and instruction.mode != "done"
                    and instruction.label != label_filter
                ):
                    raise RuntimeError(
                        "Displayed action does not match the requested collection command"
                    )
                announcement = (instruction.stage_number, instruction.mode)
                if announcement != last_announcement:
                    _announce_guided_instruction(instruction)
                    last_announcement = announcement
            current_physical = physical_side_for_image_zone(image_zone)
            if (
                instruction is not None
                and instruction.label == "lift"
                and instruction.mode == "prepare"
                and geometry_trusted
                and plane_tip is not None
                and current_physical == instruction.physical_side
            ):
                lift_anchor_plane = np.asarray(plane_tip, dtype=np.float64)

            effective_image_zone, lift_off_zone_tracking = _effective_lift_image_zone(
                instruction,
                image_zone,
                plane_tip,
                lift_anchor_plane,
            )
            current_physical = physical_side_for_image_zone(effective_image_zone)
            active_label = (
                instruction.label
                if instruction is not None and instruction.mode == "record"
                else label
            )
            settling = protocol is None and timestamp < settle_until
            valid_sample = (
                active_label is not None
                and not settling
                and geometry_trusted
                and hand is not None
                and reference_tip is not None
                and plane_tip is not None
                and effective_image_zone is not None
                and (
                    instruction is None
                    or current_physical == instruction.physical_side
                )
            )
            if valid_sample:
                assert active_label is not None
                if label_filter is not None and active_label != label_filter:
                    raise RuntimeError("Refusing to write a sample with the wrong action label")
                features = feature_extractor.extract(
                    timestamp=timestamp,
                    frame_bgr=tracking_frame,
                    hand=hand,
                    reference_tip=reference_tip,
                    plane_tip=plane_tip,
                    image_zone=effective_image_zone,
                )
                sample = features.to_sample(active_label)
                sample["physical_hand"] = SUPPORTED_PHYSICAL_HAND
                if instruction is not None:
                    sample["collection_mode"] = "guided"
                    sample["collection_stage"] = canonical_stage or instruction.stage_number
                samples_file.write(json.dumps(sample) + "\n")
                samples_file.flush()
                counts[active_label] += 1
            elif hand is None or effective_image_zone is None:
                feature_extractor.reset_motion()

            if protocol is not None:
                instruction = protocol.update(timestamp, valid_sample)

            display = tracking_frame.copy()
            if alignment is not None:
                _draw_surface(display, surface, alignment)
            if hand is not None:
                _draw_hand(display, hand)
            _draw_crop(display, crop_y)
            if instruction is not None:
                _draw_guided_capture_instructions(
                    display,
                    instruction,
                    counts,
                    effective_image_zone,
                    geometry_trusted,
                    valid_sample,
                    lift_off_zone_tracking,
                )
            else:
                _draw_capture_instructions(
                    display,
                    label,
                    counts,
                    image_zone,
                    geometry_trusted,
                    settling,
                )
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                completed = True
                break
            if protocol is not None and protocol.done:
                completed = True
                cv2.waitKey(300)
                break
            if protocol is None and key in LABEL_KEYS:
                label = LABEL_KEYS[key]
                settle_until = time.monotonic() + 0.35
                feature_extractor.reset_motion()
            elif protocol is None and key == ord("0"):
                label = None
                feature_extractor.reset_motion()
            elif key == ord("["):
                crop_y = max(40, crop_y - 3)
                feature_extractor.reset_motion()
            elif key == ord("]"):
                crop_y = min(profile.image_height - 10, crop_y + 3)
                feature_extractor.reset_motion()
    finally:
        if samples_file is not None:
            samples_file.close()
        if tracker is not None:
            tracker.close()
        capture.release()
        cv2.destroyAllWindows()
        _write_metadata(
            metadata_path,
            reference_sha256,
            profile,
            counts,
            complete=completed,
            replaces=replacement_keys,
            requested_label=label_filter,
        )

    print(f"Contact samples saved locally: {session_dir}")
    if not auto_train:
        return None
    all_samples = _profile_training_samples(profile_dir, reference_sha256)
    missing = _missing_training_groups(all_samples)
    if missing:
        missing_labels = sorted({label for _side, label in missing})
        print(
            "数据已保存并累加；暂不训练模型。还需要采集："
            + ", ".join(label.upper() for label in missing_labels)
        )
        return None

    report = fit_contact_model_for_profile(profile_dir)
    print_contact_training_report(report)
    return report
