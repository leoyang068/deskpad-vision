from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config import Profile
from ..crop import BoundarySampler, detect_horizontal_boundary
from ..plane import bounding_rect_for_quad
from .deep_lines import detect_lines_with_model
from .geometry import (
    GeometryError,
    add_constrained_gradient_sides,
    solve_geometry,
)
from .types import CalibrationGeometry, Line


@dataclass(frozen=True)
class CalibrationOutput:
    profile: Profile
    geometry: CalibrationGeometry
    reference_path: Path
    preview_path: Path
    report_path: Path


def _temporal_median(frames: Sequence[np.ndarray]) -> np.ndarray:
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)


def _select_model_frames(frames: Sequence[np.ndarray], crop_y: int) -> List[np.ndarray]:
    # Fuse first, infer once. Discrete line selection on several nearly identical
    # frames can jump between parallel highlights even when the camera is static.
    # A temporal median preserves their shared physical edges and removes sensor
    # noise before the learned detector makes that discrete decision.
    median = _temporal_median(frames)
    return [median[:crop_y].copy()]


def _capture_motion_score(frames: Sequence[np.ndarray], crop_y: int) -> float:
    if len(frames) <= 1:
        return 0.0
    median = _temporal_median(frames)
    # Only assess the upper laptop area. Movement below the mirror crop can be a
    # shirt or chair and must not invalidate otherwise stable laptop geometry.
    assessment_height = max(80, min(crop_y, int(round(crop_y * 0.58))))
    median_gray = cv2.cvtColor(median[:assessment_height], cv2.COLOR_BGR2GRAY)
    differences = []
    for frame in frames:
        gray = cv2.cvtColor(frame[:assessment_height], cv2.COLOR_BGR2GRAY)
        differences.append(float(np.mean(cv2.absdiff(gray, median_gray))))
    return float(np.percentile(differences, 90))


def analyze_frames(
    frames: Sequence[np.ndarray], crop_y: int, debug_dir: Optional[Path] = None
) -> Tuple[np.ndarray, CalibrationGeometry, List[str]]:
    if not frames:
        raise ValueError("至少需要一帧图像")
    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        raise ValueError("多帧校准图像尺寸不一致")
    crop_y = max(100, min(int(crop_y), shape[0]))
    reference = _temporal_median(frames)
    motion_score = _capture_motion_score(frames, crop_y)
    if motion_score > 18.0:
        raise GeometryError(
            f"采集期间电脑或镜子有明显移动（变化分数 {motion_score:.1f}），请保持静止重试"
        )
    model_frames = _select_model_frames(frames, crop_y)
    line_sets = detect_lines_with_model(model_frames)

    geometries = []
    rejected = []
    for index, (model_frame, lines) in enumerate(zip(model_frames, line_sets)):
        try:
            geometries.append(solve_geometry(lines, model_frame.shape))
        except GeometryError as primary_error:
            try:
                augmented = add_constrained_gradient_sides(model_frame, lines)
                geometry = solve_geometry(augmented, model_frame.shape)
                geometry.metrics["gradient_side_recovery"] = 1.0
                geometries.append(geometry)
            except GeometryError as recovery_error:
                rejected.append(
                    f"frame {index}: {primary_error}；侧边恢复失败：{recovery_error}"
                )
    if not geometries:
        details = "；".join(rejected) if rejected else "没有有效几何解"
        suffix = ""
        if debug_dir is not None:
            _save_failure_debug(debug_dir, model_frames, line_sets, rejected)
            suffix = f"；诊断图已保存到 {debug_dir}"
        raise GeometryError(f"时间中值图质量门未通过：{details}{suffix}")
    geometry = geometries[0]
    geometry.metrics["capture_frame_count"] = float(len(frames))
    geometry.metrics["capture_motion_score"] = motion_score
    reference[crop_y:, :] = 0
    return reference, geometry, rejected


def _save_failure_debug(
    debug_dir: Path,
    frames: Sequence[np.ndarray],
    line_sets: Sequence[np.ndarray],
    errors: Sequence[str],
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    for index, (frame, lines) in enumerate(zip(frames, line_sets)):
        cv2.imwrite(str(debug_dir / f"frame-{index}.png"), frame)
        np.save(debug_dir / f"lines-{index}.npy", lines)
        overlay = frame.copy()
        for raw in lines:
            length = float(np.linalg.norm(raw[1] - raw[0]))
            if length < 24:
                continue
            first, second = np.rint(raw).astype(int)
            color = (0, 210, 255) if length >= 80 else (255, 170, 0)
            cv2.line(overlay, tuple(first), tuple(second), color, 1, cv2.LINE_AA)
        cv2.imwrite(str(debug_dir / f"lines-{index}.png"), overlay)
    (debug_dir / "errors.json").write_text(
        json.dumps(list(errors), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _draw_line(frame: np.ndarray, line: Line, color: Tuple[int, int, int], width: int) -> None:
    first = tuple(np.rint(line[0]).astype(int))
    second = tuple(np.rint(line[1]).astype(int))
    cv2.line(frame, first, second, color, width, cv2.LINE_AA)


def render_preview(
    reference: np.ndarray, crop_y: int, geometry: CalibrationGeometry
) -> np.ndarray:
    preview = reference.copy()
    evidence_layer = preview.copy()
    for evidence in geometry.evidence_lines:
        length = float(np.hypot(
            evidence.second[0] - evidence.first[0],
            evidence.second[1] - evidence.first[1],
        ))
        if length < 34:
            continue
        _draw_line(evidence_layer, evidence.points, (150, 150, 150), 1)
    cv2.addWeighted(evidence_layer, 0.28, preview, 0.72, 0, preview)

    surface = np.rint(np.asarray(geometry.surface_quad)).astype(np.int32)
    trackpad = np.rint(np.asarray(geometry.trackpad_quad)).astype(np.int32)
    fill = preview.copy()
    cv2.fillPoly(fill, [surface], (255, 180, 40))
    cv2.addWeighted(fill, 0.10, preview, 0.90, 0, preview)
    cv2.polylines(preview, [surface], True, (255, 190, 50), 2, cv2.LINE_AA)
    cv2.polylines(preview, [trackpad], True, (220, 80, 220), 3, cv2.LINE_AA)
    _draw_line(preview, geometry.chassis_left, (0, 150, 255), 3)
    _draw_line(preview, geometry.chassis_right, (0, 150, 255), 3)
    _draw_line(preview, geometry.chassis_front, (0, 150, 255), 3)
    cv2.line(preview, (0, crop_y), (preview.shape[1] - 1, crop_y), (0, 0, 255), 2)

    labels = [
        f"CALIBRATION V2  confidence {geometry.confidence:.2f}",
        "orange: physical chassis L/R/front",
        "purple: detected physical trackpad",
        "cyan: perspective-correct palm-rest band",
    ]
    for row, label in enumerate(labels):
        y = 27 + row * 24
        cv2.putText(preview, label, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(
            preview,
            label,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 245, 245),
            1,
        )
    return preview


def save_output(
    profile_dir: Path,
    reference: np.ndarray,
    crop_y: int,
    geometry: CalibrationGeometry,
    rejected_frames: Sequence[str] = (),
) -> CalibrationOutput:
    image_height, image_width = reference.shape[:2]
    trackpad_rect = bounding_rect_for_quad(geometry.trackpad_quad)
    top_y = min(point[1] for point in geometry.trackpad_quad)
    keyboard_bottom = max(60, int(round(top_y - image_height * 0.015)))
    profile = Profile(
        image_width=image_width,
        image_height=image_height,
        crop_y=crop_y,
        trackpad_rect=trackpad_rect,
        keyboard_rect=(0, 0, image_width, keyboard_bottom),
        trackpad_quad=geometry.trackpad_quad,
        surface_quad=geometry.surface_quad,
        calibration_version=2,
    )

    profile_dir.mkdir(parents=True, exist_ok=True)
    reference_path = profile_dir / profile.reference_name
    preview_path = profile_dir / "calibration-v2-preview.png"
    report_path = profile_dir / "calibration-v2-report.json"
    preview = render_preview(reference, crop_y, geometry)

    # Nothing is written until every model and geometry quality gate has passed.
    if not cv2.imwrite(str(reference_path), reference):
        raise RuntimeError(f"无法保存参考图：{reference_path}")
    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(f"无法保存校准预览：{preview_path}")
    profile.save(profile_dir)
    report = {
        "version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "detector": "DeepLSD MegaDepth",
        "quality_gate": "passed",
        "crop_y": crop_y,
        "geometry": geometry.to_dict(),
        "rejected_frames": list(rejected_frames),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return CalibrationOutput(profile, geometry, reference_path, preview_path, report_path)


def capture_frames(
    camera_index: int, width: int, height: int, frame_count: int = 9
) -> Tuple[List[np.ndarray], int]:
    # Importing here avoids coupling the model-independent app startup to V2.
    from ..app import _fit_frame, _open_camera, _read_frame

    capture = _open_camera(camera_index, width, height)
    try:
        print("3 秒后开始校准。请移开双手，并保持电脑和镜子不动。")
        time.sleep(3)
        for _ in range(10):
            _read_frame(capture)
        sampler = BoundarySampler(sample_count=24)
        crop_y: Optional[int] = None
        while crop_y is None:
            frame = _fit_frame(_read_frame(capture), width, height)
            detected = sampler.add(frame)
            if detected is not None:
                crop_y = max(100, detected - 14)
        frames = []
        for _ in range(max(3, frame_count)):
            frames.append(_fit_frame(_read_frame(capture), width, height))
        return frames, crop_y
    finally:
        capture.release()


def calibrate_v2(
    profile_dir: Path,
    camera_index: int = 0,
    width: int = 1280,
    height: int = 720,
    image_path: Optional[Path] = None,
    crop_y: Optional[int] = None,
) -> CalibrationOutput:
    if image_path is not None:
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"无法读取图像：{image_path}")
        frames = [frame]
        if crop_y is None:
            crop_y = max(100, detect_horizontal_boundary(frame) - 14)
    else:
        frames, detected_crop = capture_frames(camera_index, width, height)
        crop_y = crop_y if crop_y is not None else detected_crop
    assert crop_y is not None

    print("正在运行校准专用强模型并求解物理几何，通常需要 10–20 秒…")
    reference, geometry, rejected = analyze_frames(
        frames,
        crop_y,
        debug_dir=profile_dir / "calibration-v2-debug",
    )
    output = save_output(profile_dir, reference, crop_y, geometry, rejected)
    print(f"校准 V2 已通过质量门并保存：置信度 {geometry.confidence:.2f}")
    print(f"三边与触控板预览：{output.preview_path}")
    print(f"详细质量报告：{output.report_path}")
    return output
