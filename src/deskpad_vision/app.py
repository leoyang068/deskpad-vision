from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .alignment import AlignmentResult, KeyboardAligner, transform_polygon
from .config import Profile
from .crop import BoundarySampler
from .geometry import ActiveZones, apply_homography, rect_polygon
from .gesture import PinchLatch
from .hand_tracking import HandObservation, MediaPipeHandTracker
from .pointer import MacPointerSink, NullPointerSink, RelativePointerMapper

WINDOW_NAME = "Deskpad Vision"
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


def _open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {index}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def _read_frame(capture: cv2.VideoCapture) -> np.ndarray:
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError("The camera stopped returning frames")
    return frame


def _fit_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _draw_text(
    frame: np.ndarray,
    text: str,
    row: int,
    color: Tuple[int, int, int] = (240, 240, 240),
) -> None:
    y = 28 + row * 25
    cv2.putText(frame, text, (13, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
    cv2.putText(frame, text, (13, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1)


def _draw_crop(frame: np.ndarray, crop_y: int) -> None:
    height = frame.shape[0]
    crop_y = max(0, min(height, crop_y))
    if crop_y < height:
        frame[crop_y:, :] = 0
    cv2.line(frame, (0, crop_y), (frame.shape[1] - 1, crop_y), (0, 0, 255), 2)


def _detect_boundary_interactively(
    capture: cv2.VideoCapture,
    target_size: Optional[Tuple[int, int]] = None,
    sample_count: int = 24,
) -> Tuple[int, np.ndarray]:
    sampler = BoundarySampler(sample_count=sample_count)
    detected: Optional[int] = None
    last_frame: Optional[np.ndarray] = None
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while detected is None:
        frame = _read_frame(capture)
        if target_size:
            frame = _fit_frame(frame, target_size[0], target_size[1])
        last_frame = frame
        detected = sampler.add(frame)
        preview = frame.copy()
        progress = min(sample_count, sampler.progress)
        _draw_text(preview, f"Detecting mirror boundary: {progress}/{sample_count}", 0)
        _draw_text(preview, "Keep the camera view still and hands out of frame", 1)
        cv2.imshow(WINDOW_NAME, preview)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            raise KeyboardInterrupt

    assert last_frame is not None
    crop_y = max(40, detected - 14)
    while True:
        frame = _read_frame(capture)
        if target_size:
            frame = _fit_frame(frame, target_size[0], target_size[1])
        last_frame = frame
        preview = frame.copy()
        _draw_crop(preview, crop_y)
        _draw_text(preview, "Proposed crop line", 0, (80, 220, 255))
        _draw_text(preview, "[ / ] adjust    Enter accept    q quit", 1)
        cv2.imshow(WINDOW_NAME, preview)
        key = cv2.waitKey(15) & 0xFF
        if key in (10, 13):
            return crop_y, last_frame
        if key == ord("["):
            crop_y = max(40, crop_y - 3)
        elif key == ord("]"):
            crop_y = min(frame.shape[0] - 10, crop_y + 3)
        elif key in (27, ord("q")):
            raise KeyboardInterrupt


def calibrate(
    profile_dir: Path,
    camera_index: int = 0,
    width: int = 1280,
    height: int = 720,
) -> None:
    capture = _open_camera(camera_index, width, height)
    try:
        crop_y, _ = _detect_boundary_interactively(capture)
        reference: Optional[np.ndarray] = None
        while reference is None:
            frame = _read_frame(capture)
            preview = frame.copy()
            _draw_crop(preview, crop_y)
            _draw_text(preview, "Remove hands, then press Space to capture reference", 0)
            _draw_text(preview, "q quits without saving", 1)
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(15) & 0xFF
            if key == ord(" "):
                reference = frame.copy()
                reference[crop_y:, :] = 0
            elif key in (27, ord("q")):
                raise KeyboardInterrupt

        selection_frame = reference[:crop_y, :].copy()
        roi = cv2.selectROI(
            "Select the physical trackpad, then press Enter",
            selection_frame,
            showCrosshair=True,
            fromCenter=False,
        )
        cv2.destroyWindow("Select the physical trackpad, then press Enter")
        tx, ty, tw, th = [int(value) for value in roi]
        if tw < 40 or th < 40:
            raise RuntimeError("The selected trackpad rectangle is too small")

        keyboard_bottom = max(60, ty - max(8, int(reference.shape[0] * 0.025)))
        profile = Profile(
            image_width=reference.shape[1],
            image_height=reference.shape[0],
            crop_y=crop_y,
            trackpad_rect=(tx, ty, tw, th),
            keyboard_rect=(0, 0, reference.shape[1], keyboard_bottom),
        )
        profile_dir.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(profile_dir / profile.reference_name), reference):
            raise RuntimeError("Failed to write the reference image")
        profile.save(profile_dir)

        confirmation = reference.copy()
        zones = ActiveZones.from_profile(profile)
        for name, rect in zones.zones.items():
            color = (80, 210, 80) if name == "left" else (255, 160, 60)
            x, y, rect_width, rect_height = rect
            cv2.rectangle(confirmation, (x, y), (x + rect_width, y + rect_height), color, 3)
        cv2.rectangle(confirmation, (tx, ty), (tx + tw, ty + th), (220, 100, 220), 2)
        _draw_text(confirmation, f"Saved profile to {profile_dir}", 0, (80, 255, 120))
        _draw_text(confirmation, "Press any key to close", 1)
        cv2.imshow(WINDOW_NAME, confirmation)
        cv2.waitKey(0)
    finally:
        capture.release()
        cv2.destroyAllWindows()


def _draw_hand(frame: np.ndarray, hand: HandObservation) -> None:
    points = [(int(round(point.x)), int(round(point.y))) for point in hand.landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (70, 210, 255), 2, cv2.LINE_AA)
    for index, point in enumerate(points):
        radius = 6 if index in (4, 8) else 3
        color = (0, 80, 255) if index == 8 else (60, 255, 120)
        cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)


def _draw_zones(
    frame: np.ndarray,
    zones: ActiveZones,
    alignment: AlignmentResult,
) -> None:
    try:
        reference_to_current = np.linalg.inv(alignment.current_to_reference)
    except np.linalg.LinAlgError:
        return
    overlay = frame.copy()
    for name, rect in zones.zones.items():
        polygon = transform_polygon(rect_polygon(rect), reference_to_current)
        polygon_int = np.rint(polygon).astype(np.int32)
        color = (60, 190, 70) if name == "left" else (230, 145, 50)
        cv2.fillPoly(overlay, [polygon_int], color)
        cv2.polylines(frame, [polygon_int], True, color, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.17, frame, 0.83, 0, frame)


def run(
    profile_dir: Path,
    camera_index: int = 0,
    enable_pointer: bool = False,
    gain: float = 1.5,
    max_jump: float = 70.0,
    invert_x: bool = False,
    invert_y: bool = False,
) -> None:
    profile = Profile.load(profile_dir)
    reference = cv2.imread(str(profile_dir / profile.reference_name), cv2.IMREAD_COLOR)
    if reference is None:
        raise FileNotFoundError(f"Reference image not found in {profile_dir}")
    reference = _fit_frame(reference, profile.image_width, profile.image_height)
    aligner = KeyboardAligner(reference, profile.keyboard_rect)
    zones = ActiveZones.from_profile(profile)
    if not zones.zones:
        raise RuntimeError("The selected trackpad leaves no usable side zones")

    capture = _open_camera(camera_index, profile.image_width, profile.image_height)
    tracker: Optional[MediaPipeHandTracker] = None
    try:
        crop_y, _ = _detect_boundary_interactively(
            capture,
            target_size=(profile.image_width, profile.image_height),
            sample_count=18,
        )
        tracker = MediaPipeHandTracker()
        sink = MacPointerSink() if enable_pointer else NullPointerSink()
        mapper = RelativePointerMapper(
            sink=sink,
            gain=gain,
            max_jump=max_jump,
            invert_x=invert_x,
            invert_y=invert_y,
        )
        pinch = PinchLatch()
        pointer_paused = False
        last_time = time.monotonic()
        fps = 0.0

        while True:
            frame = _fit_frame(_read_frame(capture), profile.image_width, profile.image_height)
            tracking_frame = frame.copy()
            tracking_frame[crop_y:, :] = 0
            alignment = aligner.align(tracking_frame, crop_y)
            hand = tracker.process(tracking_frame)

            zone: Optional[str] = None
            reference_tip: Optional[Tuple[float, float]] = None
            clutch = False
            movement = None
            if alignment is not None and hand is not None:
                reference_tip = apply_homography(hand.index_tip, alignment.current_to_reference)
                if reference_tip is not None:
                    zone = zones.containing(reference_tip)
                clutch = pinch.update(hand.pinch_ratio)
                if pointer_paused:
                    mapper.reset()
                else:
                    movement = mapper.update(reference_tip, zone, clutch) if reference_tip else None
            else:
                pinch.reset()
                mapper.reset()

            display = tracking_frame.copy()
            if alignment is not None:
                _draw_zones(display, zones, alignment)
                for point in alignment.current_inliers[:80]:
                    cv2.circle(display, tuple(np.rint(point).astype(int)), 2, (255, 200, 40), -1)
            if hand is not None:
                _draw_hand(display, hand)
            _draw_crop(display, crop_y)

            now = time.monotonic()
            instantaneous = 1.0 / max(1e-6, now - last_time)
            fps = instantaneous if fps == 0 else fps * 0.9 + instantaneous * 0.1
            last_time = now

            if alignment is None:
                alignment_text = "ALIGNMENT LOST - pointer frozen"
                alignment_color = (40, 40, 255)
            else:
                alignment_text = (
                    f"aligned: {alignment.inlier_count}/{alignment.match_count} inliers  "
                    f"error {alignment.median_error:.1f}px"
                )
                alignment_color = (80, 255, 120)
            _draw_text(display, alignment_text, 0, alignment_color)
            _draw_text(
                display,
                f"zone: {zone or '-'}   pinch: {'CLUTCH' if clutch else 'open'}   fps: {fps:.0f}",
                1,
                (80, 230, 255) if clutch else (230, 230, 230),
            )
            mode = "LIVE POINTER" if enable_pointer and not pointer_paused else "SAFE PREVIEW"
            if enable_pointer and pointer_paused:
                mode = "POINTER PAUSED"
            direction = f"invert x={mapper.invert_x} y={mapper.invert_y}"
            _draw_text(display, f"{mode}   {direction}   q quit / p pause / x,y invert", 2)
            if movement is not None:
                _draw_text(display, f"delta: {movement[0]:+.1f}, {movement[1]:+.1f}", 3)

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("p") and enable_pointer:
                pointer_paused = not pointer_paused
                mapper.reset()
            elif key == ord("x"):
                mapper.invert_x = not mapper.invert_x
                mapper.reset()
            elif key == ord("y"):
                mapper.invert_y = not mapper.invert_y
                mapper.reset()
            elif key == ord("["):
                crop_y = max(40, crop_y - 3)
                mapper.reset()
            elif key == ord("]"):
                crop_y = min(profile.image_height - 10, crop_y + 3)
                mapper.reset()
    finally:
        if tracker is not None:
            tracker.close()
        capture.release()
        cv2.destroyAllWindows()
