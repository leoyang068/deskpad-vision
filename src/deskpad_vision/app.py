from __future__ import annotations

import time
from pathlib import Path
from threading import Event
from typing import Optional, Tuple

import cv2
import numpy as np

from .alignment import (
    AlignmentResult,
    KeyboardAligner,
    LockedAlignmentGuard,
    transform_polygon,
)
from .browser_calibration import select_geometry_in_browser
from .config import Profile, Quad
from .contact import (
    ContactFeatureExtractor,
    ContactLatch,
    ContactState,
    ContactZoneGrace,
    ZonedContactModel,
    physical_side_for_image_zone,
    sha256_file,
)
from .crop import BoundarySampler
from .deck_detection import detect_deck_side_edges, surface_quad_from_edges
from .geometry import apply_homography
from .gesture import PinchLatch, TouchMouseGesture
from .hand_tracking import HandObservation, MediaPipeHandTracker
from .plane import PlanarSurface, bounding_rect_for_quad
from .pointer import MacPointerSink, NullPointerSink, RelativePointerMapper
from .trackpad_detection import detect_trackpad

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


def _ensure_camera_access() -> None:
    """Ask macOS for camera access and wait for the user's response."""

    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
    except ImportError:
        # OpenCV can still request access itself, but it cannot wait for the async response.
        return

    status = int(AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo))
    not_determined, restricted, denied, authorized = 0, 1, 2, 3
    if status == authorized:
        return
    if status in (restricted, denied):
        raise RuntimeError(
            "Camera access is disabled. In System Settings > Privacy & Security > Camera, "
            "enable Terminal (or the terminal app you are using), then quit and reopen it."
        )
    if status != not_determined:
        raise RuntimeError(f"Unexpected macOS camera authorization status: {status}")

    finished = Event()
    result = {"granted": False}

    def on_result(granted: bool) -> None:
        result["granted"] = bool(granted)
        finished.set()

    print("macOS camera permission requested. Click Allow in the system dialog...")
    AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVMediaTypeVideo, on_result)
    if not finished.wait(timeout=120):
        raise RuntimeError("Timed out waiting for the macOS camera permission dialog")
    if not result["granted"]:
        raise RuntimeError(
            "Camera access was not granted. Enable it in System Settings > "
            "Privacy & Security > Camera, then retry."
        )


def _open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    _ensure_camera_access()
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


def _clamp_roi(
    rect: Tuple[int, int, int, int], image_width: int, image_height: int
) -> Tuple[int, int, int, int]:
    x, y, width, height = rect
    width = max(40, min(int(width), image_width))
    height = max(40, min(int(height), image_height))
    x = max(0, min(int(x), image_width - width))
    y = max(0, min(int(y), image_height - height))
    return x, y, width, height


def _default_trackpad_roi(image_width: int, image_height: int) -> Tuple[int, int, int, int]:
    width = max(40, int(round(image_width * 0.42)))
    height = max(40, int(round(image_height * 0.38)))
    x = (image_width - width) // 2
    y = max(0, image_height - height - max(8, image_height // 40))
    return _clamp_roi((x, y, width, height), image_width, image_height)


def _quad_from_rect(rect: Tuple[int, int, int, int]) -> Quad:
    x, y, width, height = rect
    return (
        (float(x), float(y)),
        (float(x + width), float(y)),
        (float(x + width), float(y + height)),
        (float(x), float(y + height)),
    )


def _line_y_at_x(start: Tuple[float, float], end: Tuple[float, float], x: float) -> float:
    delta_x = end[0] - start[0]
    if abs(delta_x) < 1e-6:
        return (start[1] + end[1]) / 2.0
    amount = (x - start[0]) / delta_x
    return start[1] + amount * (end[1] - start[1])


def _default_surface_quad(
    trackpad_quad: Quad, image_width: int, crop_y: int, margin: int = 18
) -> Quad:
    left = float(margin)
    right = float(image_width - margin)
    top_left = _line_y_at_x(trackpad_quad[0], trackpad_quad[1], left)
    top_right = _line_y_at_x(trackpad_quad[0], trackpad_quad[1], right)
    bottom_left = _line_y_at_x(trackpad_quad[3], trackpad_quad[2], left)
    bottom_right = _line_y_at_x(trackpad_quad[3], trackpad_quad[2], right)

    def usable_y(value: float) -> float:
        return max(0.0, min(float(crop_y - 2), value))

    return (
        (left, usable_y(top_left)),
        (right, usable_y(top_right)),
        (right, usable_y(bottom_right)),
        (left, usable_y(bottom_left)),
    )


def _rect_from_points(
    start: Tuple[int, int], end: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    left, right = sorted((start[0], end[0]))
    top, bottom = sorted((start[1], end[1]))
    return left, top, right - left, bottom - top


def _select_trackpad(frame: np.ndarray) -> Tuple[int, int, int, int]:
    """Select the trackpad with a mouse, with a keyboard-only fallback for macOS."""

    image_height, image_width = frame.shape[:2]
    initial = _default_trackpad_roi(image_width, image_height)
    state = {
        "rect": initial,
        "drag_start": None,
        "click_start": None,
        "cursor": None,
    }

    def on_mouse(event: int, x: int, y: int, _flags: int, _data: object) -> None:
        point = (max(0, min(x, image_width - 1)), max(0, min(y, image_height - 1)))
        state["cursor"] = point
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag_start"] = point
        elif event == cv2.EVENT_MOUSEMOVE and state["drag_start"] is not None:
            state["rect"] = _rect_from_points(state["drag_start"], point)
        elif event == cv2.EVENT_LBUTTONUP and state["drag_start"] is not None:
            start = state["drag_start"]
            state["drag_start"] = None
            candidate = _rect_from_points(start, point)
            if candidate[2] >= 8 and candidate[3] >= 8:
                state["rect"] = candidate
                state["click_start"] = None
            elif state["click_start"] is None:
                state["click_start"] = point
            else:
                state["rect"] = _rect_from_points(state["click_start"], point)
                state["click_start"] = None

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    move_step = max(3, image_width // 250)
    size_step = max(6, image_width // 125)

    try:
        while True:
            preview = frame.copy()
            rect = state["rect"]
            click_start = state["click_start"]
            cursor = state["cursor"]
            if click_start is not None and cursor is not None:
                rect = _rect_from_points(click_start, cursor)

            x, y, width, height = rect
            valid = width >= 40 and height >= 40
            color = (80, 255, 120) if valid else (40, 40, 255)
            cv2.rectangle(preview, (x, y), (x + width, y + height), color, 3)
            _draw_text(preview, "Mark the physical trackpad", 0, (80, 220, 255))
            _draw_text(preview, "Mouse: drag, or click two opposite corners", 1)
            _draw_text(preview, "Keys: WASD move | J/L width -/+ | I/K height -/+", 2)
            _draw_text(preview, "Enter accept | R reset | Q cancel", 3)
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(15) & 0xFF

            if key in (10, 13):
                if valid:
                    return _clamp_roi(rect, image_width, image_height)
                continue
            if key in (27, ord("q")):
                raise KeyboardInterrupt
            if key == ord("r"):
                state["rect"] = initial
                state["click_start"] = None
                continue

            x, y, width, height = state["rect"]
            if key == ord("a"):
                x -= move_step
            elif key == ord("d"):
                x += move_step
            elif key == ord("w"):
                y -= move_step
            elif key == ord("s"):
                y += move_step
            elif key in (ord("j"), ord("l")):
                delta = -size_step if key == ord("j") else size_step
                x -= delta // 2
                width += delta
            elif key in (ord("i"), ord("k")):
                delta = -size_step if key == ord("i") else size_step
                y -= delta // 2
                height += delta
            else:
                continue
            state["rect"] = _clamp_roi((x, y, width, height), image_width, image_height)
            state["click_start"] = None
    finally:
        cv2.setMouseCallback(WINDOW_NAME, lambda *_args: None)


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


def _detect_boundary_automatically(
    capture: cv2.VideoCapture,
    target_size: Optional[Tuple[int, int]] = None,
    sample_count: int = 24,
    countdown_seconds: int = 3,
) -> Tuple[int, np.ndarray]:
    """Capture a stable reference without relying on OpenCV window input."""

    print(
        f"Camera sampling starts in {countdown_seconds} seconds. "
        "Remove your hands from the view..."
    )
    time.sleep(countdown_seconds)

    # Discard a few frames so camera exposure has time to settle.
    for _ in range(12):
        _read_frame(capture)

    sampler = BoundarySampler(sample_count=sample_count)
    detected: Optional[int] = None
    last_frame: Optional[np.ndarray] = None
    while detected is None:
        frame = _read_frame(capture)
        if target_size:
            frame = _fit_frame(frame, target_size[0], target_size[1])
        last_frame = frame
        detected = sampler.add(frame)

    assert last_frame is not None
    crop_y = max(40, detected - 14)
    return crop_y, last_frame


def calibrate(
    profile_dir: Path,
    camera_index: int = 0,
    width: int = 1280,
    height: int = 720,
    automatic: bool = False,
    browser: bool = False,
) -> None:
    capture = _open_camera(camera_index, width, height)
    try:
        trackpad_quad: Optional[Quad] = None
        surface_quad: Optional[Quad] = None
        if automatic or browser:
            crop_y, raw_reference = _detect_boundary_automatically(capture)
            trackpad_detection = detect_trackpad(raw_reference, crop_y)
            if trackpad_detection is not None:
                initial_trackpad = trackpad_detection.quad
                initial_rect = bounding_rect_for_quad(initial_trackpad)
                trackpad_confidence = trackpad_detection.confidence
            else:
                initial_rect = _default_trackpad_roi(raw_reference.shape[1], crop_y)
                initial_trackpad = _quad_from_rect(initial_rect)
                trackpad_confidence = None
            deck_edges = detect_deck_side_edges(raw_reference, crop_y)
            initial_surface = (
                surface_quad_from_edges(initial_trackpad, deck_edges)
                if deck_edges is not None
                else None
            )
            if initial_surface is None:
                initial_surface = _default_surface_quad(
                    initial_trackpad, raw_reference.shape[1], crop_y
                )
            if browser:
                crop_y, trackpad_quad, surface_quad = select_geometry_in_browser(
                    raw_reference,
                    crop_y,
                    initial_trackpad,
                    initial_surface,
                    deck_edges,
                    trackpad_confidence,
                )
                tx, ty, tw, th = bounding_rect_for_quad(trackpad_quad)
            else:
                tx, ty, tw, th = initial_rect
                trackpad_quad = initial_trackpad
                surface_quad = initial_surface
            reference = raw_reference.copy()
            reference[crop_y:, :] = 0
        else:
            crop_y, _ = _detect_boundary_interactively(capture)
            reference = None
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
            tx, ty, tw, th = _select_trackpad(selection_frame)
        if tw < 40 or th < 40:
            raise RuntimeError("The selected trackpad rectangle is too small")

        keyboard_bottom = max(60, ty - max(8, int(reference.shape[0] * 0.025)))
        profile = Profile(
            image_width=reference.shape[1],
            image_height=reference.shape[0],
            crop_y=crop_y,
            trackpad_rect=(tx, ty, tw, th),
            keyboard_rect=(0, 0, reference.shape[1], keyboard_bottom),
            trackpad_quad=trackpad_quad,
            surface_quad=surface_quad,
        )
        profile_dir.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(profile_dir / profile.reference_name), reference):
            raise RuntimeError("Failed to write the reference image")
        profile.save(profile_dir)

        confirmation = reference.copy()
        plane = PlanarSurface.from_profile(profile)
        overlay = confirmation.copy()
        for name, polygon in plane.reference_zones.items():
            color = (80, 210, 80) if name == "left" else (255, 160, 60)
            polygon_int = np.rint(polygon).astype(np.int32)
            cv2.fillPoly(overlay, [polygon_int], color)
            cv2.polylines(confirmation, [polygon_int], True, color, 3, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.17, confirmation, 0.83, 0, confirmation)
        cv2.polylines(
            confirmation,
            [np.rint(plane.trackpad_reference).astype(np.int32)],
            True,
            (220, 100, 220),
            3,
            cv2.LINE_AA,
        )
        cv2.polylines(
            confirmation,
            [np.rint(plane.surface_reference).astype(np.int32)],
            True,
            (0, 170, 255),
            2,
            cv2.LINE_AA,
        )
        preview_path = profile_dir / "calibration-preview.png"
        cv2.imwrite(str(preview_path), confirmation)
        if automatic or browser:
            print(f"Calibration saved to {profile_dir}")
            label = "selected" if browser else "estimated"
            print(f"Crop line: y={crop_y}; {label} trackpad: {(tx, ty, tw, th)}")
            if trackpad_confidence is None:
                print("Trackpad edge detection was not reliable; the fallback shape was used.")
            else:
                print(f"Trackpad edge confidence: {trackpad_confidence:.2f}")
            print(f"Preview image: {preview_path}")
        else:
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


def _draw_surface(
    frame: np.ndarray,
    surface: PlanarSurface,
    alignment: AlignmentResult,
) -> None:
    try:
        reference_to_current = np.linalg.inv(alignment.current_to_reference)
    except np.linalg.LinAlgError:
        return
    overlay = frame.copy()
    for name, reference_polygon in surface.reference_zones.items():
        polygon = transform_polygon(reference_polygon, reference_to_current)
        polygon_int = np.rint(polygon).astype(np.int32)
        color = (60, 190, 70) if name == "left" else (230, 145, 50)
        cv2.fillPoly(overlay, [polygon_int], color)
        cv2.polylines(frame, [polygon_int], True, color, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.17, frame, 0.83, 0, frame)
    surface_polygon = transform_polygon(surface.surface_reference, reference_to_current)
    cv2.polylines(
        frame,
        [np.rint(surface_polygon).astype(np.int32)],
        True,
        (0, 170, 255),
        2,
        cv2.LINE_AA,
    )
    trackpad_polygon = transform_polygon(surface.trackpad_reference, reference_to_current)
    cv2.polylines(
        frame,
        [np.rint(trackpad_polygon).astype(np.int32)],
        True,
        (220, 100, 220),
        2,
        cv2.LINE_AA,
    )


def run(
    profile_dir: Path,
    camera_index: int = 0,
    enable_pointer: bool = False,
    gain: float = 1.5,
    max_jump: float = 70.0,
    invert_x: bool = True,
    invert_y: bool = False,
    control_mode: str = "index",
    enable_click: bool = False,
    enable_drag: bool = False,
) -> None:
    if control_mode not in {"index", "pinch", "contact"}:
        raise ValueError("control_mode must be 'index', 'pinch', or 'contact'")
    profile = Profile.load(profile_dir)
    reference = cv2.imread(str(profile_dir / profile.reference_name), cv2.IMREAD_COLOR)
    if reference is None:
        raise FileNotFoundError(f"Reference image not found in {profile_dir}")
    reference = _fit_frame(reference, profile.image_width, profile.image_height)
    aligner = KeyboardAligner(reference, profile.keyboard_rect)
    alignment_guard = (
        LockedAlignmentGuard(profile.image_width, profile.keyboard_rect)
        if profile.calibration_version >= 2
        else None
    )
    last_stable_alignment: Optional[AlignmentResult] = None
    surface = PlanarSurface.from_profile(profile)
    if not surface.plane_zones:
        raise RuntimeError("The selected trackpad leaves no usable side zones")

    capture = _open_camera(camera_index, profile.image_width, profile.image_height)
    tracker: Optional[MediaPipeHandTracker] = None
    try:
        crop_y, _ = _detect_boundary_automatically(
            capture,
            target_size=(profile.image_width, profile.image_height),
            sample_count=18,
            countdown_seconds=2,
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
        contact_model_path = profile_dir / "contact-model.json"
        contact_model: Optional[ZonedContactModel] = None
        if contact_model_path.exists():
            try:
                contact_model = ZonedContactModel.load(contact_model_path)
                if contact_model.reference_sha256 != sha256_file(
                    profile_dir / profile.reference_name
                ):
                    print("deskpad: contact model belongs to an older calibration; ignoring it")
                    contact_model = None
            except (OSError, ValueError, TypeError) as exc:
                print(f"deskpad: ignoring invalid contact model: {exc}")
        contact_features = ContactFeatureExtractor(reference)
        contact_latches = {}
        if contact_model is not None:
            for physical_zone, zone_model in contact_model.zone_models.items():
                contact_latches[physical_zone] = ContactLatch(
                    on_threshold=zone_model.threshold,
                    off_threshold=max(0.20, zone_model.threshold - 0.22),
                )
        if control_mode == "contact" and contact_model is None:
            raise RuntimeError(
                "Contact control needs a trained model. Run `deskpad collect-contact` first."
            )
        touch_mouse = TouchMouseGesture()
        contact_zone_grace = ContactZoneGrace()
        pointer_paused = False
        last_gesture_action: Optional[str] = None
        last_gesture_action_time = 0.0
        last_time = time.monotonic()
        fps = 0.0

        while True:
            frame_time = time.monotonic()
            frame = _fit_frame(_read_frame(capture), profile.image_width, profile.image_height)
            tracking_frame = frame.copy()
            tracking_frame[crop_y:, :] = 0
            measured_alignment = aligner.align(tracking_frame, crop_y)
            geometry_trusted = measured_alignment is not None
            if alignment_guard is None:
                alignment = measured_alignment
            else:
                geometry_trusted = alignment_guard.update(measured_alignment)
                if (
                    measured_alignment is not None
                    and alignment_guard.last_drift is not None
                    and alignment_guard.last_drift <= alignment_guard.max_drift
                ):
                    last_stable_alignment = measured_alignment
                preview_basis = measured_alignment or last_stable_alignment
                if preview_basis is not None:
                    alignment = alignment_guard.preview_result(preview_basis)
                else:
                    alignment = AlignmentResult(
                        current_to_reference=np.eye(3, dtype=np.float64),
                        match_count=0,
                        inlier_count=0,
                        median_error=float("inf"),
                        confidence=0.0,
                        current_inliers=np.empty((0, 2), dtype=np.float32),
                    )
            hand = tracker.process(tracking_frame)

            zone: Optional[str] = None
            contact_zone: Optional[str] = None
            reference_tip: Optional[Tuple[float, float]] = None
            plane_tip: Optional[Tuple[float, float]] = None
            clutch = False
            movement = None
            interaction_action: Optional[str] = None
            contact_probability: Optional[float] = None
            contact_state = ContactState.AIR
            if alignment is not None and hand is not None:
                reference_tip = apply_homography(hand.index_tip, alignment.current_to_reference)
                if reference_tip is not None:
                    plane_tip = surface.to_plane(reference_tip)
                if plane_tip is not None:
                    zone = surface.containing_plane(plane_tip)
                clutch = pinch.update(hand.pinch_ratio)
                if geometry_trusted:
                    contact_zone = contact_zone_grace.update(
                        timestamp=frame_time,
                        detected_image_zone=zone,
                        plane_tip=plane_tip,
                        contact_active=any(latch.active for latch in contact_latches.values()),
                    )
                else:
                    contact_zone_grace.reset()
                if (
                    contact_model is not None
                    and geometry_trusted
                    and reference_tip is not None
                    and plane_tip is not None
                    and contact_zone is not None
                ):
                    features = contact_features.extract(
                        timestamp=frame_time,
                        frame_bgr=tracking_frame,
                        hand=hand,
                        reference_tip=reference_tip,
                        plane_tip=plane_tip,
                        image_zone=contact_zone,
                    )
                    contact_probability = contact_model.probability(features)
                    active_latch = contact_latches[features.physical_zone]
                    for physical_zone, latch in contact_latches.items():
                        if physical_zone != features.physical_zone:
                            latch.reset()
                    contact_state = active_latch.update(contact_probability)
                else:
                    contact_features.reset_motion()
                    for latch in contact_latches.values():
                        latch.reset()
            else:
                pinch.reset()
                mapper.reset()
                contact_features.reset_motion()
                contact_zone_grace.reset()
                for latch in contact_latches.values():
                    latch.reset()

            mouse_input_valid = (
                geometry_trusted
                and hand is not None
                and plane_tip is not None
                and zone is not None
            )
            if control_mode == "contact":
                interaction_action = touch_mouse.update(
                    timestamp=frame_time,
                    point=plane_tip,
                    zone=contact_zone,
                    touching=contact_state == ContactState.TOUCH,
                    released=contact_state == ContactState.RELEASE,
                    valid=(
                        geometry_trusted
                        and hand is not None
                        and plane_tip is not None
                        and contact_zone is not None
                        and not pointer_paused
                    ),
                    sink=sink,
                    emit_click=enable_pointer and enable_click and not pointer_paused,
                    emit_drag=enable_pointer and enable_drag and not pointer_paused,
                )
                movement_clutch = contact_state == ContactState.TOUCH
            elif control_mode == "pinch":
                touch_mouse.cancel(sink)
                movement_clutch = clutch
            else:
                # INDEX mode intentionally has no gesture clutch: entering either
                # usable side zone starts relative control and leaving it stops.
                touch_mouse.cancel(sink)
                movement_clutch = True

            if pointer_paused or not mouse_input_valid:
                mapper.reset()
            else:
                movement = mapper.update(plane_tip, zone, movement_clutch)
            if interaction_action is not None:
                last_gesture_action = interaction_action
                last_gesture_action_time = frame_time

            display = tracking_frame.copy()
            if alignment is not None:
                _draw_surface(display, surface, alignment)
            if measured_alignment is not None:
                for point in measured_alignment.current_inliers[:80]:
                    cv2.circle(display, tuple(np.rint(point).astype(int)), 2, (255, 200, 40), -1)
            if hand is not None:
                _draw_hand(display, hand)
            _draw_crop(display, crop_y)

            now = time.monotonic()
            instantaneous = 1.0 / max(1e-6, now - last_time)
            fps = instantaneous if fps == 0 else fps * 0.9 + instantaneous * 0.1
            last_time = now

            if alignment_guard is not None and not geometry_trusted:
                drift = alignment_guard.last_drift
                if drift is not None and drift > alignment_guard.max_drift:
                    alignment_text = (
                        f"PLANE VISIBLE / POINTER FROZEN - drift {drift:.1f}px"
                    )
                else:
                    alignment_text = "PLANE VISIBLE / checking drift - pointer frozen"
                alignment_color = (40, 180, 255)
            elif alignment is None:
                alignment_text = "ALIGNMENT LOST - pointer frozen"
                alignment_color = (40, 40, 255)
            elif alignment_guard is not None:
                alignment_text = (
                    f"geometry LOCKED: {alignment.inlier_count}/{alignment.match_count} inliers  "
                    f"drift {alignment_guard.last_drift or 0.0:.1f}px"
                )
                alignment_color = (80, 255, 120)
            else:
                alignment_text = (
                    f"aligned: {alignment.inlier_count}/{alignment.match_count} inliers  "
                    f"error {alignment.median_error:.1f}px"
                )
                alignment_color = (80, 255, 120)
            _draw_text(display, alignment_text, 0, alignment_color)
            physical_zone = physical_side_for_image_zone(zone)
            zone_text = "-"
            if zone is not None:
                zone_text = f"image-{zone} / physical-{physical_zone}"
            if control_mode == "index":
                input_text = f"index: {'ACTIVE' if zone is not None else 'outside'}"
                input_active = zone is not None
            elif control_mode == "pinch":
                input_text = f"pinch: {'CLUTCH' if clutch else 'open'}"
                input_active = clutch
            else:
                input_text = f"contact control: {contact_state.value}"
                input_active = contact_state == ContactState.TOUCH
            _draw_text(
                display,
                f"zone: {zone_text}   {input_text}   fps: {fps:.0f}",
                1,
                (80, 230, 255) if input_active else (230, 230, 230),
            )
            if contact_model is None:
                contact_text = "contact: UNCALIBRATED - run deskpad collect-contact"
                contact_color = (80, 220, 255)
            else:
                probability_text = (
                    "-" if contact_probability is None else f"{contact_probability:.2f}"
                )
                contact_text = (
                    f"contact: {contact_state.value}   probability: {probability_text}   "
                    f"physical hand: {contact_model.physical_hand.upper()}"
                )
                contact_color = (
                    (80, 255, 120)
                    if contact_state == ContactState.TOUCH
                    else (230, 230, 230)
                )
            _draw_text(display, contact_text, 2, contact_color)
            mode = "LIVE POINTER" if enable_pointer and not pointer_paused else "SAFE PREVIEW"
            if enable_pointer and pointer_paused:
                mode = "POINTER PAUSED"
            direction = f"invert x={mapper.invert_x} y={mapper.invert_y}"
            buttons = "buttons PREVIEW"
            if enable_pointer:
                buttons = (
                    f"click {'ON' if enable_click else 'OFF'} / "
                    f"drag {'ON' if enable_drag else 'OFF'}"
                )
            _draw_text(
                display,
                f"{mode}   control {control_mode.upper()}   {buttons}",
                3,
            )
            _draw_text(display, f"{direction}   q quit / p pause / x,y invert", 4)
            if movement is not None:
                _draw_text(display, f"delta: {movement[0]:+.1f}, {movement[1]:+.1f}", 5)
            if (
                last_gesture_action is not None
                and frame_time - last_gesture_action_time <= 1.0
            ):
                _draw_text(display, f"mouse gesture: {last_gesture_action}", 6, (80, 230, 255))

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("p") and enable_pointer:
                pointer_paused = not pointer_paused
                mapper.reset()
                touch_mouse.cancel(sink)
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
        if "touch_mouse" in locals() and "sink" in locals():
            touch_mouse.cancel(sink)
        if tracker is not None:
            tracker.close()
        capture.release()
        cv2.destroyAllWindows()
