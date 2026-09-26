# Deskpad Vision

Deskpad Vision is an early macOS prototype that uses a mirror-mounted webcam to turn the
palm-rest surfaces on the left and right of a MacBook trackpad into virtual pointing areas.

The current milestone deliberately separates two problems:

1. **Surface tracking:** crop the mirror boundary, recognize the keyboard, and map every frame
   back into one stable reference view even when the screen angle changes.
2. **Input:** track the index finger and use an index-thumb pinch as a safe clutch for relative
   pointer movement.

True fingertip-to-surface contact and tap recognition are the next milestone. The pointer is
disabled by default so an imperfect calibration cannot move the real cursor unexpectedly.

## What is implemented

- Automatic detection of the persistent horizontal mirror boundary, with manual fine tuning.
- One-time reference capture and trackpad rectangle selection.
- Keyboard feature matching with ORB + RANSAC homography.
- Automatically derived left and right palm-rest zones.
- MediaPipe single-hand landmark tracking.
- Pinch hysteresis, smoothed relative motion, jump rejection, and alignment fail-safe behavior.
- Optional macOS pointer output through Quartz.
- A visual overlay for crop, alignment quality, zones, fingertip, and clutch state.

## Requirements

- macOS with a camera and the mirror attachment.
- Python 3.9 or newer.
- A reasonably even light source. Keyboard glare reduces alignment quality.
- Camera permission. Real pointer output also requires Accessibility permission.

## Install

```bash
cd deskpad-vision
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## 1. Calibrate once

Make sure the keyboard and both palm-rest areas are visible and remove your hands from the
frame. Then run:

```bash
deskpad calibrate
```

Calibration has three short steps:

1. The app collects several frames and proposes a red crop line above the mirror boundary.
2. Use `[` and `]` to move it, then press `Enter`.
3. Press `Space` on a clear frame, drag a rectangle around the physical trackpad, and confirm.

The default profile is stored in `~/.deskpad-vision`. It contains a reference image and a small
JSON file with the rectangles and settings. Recalibration is only necessary if the mirror mount
or camera framing changes substantially.

## 2. Preview safely

```bash
deskpad run
```

Pinch the index finger and thumb while the index fingertip is over either colored side zone.
The overlay should show `CLUTCH`, but the real pointer will not move.

Runtime keys:

- `q` or `Esc`: quit
- `p`: pause/resume pointer output (only when output was enabled on launch)
- `[` / `]`: adjust the crop line
- `x`: invert horizontal pointer direction
- `y`: invert vertical pointer direction

## 3. Enable the macOS pointer

After preview mode is stable:

```bash
deskpad run --enable-pointer
```

macOS will need to allow the terminal application under **System Settings → Privacy & Security →
Accessibility**. Camera access is requested separately by macOS.

Useful tuning options:

```bash
deskpad run --camera 0 --gain 1.5 --max-jump 70
deskpad run --invert-x
deskpad --help
```

## Safety behavior

Deskpad Vision stops producing pointer movement when:

- the keyboard alignment does not have enough inlier matches;
- reprojection error is too high;
- the hand or fingertip is lost;
- the fingertip leaves both side zones;
- the pinch clutch is released; or
- a single-frame movement exceeds the configured jump limit.

The center trackpad area is excluded, so the physical trackpad remains usable.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Current limitations

- A dark, reflective keyboard cover can reduce the number of stable feature matches.
- The palm-rest zones are treated as a single reference plane. Millimeter-accurate height is not
  estimated yet.
- Pinch is an intentional temporary clutch, not surface contact detection.
- Only one hand is tracked in the current milestone.

## Next milestone: real touch

The next stage will record short labeled sequences of hover, touch, slide, and lift. A temporal
contact estimator will combine landmark depth changes, finger joint angles, fingertip velocity,
background difference, and the local fingertip-shadow gap. Its output will drive an
`AIR → CANDIDATE → TOUCH → RELEASE` state machine with asymmetric thresholds to prioritize
avoiding false clicks.

