# Deskpad Vision

Deskpad Vision turns the palm-rest surfaces beside a MacBook trackpad into camera-tracked pointing
areas using a small mirror mounted over the webcam.

<p align="center">
  <a href="https://www.youtube.com/shorts/NJ3W3RG85_Y">
    <img src="https://img.youtube.com/vi/NJ3W3RG85_Y/hqdefault.jpg" width="560" alt="Deskpad Vision demo video">
  </a>
  <br>
  <strong><a href="https://www.youtube.com/shorts/NJ3W3RG85_Y">▶ Watch the real-world demo</a></strong>
</p>

The default control is deliberately simple: when the right index fingertip enters either colored
side zone, its motion controls the pointer; leaving both zones stops control. Pinch and learned
surface-contact modes remain available for experiments. [中文技术设计](docs/design.zh-CN.md)

## What works

- Calibration V2 detects the laptop's two side edges, front edge, and the real trackpad edges.
- Perspective geometry maps both palm-rest regions into one canonical physical plane.
- A strong DeepLSD line model runs only during calibration; live tracking stays lightweight.
- Multi-frame calibration, locked runtime alignment, and drift alarms prevent jumping polygons.
- MediaPipe tracks one right hand and its index fingertip in real time.
- Relative pointer motion includes smoothing, gain, axis control, and jump rejection.
- A personalized AIR/TOUCH/SLIDE/LIFT collector, trainer, and temporal contact state machine exist,
  but contact is not the default mouse gate while its accuracy is still being improved.
- Real macOS pointer events use Quartz and remain opt-in.

## Architecture and development method

The engineering workflow was inspired by a friend's tennis ball/court tracking project. No tennis
code is copied. The transferable idea is to separate perception, temporal stabilization, physical
geometry, event inference, and evaluation instead of asking one model to guess the final answer.

| Tennis-system pattern | Deskpad Vision adaptation |
| --- | --- |
| Court lines + homography/camera pose | Laptop/trackpad edges + canonical palm-rest plane |
| Ball/player detection | Index fingertip + 21 hand landmarks |
| Multi-frame tracking | Temporal calibration fusion, locked alignment, and state hysteresis |
| Physics-informed optimization | Vanishing-point, containment, convexity, and drift constraints |
| Landing/bounce events | Zone entry, movement, contact, release, tap, and drag |
| Trajectory visualization and metrics | Overlays, calibration previews, JSON reports, and quality gates |

```text
camera → crop/temporal fusion → calibration-only learned lines → constrained plane geometry
       → locked runtime alignment → hand landmarks → zone/gesture state → relative mouse events
```

This is the central design rule: **strong models run rarely; small deterministic components run
live; uncertain geometry fails closed**. Kalman, ByteTrack, and optical flow informed the temporal
design, but are not currently claimed as implemented components.

## Requirements and install

- macOS, Python 3.9+, a camera, and the mirror attachment.
- Camera permission; real pointer output also needs Accessibility permission.

```bash
git clone https://github.com/leoyang068/deskpad-vision.git
cd deskpad-vision
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
./scripts/setup-calibration-v2.sh
```

The last command creates an ignored Python 3.10 environment for the official DeepLSD code and
checkpoint, keeping PyTorch out of the lightweight live runtime.

## Quick start

### 1. Calibrate after changing the screen or mirror angle

Remove both hands, keep the computer still, and run:

```bash
deskpad calibrate --v2
```

The solver fuses a short camera burst, detects line evidence once, and searches for a physically
consistent laptop/trackpad geometry. Image-space rectangles are not assumed: parallel physical
edges may converge under perspective. A low-confidence result never overwrites the last valid
profile.

Review the generated files:

```text
~/.deskpad-vision/calibration-v2-preview.png
~/.deskpad-vision/calibration-v2-report.json
```

Orange marks the chassis edges, purple the trackpad, and cyan the usable palm-rest depth band.
The browser-assisted fallback is `deskpad calibrate`; legacy automatic and OpenCV modes are
available as `--auto` and `--opencv`.

### 2. Preview without controlling the real cursor

```bash
deskpad run
```

Move the right index fingertip into either colored side zone. The overlay should show
`control INDEX`, `index: ACTIVE`, and changing deltas. The center trackpad is excluded. Because the
mirror reverses physical left/right, the overlay explicitly distinguishes `image-left` from
`physical-right`.

### 3. Enable real pointer movement

```bash
deskpad run --enable-pointer
```

For the current mirror, horizontal motion is inverted and vertical motion is not. Override or tune
the mapping when needed:

```bash
deskpad run --enable-pointer --gain 1.5 --max-jump 70
deskpad run --enable-pointer --no-invert-x --invert-y
```

Runtime keys:

- `q` / `Esc`: quit
- `p`: pause or resume real pointer output
- `x`, `y`: toggle the corresponding pointer axis
- `[`, `]`: adjust the lower mirror crop line

Alternative movement gates are preserved:

```bash
deskpad run --control-mode pinch --enable-pointer
deskpad run --control-mode contact
```

## Experimental contact training

The camera cannot universally infer millimeter-scale contact from one RGB frame. Training is kept
personalized to the current camera, mirror, hand, surface, and lighting. This milestone uses the
right index finger on **both** physical sides.

Collect one label at a time; each command guides physical-right and then physical-left:

```bash
deskpad collect-air
deskpad collect-touch
deskpad collect-slide
deskpad collect-lift
```

| Label | Required action | Training role |
| --- | --- | --- |
| AIR | Hover 1–3 cm above the surface and move | Negative |
| TOUCH | Lightly touch and remain still | Positive |
| SLIDE | Stay in contact while sliding | Positive |
| LIFT | Start touching, then lift about 1 cm | Negative/release |

Each side gives 6 seconds to prepare and records 5 seconds of valid frames. The clock pauses when
the hand, geometry, or requested side is invalid. During LIFT, the original side is locked after
recording starts, so normal perspective drift outside the colored polygon does not discard the
release.

Commands can be repeated to add examples. Training starts automatically once all four labels have
enough data on both sides, or can be rerun with `deskpad train-contact`. Selective and full resets
archive rather than permanently delete data:

```bash
deskpad reset-contact --label lift
deskpad reset-contact
```

### What is trained

- Features combine hand pose, relative landmark depth, joint angles, plane motion, fingertip
  texture/shadow, and difference from the hand-free reference image.
- Physical-left and physical-right logistic models are fitted separately because the same right
  hand has different pose distributions across the laptop.
- Adjacent frames are split in temporal blocks, avoiding unrealistically easy random-frame
  validation leakage.
- Threshold selection prioritizes low false-contact rate. Models are rejected when validation
  false positives exceed 8% or recall falls below 45%.
- Runtime probabilities pass through multi-frame confirmation, asymmetric hysteresis, and bounded
  off-zone release grace before becoming AIR/TOUCH/RELEASE.

For future training, record several short sessions under different lighting, finger heights,
speeds, and wrist angles. Add hard negatives near boundaries, keep calibration/session identity,
and evaluate on entire unseen sessions. A future sequence/depth model should replace the current
model only if held-out false contacts improve—not merely because its training accuracy is higher.

Contact-driven movement, click, and drag remain behind explicit flags:

```bash
deskpad run --control-mode contact --enable-pointer
deskpad run --control-mode contact --enable-pointer --enable-click
deskpad run --control-mode contact --enable-pointer --enable-click --enable-drag
```

## Safety and local files

Pointer output stops when alignment is untrusted, the hand is lost, the fingertip leaves both
zones, or a frame exceeds the jump limit. Zone changes re-anchor motion. Cancelling a live drag
always sends mouse-up so macOS cannot remain stuck in a held-button state.

Profiles, samples, models, archives, previews, and reports stay under `~/.deskpad-vision`. Samples
are tied to the calibration reference hash so incompatible geometries are not mixed.

If camera capture is denied, enable the terminal app under **System Settings → Privacy & Security
→ Camera**, then fully restart it. Real pointer control additionally requires the terminal app
under **Accessibility**.

## Tests

```bash
python -m unittest discover -s tests -v
ruff check src tests
```

## Current limits and roadmap

- V2 expects both outer chassis sides and the front edge to be visible.
- Moving the computer or mirror after calibration intentionally freezes output and requires V2
  recalibration.
- One right hand is supported; a left hand needs a separate dataset/model.
- Dark reflective surfaces and glare remain difficult for monocular contact inference.
- Next: collect session-separated hard negatives, compare a small temporal/depth model, tune live
  pointer gain, then promote click/drag only after measured false-contact tests pass.
