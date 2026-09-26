from __future__ import annotations

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .config import Quad
from .deck_detection import DeckSideEdges

_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Deskpad Vision 平面标定</title>
  <style>
    :root { color-scheme: dark; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
    body { margin: 0; background: #111318; color: #f4f5f7; }
    main { width: min(1180px, calc(100% - 32px)); margin: 24px auto 40px; }
    h1 { font-size: 24px; margin: 0 0 8px; }
    p { color: #b8bec9; line-height: 1.5; margin: 6px 0; }
    strong { color: #f4f5f7; }
    .panel { background: #1b1f27; border: 1px solid #343a46; border-radius: 14px;
             padding: 16px; margin-top: 16px; }
    canvas { display: block; width: 100%; height: auto; border-radius: 9px;
             cursor: grab; touch-action: none; background: #000; }
    canvas.dragging { cursor: grabbing; }
    .controls { display: grid; grid-template-columns: 1fr auto auto; gap: 12px;
                align-items: center; margin-top: 14px; }
    input[type="range"] { width: 100%; accent-color: #ff453a; }
    button { border: 0; border-radius: 9px; padding: 11px 18px; font-size: 15px;
             font-weight: 650; color: white; background: #0a84ff; cursor: pointer; }
    button.secondary { background: #3a3f49; }
    button:disabled { opacity: .55; cursor: default; }
    #status { min-height: 24px; color: #71e38b; font-weight: 600; margin-top: 10px; }
    .legend { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 10px; font-size: 14px; }
    .swatch { display: inline-block; width: 12px; height: 12px; border-radius: 3px;
              margin-right: 5px; vertical-align: -1px; }
    @media (max-width: 680px) { .controls { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <h1>Deskpad Vision 平面标定</h1>
  <p><strong>紫色 T1–T4：</strong><span id="trackpadHint"></span>
    顺序为左上、右上、右下、左下；请逐角核对，可直接拖动。</p>
  <p><strong>青色虚线：</strong>图像中的透视参考线，只辅助判断方向，<strong>不等于电脑外框</strong>。</p>
  <p><strong>橙色 P1–P4：</strong>真实电脑底座可用平面的四角。请把左右边拖到比键盘更宽的
    机身外沿，上下边分别沿触控板的上下边方向。</p>
  <p>绿色和蓝色只会出现在橙色边界内。保存前必须目视确认紫色触控板和橙色机身边界。</p>
  <section class="panel">
    <canvas id="canvas"></canvas>
    <div class="legend">
      <span><i class="swatch" style="background:#bf5af2"></i>实体触控板四角</span>
      <span><i class="swatch" style="background:#ff9f0a"></i>可用平面外边界</span>
      <span><i class="swatch" style="background:#64d2ff"></i>透视方向参考</span>
      <span><i class="swatch" style="background:#30d158"></i>左侧扩展区</span>
      <span><i class="swatch" style="background:#0a84ff"></i>右侧扩展区</span>
      <span><i class="swatch" style="background:#ff453a"></i>裁切线</span>
    </div>
    <div class="controls">
      <label>裁切线高度：<strong id="cropValue"></strong>
        <input id="crop" type="range">
      </label>
      <button id="reset" class="secondary">恢复初始位置</button>
      <button id="save">保存平面标定</button>
    </div>
    <div id="status"></div>
  </section>
</main>
<script>
const config = __CONFIG__;
const canvas = document.querySelector('#canvas');
const ctx = canvas.getContext('2d');
const cropInput = document.querySelector('#crop');
const cropValue = document.querySelector('#cropValue');
const resetButton = document.querySelector('#reset');
const saveButton = document.querySelector('#save');
const status = document.querySelector('#status');
const trackpadHint = document.querySelector('#trackpadHint');
const image = new Image();
let cropY = config.cropY;
let trackpad = config.trackpadQuad.map(point => [...point]);
let surface = config.surfaceQuad.map(point => [...point]);
const sideEdges = config.sideEdges;
let activeHandle = null;

canvas.width = config.imageWidth;
canvas.height = config.imageHeight;
cropInput.min = 40;
cropInput.max = config.imageHeight - 10;
cropInput.value = cropY;

function canvasPoint(event) {
  const bounds = canvas.getBoundingClientRect();
  const scaledX = (event.clientX - bounds.left) * canvas.width / bounds.width;
  const scaledY = (event.clientY - bounds.top) * canvas.height / bounds.height;
  return [
    Math.max(0, Math.min(canvas.width - 1, scaledX)),
    Math.max(0, Math.min(canvas.height - 1, scaledY))
  ];
}

function polygon(points, fill, stroke, lineWidth = 3) {
  ctx.beginPath();
  points.forEach((point, index) => {
    if (index === 0) ctx.moveTo(point[0], point[1]);
    else ctx.lineTo(point[0], point[1]);
  });
  ctx.closePath();
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  ctx.strokeStyle = stroke;
  ctx.lineWidth = lineWidth;
  ctx.stroke();
}

function lerp(a, b, amount) {
  return [a[0] + (b[0] - a[0]) * amount, a[1] + (b[1] - a[1]) * amount];
}

function drawHandle(point, color, label) {
  ctx.beginPath();
  ctx.arc(point[0], point[1], 9, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.font = 'bold 16px -apple-system, sans-serif';
  ctx.lineWidth = 4;
  ctx.strokeStyle = 'rgba(0,0,0,.85)';
  ctx.strokeText(label, point[0] + 12, point[1] - 10);
  ctx.fillStyle = '#fff';
  ctx.fillText(label, point[0] + 12, point[1] - 10);
}

function zones() {
  const gapFraction = 0.045;
  const leftTop = lerp(trackpad[0], surface[0], gapFraction);
  const leftBottom = lerp(trackpad[3], surface[3], gapFraction);
  const rightTop = lerp(trackpad[1], surface[1], gapFraction);
  const rightBottom = lerp(trackpad[2], surface[2], gapFraction);
  return {
    left: [surface[0], leftTop, leftBottom, surface[3]],
    right: [rightTop, surface[1], surface[2], rightBottom]
  };
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  if (sideEdges) {
    ctx.save();
    ctx.setLineDash([12, 8]);
    ctx.strokeStyle = '#64d2ff';
    ctx.lineWidth = 3;
    [sideEdges.left, sideEdges.right].forEach(line => {
      ctx.beginPath();
      ctx.moveTo(line[0][0], line[0][1]);
      ctx.lineTo(line[1][0], line[1][1]);
      ctx.stroke();
    });
    ctx.restore();
  }
  const activeZones = zones();
  polygon(activeZones.left, 'rgba(48,209,88,.20)', '#30d158');
  polygon(activeZones.right, 'rgba(10,132,255,.20)', '#0a84ff');
  polygon(surface, 'rgba(255,159,10,.035)', '#ff9f0a', 2);
  polygon(trackpad, 'rgba(191,90,242,.14)', '#bf5af2');
  surface.forEach((point, index) => drawHandle(point, '#ff9f0a', `P${index + 1}`));
  trackpad.forEach((point, index) => drawHandle(point, '#bf5af2', `T${index + 1}`));
  ctx.fillStyle = 'rgba(0,0,0,.72)';
  ctx.fillRect(0, cropY, canvas.width, canvas.height - cropY);
  ctx.strokeStyle = '#ff453a';
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(0, cropY);
  ctx.lineTo(canvas.width, cropY);
  ctx.stroke();
  cropValue.textContent = `y=${Math.round(cropY)}`;
}

function closestHandle(point) {
  const bounds = canvas.getBoundingClientRect();
  const radius = 24 * canvas.width / bounds.width;
  let best = null;
  let bestDistance = radius;
  [['trackpad', trackpad], ['surface', surface]].forEach(([kind, points]) => {
    points.forEach((candidate, index) => {
      const distance = Math.hypot(point[0] - candidate[0], point[1] - candidate[1]);
      if (distance < bestDistance) {
        best = {kind, index};
        bestDistance = distance;
      }
    });
  });
  return best;
}

canvas.addEventListener('pointerdown', event => {
  activeHandle = closestHandle(canvasPoint(event));
  if (!activeHandle) {
    status.style.color = '#ff9f0a';
    status.textContent = '请拖动紫色或橙色圆点，而不是区域内部。';
    return;
  }
  canvas.setPointerCapture(event.pointerId);
  canvas.classList.add('dragging');
  status.textContent = '';
});

canvas.addEventListener('pointermove', event => {
  if (!activeHandle) return;
  const point = canvasPoint(event);
  point[1] = Math.min(point[1], cropY - 2);
  const target = activeHandle.kind === 'trackpad' ? trackpad : surface;
  target[activeHandle.index] = point.map(Math.round);
  draw();
});

canvas.addEventListener('pointerup', () => {
  activeHandle = null;
  canvas.classList.remove('dragging');
});

cropInput.addEventListener('input', () => {
  cropY = Number(cropInput.value);
  draw();
});

function showInitialStatus() {
  if (config.trackpadConfidence === null) {
    trackpadHint.textContent = '程序没有可靠识别出触控板，当前紫框只是初始候选，必须手动修正。';
    status.style.color = '#ff9f0a';
    status.textContent = '未可靠识别触控板：请先把四个紫点拖到真实触控板四角。';
  } else {
    trackpadHint.textContent =
      `程序根据真实边缘检测得到（置信度 ${config.trackpadConfidence.toFixed(2)}）。`;
    status.style.color = '#71e38b';
    status.textContent = '已检测触控板边缘；请核对紫色四角，并手动确认橙色机身边界。';
  }
}

resetButton.addEventListener('click', () => {
  cropY = config.cropY;
  trackpad = config.trackpadQuad.map(point => [...point]);
  surface = config.surfaceQuad.map(point => [...point]);
  cropInput.value = cropY;
  showInitialStatus();
  draw();
});

saveButton.addEventListener('click', async () => {
  saveButton.disabled = true;
  status.style.color = '#b8bec9';
  status.textContent = '正在验证平面并保存…';
  try {
    const response = await fetch('/save', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        cropY: Math.round(cropY),
        trackpadQuad: trackpad,
        surfaceQuad: surface
      })
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || '保存失败');
    status.style.color = '#71e38b';
    status.textContent = '平面标定已保存。可关闭页面，回到 Terminal 运行 deskpad run。';
  } catch (error) {
    saveButton.disabled = false;
    status.style.color = '#ff453a';
    status.textContent = error.message;
  }
});

image.onload = () => { draw(); showInitialStatus(); };
image.src = '/frame.jpg';
</script>
</body>
</html>
"""


def _parse_quad(raw: Any, name: str) -> Quad:
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError(f"{name} must have four corner points")
    points = []
    for raw_point in raw:
        if not isinstance(raw_point, list) or len(raw_point) != 2:
            raise ValueError(f"{name} contains an invalid point")
        points.append((float(raw_point[0]), float(raw_point[1])))
    return points[0], points[1], points[2], points[3]


def _validate_quad(quad: Quad, name: str, image_width: int, crop_y: int) -> np.ndarray:
    points = np.asarray(quad, dtype=np.float32)
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains a non-finite point")
    if np.any(points[:, 0] < 0) or np.any(points[:, 0] >= image_width):
        raise ValueError(f"{name} is outside the image width")
    if np.any(points[:, 1] < 0) or np.any(points[:, 1] >= crop_y):
        raise ValueError(f"{name} must stay above the crop line")
    if not cv2.isContourConvex(points):
        raise ValueError(f"{name} corners cross; keep the TL, TR, BR, BL order")
    if abs(float(cv2.contourArea(points))) < 1600.0:
        raise ValueError(f"{name} is too small")
    return points


def _validate_selection(
    payload: Dict[str, Any], image_width: int, image_height: int
) -> Tuple[int, Quad, Quad]:
    crop_y = int(payload["cropY"])
    if not 40 <= crop_y <= image_height - 10:
        raise ValueError("Crop line is outside the image")
    trackpad_quad = _parse_quad(payload["trackpadQuad"], "Trackpad")
    surface_quad = _parse_quad(payload["surfaceQuad"], "Surface")
    trackpad_points = _validate_quad(trackpad_quad, "Trackpad", image_width, crop_y)
    surface_points = _validate_quad(surface_quad, "Surface", image_width, crop_y)
    for point in trackpad_points:
        signed_distance = cv2.pointPolygonTest(surface_points, tuple(point), True)
        if signed_distance < -3.0:
            raise ValueError("The orange surface boundary must contain the purple trackpad")
    return crop_y, trackpad_quad, surface_quad


def select_geometry_in_browser(
    frame: np.ndarray,
    initial_crop_y: int,
    initial_trackpad_quad: Quad,
    initial_surface_quad: Quad,
    deck_edges: Optional[DeckSideEdges] = None,
    trackpad_confidence: Optional[float] = None,
) -> Tuple[int, Quad, Quad]:
    """Serve a one-shot localhost page for reliable perspective-plane calibration."""

    image_height, image_width = frame.shape[:2]
    encoded_ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not encoded_ok:
        raise RuntimeError("Could not encode the calibration frame")
    frame_bytes = encoded.tobytes()
    config = {
        "imageWidth": image_width,
        "imageHeight": image_height,
        "cropY": initial_crop_y,
        "trackpadQuad": initial_trackpad_quad,
        "surfaceQuad": initial_surface_quad,
        "trackpadConfidence": trackpad_confidence,
        "sideEdges": (
            {
                "left": deck_edges.left,
                "right": deck_edges.right,
                "confidence": deck_edges.confidence,
            }
            if deck_edges is not None
            else None
        ),
    }
    page_bytes = _PAGE.replace("__CONFIG__", json.dumps(config)).encode("utf-8")
    result: Dict[str, Optional[Tuple[int, Quad, Quad]]] = {"selection": None}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            return

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", page_bytes)
            elif self.path == "/frame.jpg":
                self._send(200, "image/jpeg", frame_bytes)
            else:
                self._send(404, "text/plain; charset=utf-8", b"Not found")

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/save":
                self._send(404, "application/json", b'{"error":"Not found"}')
                return
            try:
                length = min(int(self.headers.get("Content-Length", "0")), 8192)
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                selection = _validate_selection(payload, image_width, image_height)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self._send(400, "application/json", body)
                return
            self._send(200, "application/json", b'{"saved":true}')
            result["selection"] = selection

    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.timeout = 0.5
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Open this calibration page if the browser does not appear: {url}")
    webbrowser.open(url)
    try:
        while result["selection"] is None:
            server.handle_request()
    finally:
        server.server_close()

    selection = result["selection"]
    assert selection is not None
    return selection
