from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class ModelRuntime:
    python: Path
    deeplsd_root: Path
    weights: Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def find_model_runtime(root: Path | None = None) -> ModelRuntime:
    root = root or project_root()
    python_override = os.environ.get("DESKPAD_CALIBRATION_PYTHON")
    weights_override = os.environ.get("DESKPAD_DEEPLSD_WEIGHTS")
    python_candidates = [
        Path(python_override) if python_override else None,
        root / ".calibration-venv310/bin/python",
        root / ".calibration-venv/bin/python",
    ]
    python = next((path for path in python_candidates if path and path.exists()), None)
    deeplsd_root = root / ".models/DeepLSD"
    weights = (
        Path(weights_override)
        if weights_override
        else root / ".model-cache/deeplsd/deeplsd_md.tar"
    )
    missing = []
    if python is None:
        missing.append("独立 Python 环境")
    if not (deeplsd_root / "deeplsd").is_dir():
        missing.append("DeepLSD 源码")
    if not weights.is_file():
        missing.append("DeepLSD 权重")
    if missing:
        joined = "、".join(missing)
        raise RuntimeError(
            f"校准 V2 缺少{joined}。请先运行 scripts/setup-calibration-v2.sh"
        )
    assert python is not None
    return ModelRuntime(python, deeplsd_root, weights)


def detect_lines_with_model(
    frames: Sequence[np.ndarray], runtime: ModelRuntime | None = None
) -> Tuple[np.ndarray, ...]:
    """Run DeepLSD in its isolated Python environment and return one array per frame."""

    if not frames:
        raise ValueError("at least one frame is required")
    runtime = runtime or find_model_runtime()
    root = project_root()
    with tempfile.TemporaryDirectory(prefix="deskpad-calibration-v2-") as raw_directory:
        directory = Path(raw_directory)
        input_paths = []
        for index, frame in enumerate(frames):
            path = directory / f"frame-{index:02d}.png"
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(f"无法写入模型输入：{path}")
            input_paths.append(path)
        output_path = directory / "lines.npz"
        command = [
            str(runtime.python),
            "-m",
            "deskpad_vision.calibration_v2.model_runner",
            "--weights",
            str(runtime.weights),
            "--output",
            str(output_path),
            *[str(path) for path in input_paths],
        ]
        environment = os.environ.copy()
        python_path = [str(root / "src"), str(runtime.deeplsd_root)]
        if environment.get("PYTHONPATH"):
            python_path.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(python_path)
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=max(120, 60 * len(frames)),
            check=False,
        )
        if completed.returncode != 0:
            details = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"强线段模型运行失败：{details}")
        if not output_path.is_file():
            raise RuntimeError("强线段模型没有生成输出")
        with np.load(output_path, allow_pickle=False) as archive:
            return tuple(
                np.asarray(archive[f"lines_{index}"], dtype=np.float32)
                for index in range(len(frames))
            )


def infer_deeplsd(images: Sequence[Path], weights: Path) -> Tuple[np.ndarray, ...]:
    """Heavy inference implementation; imported only by the isolated runner."""

    import torch

    # DeepLSD's inference-only path never calls Ceres, but upstream imports its
    # optional extension unconditionally. A disabled shim keeps the lightweight
    # official quickstart installation usable on Apple Silicon.
    refinement = types.ModuleType("line_refinement")

    def refinement_disabled(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("optional Ceres refinement is disabled")

    refinement.line_optim = refinement_disabled  # type: ignore[attr-defined]
    sys.modules.setdefault("line_refinement", refinement)
    from deeplsd.models.deeplsd import DeepLSD

    configuration = {
        "detect_lines": True,
        "line_detection_params": {
            "merge": True,
            "filtering": True,
            "grad_thresh": 2,
            "grad_nfa": False,
            "optimize": False,
        },
    }
    network = DeepLSD(configuration)
    try:
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    except TypeError:  # torch versions before weights_only was added
        checkpoint = torch.load(weights, map_location="cpu")
    network.load_state_dict(checkpoint["model"])
    network.eval()

    outputs = []
    for path in images:
        frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise RuntimeError(f"无法读取模型输入：{path}")
        original_height, original_width = frame.shape
        padded_height = int(math.ceil(original_height / 32.0) * 32)
        padded_width = int(math.ceil(original_width / 32.0) * 32)
        padded = cv2.copyMakeBorder(
            frame,
            0,
            padded_height - original_height,
            0,
            padded_width - original_width,
            cv2.BORDER_REFLECT_101,
        )
        tensor = torch.tensor(padded, dtype=torch.float32)[None, None] / 255.0
        with torch.no_grad():
            prediction = network({"image": tensor})
        lines = np.asarray(prediction["lines"][0], dtype=np.float32)
        midpoints = np.mean(lines, axis=1)
        valid = (
            (midpoints[:, 0] >= 0)
            & (midpoints[:, 0] < original_width)
            & (midpoints[:, 1] >= 0)
            & (midpoints[:, 1] < original_height)
        )
        outputs.append(lines[valid])
    return tuple(outputs)
