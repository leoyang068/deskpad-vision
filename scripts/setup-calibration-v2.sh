#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
UV_BIN=${UV_BIN:-uv}
PYTHON_VERSION=3.10
ENV_DIR="$ROOT/.calibration-venv310"
MODEL_DIR="$ROOT/.models/DeepLSD"
WEIGHT_DIR="$ROOT/.model-cache/deeplsd"
WEIGHT_FILE="$WEIGHT_DIR/deeplsd_md.tar"

command -v "$UV_BIN" >/dev/null 2>&1 || {
  echo "uv is required. Install it from https://docs.astral.sh/uv/" >&2
  exit 1
}

"$UV_BIN" python install "$PYTHON_VERSION"
"$UV_BIN" venv --python "$PYTHON_VERSION" "$ENV_DIR"
"$UV_BIN" pip install --python "$ENV_DIR/bin/python" \
  'torch>=2.5' torchvision 'numpy<2' 'opencv-python-headless<4.12' \
  scipy pyyaml omegaconf kornia scikit-image h5py shapely scikit-build cmake

if [ ! -d "$MODEL_DIR/.git" ]; then
  mkdir -p "$ROOT/.models"
  git clone --recursive https://github.com/cvg/DeepLSD.git "$MODEL_DIR"
fi

"$UV_BIN" pip install --python "$ENV_DIR/bin/python" \
  "$MODEL_DIR/third_party/pytlsd"
"$UV_BIN" pip install --python "$ENV_DIR/bin/python" --no-deps -e "$MODEL_DIR"

mkdir -p "$WEIGHT_DIR"
if [ ! -f "$WEIGHT_FILE" ]; then
  curl -L https://cvg-data.inf.ethz.ch/DeepLSD/deeplsd_md.tar -o "$WEIGHT_FILE"
fi

echo "Calibration V2 runtime is ready."
