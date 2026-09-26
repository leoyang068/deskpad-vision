from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .deep_lines import infer_deeplsd


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Isolated DeepLSD runner for calibration V2")
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args(argv)
    lines = infer_deeplsd(args.images, args.weights)
    np.savez_compressed(
        args.output,
        **{f"lines_{index}": value for index, value in enumerate(lines)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
