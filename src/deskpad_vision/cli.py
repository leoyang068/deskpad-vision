from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .app import calibrate, run
from .config import default_profile_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deskpad",
        description="Camera-tracked virtual palm-rest surfaces for macOS.",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=default_profile_dir(),
        help="Profile directory (default: ~/.deskpad-vision)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate_parser = subparsers.add_parser("calibrate", help="Create a reference profile")
    calibrate_parser.add_argument("--camera", type=int, default=0)
    calibrate_parser.add_argument("--width", type=int, default=1280)
    calibrate_parser.add_argument("--height", type=int, default=720)

    run_parser = subparsers.add_parser("run", help="Run tracking and pointer preview")
    run_parser.add_argument("--camera", type=int, default=0)
    run_parser.add_argument(
        "--enable-pointer",
        action="store_true",
        help="Actually move the macOS pointer. Preview is the safe default.",
    )
    run_parser.add_argument("--gain", type=float, default=1.5)
    run_parser.add_argument("--max-jump", type=float, default=70.0)
    run_parser.add_argument("--invert-x", action="store_true")
    run_parser.add_argument("--invert-y", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "calibrate":
            calibrate(
                profile_dir=args.profile_dir,
                camera_index=args.camera,
                width=args.width,
                height=args.height,
            )
        elif args.command == "run":
            run(
                profile_dir=args.profile_dir,
                camera_index=args.camera,
                enable_pointer=args.enable_pointer,
                gain=args.gain,
                max_jump=args.max_jump,
                invert_x=args.invert_x,
                invert_y=args.invert_y,
            )
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"deskpad: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
