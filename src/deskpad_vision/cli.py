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
    calibration_mode = calibrate_parser.add_mutually_exclusive_group()
    calibration_mode.add_argument(
        "--auto",
        action="store_true",
        help="Calibrate without opening an interactive window.",
    )
    calibration_mode.add_argument(
        "--browser",
        action="store_true",
        help="Select the perspective plane in a local browser page (the default).",
    )
    calibration_mode.add_argument(
        "--opencv",
        action="store_true",
        help="Use the legacy OpenCV window calibration.",
    )
    calibration_mode.add_argument(
        "--v2",
        action="store_true",
        help="Use learned lines plus constrained physical-plane geometry.",
    )
    calibrate_parser.add_argument(
        "--image",
        type=Path,
        help="Run V2 on an existing image instead of capturing the camera.",
    )
    calibrate_parser.add_argument(
        "--crop-y",
        type=int,
        help="Override the lower mirror crop line for V2.",
    )

    run_parser = subparsers.add_parser("run", help="Run tracking and pointer preview")
    run_parser.add_argument("--camera", type=int, default=0)
    run_parser.add_argument(
        "--enable-pointer",
        action="store_true",
        help="Actually move the macOS pointer. Preview is the safe default.",
    )
    run_parser.add_argument("--gain", type=float, default=1.5)
    run_parser.add_argument("--max-jump", type=float, default=70.0)
    run_parser.add_argument(
        "--invert-x",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Invert horizontal pointer motion for the mirror view (default: enabled).",
    )
    run_parser.add_argument(
        "--invert-y",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Invert vertical pointer motion (default: disabled).",
    )
    run_parser.add_argument(
        "--control-mode",
        choices=("index", "pinch", "contact"),
        default="index",
        help="Control by index presence, pinch, or trained surface contact.",
    )
    run_parser.add_argument(
        "--enable-click",
        action="store_true",
        help="Send real left clicks for short contact taps (requires --enable-pointer).",
    )
    run_parser.add_argument(
        "--enable-drag",
        action="store_true",
        help="Send real hold-to-drag events (requires --enable-pointer).",
    )

    contact_parser = subparsers.add_parser(
        "collect-contact",
        help="Run the full guided contact collection sequence",
    )
    contact_parser.add_argument("--camera", type=int, default=0)
    contact_parser.add_argument(
        "--no-train",
        action="store_true",
        help="Only save samples; do not fit the contact model after capture.",
    )
    contact_mode = contact_parser.add_mutually_exclusive_group()
    contact_mode.add_argument(
        "--manual",
        action="store_true",
        help="Use number-key labels instead of the default automatic guided sequence.",
    )
    contact_mode.add_argument(
        "--redo-stage",
        type=int,
        choices=range(1, 9),
        metavar="1-8",
        help="Record only one guided step and replace its earlier samples.",
    )
    contact_parser.add_argument("--prepare-seconds", type=float, default=6.0)
    contact_parser.add_argument("--record-seconds", type=float, default=5.0)

    action_help = {
        "air": "Collect only hovering-without-contact examples",
        "touch": "Collect only stationary surface-touch examples",
        "slide": "Collect only sliding-while-touching examples",
        "lift": "Collect only lift-off-from-surface examples",
    }
    for label, help_text in action_help.items():
        action_parser = subparsers.add_parser(f"collect-{label}", help=help_text)
        action_parser.set_defaults(collection_label=label)
        action_parser.add_argument("--camera", type=int, default=0)
        action_parser.add_argument(
            "--no-train",
            action="store_true",
            help="Only save samples; do not fit the model after capture.",
        )
        action_parser.add_argument("--prepare-seconds", type=float, default=6.0)
        action_parser.add_argument("--record-seconds", type=float, default=5.0)

    subparsers.add_parser(
        "train-contact",
        help="Refit the local contact model from all samples for the current calibration",
    )
    reset_contact_parser = subparsers.add_parser(
        "reset-contact",
        help="Archive and clear all current contact samples and models",
    )
    reset_contact_parser.add_argument(
        "--label",
        choices=("air", "touch", "slide", "lift"),
        help="Archive only this action's sessions while preserving the others.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "calibrate":
            if args.v2:
                from .calibration_v2.workflow import calibrate_v2

                calibrate_v2(
                    profile_dir=args.profile_dir,
                    camera_index=args.camera,
                    width=args.width,
                    height=args.height,
                    image_path=args.image,
                    crop_y=args.crop_y,
                )
            else:
                use_browser = args.browser or not (args.auto or args.opencv)
                calibrate(
                    profile_dir=args.profile_dir,
                    camera_index=args.camera,
                    width=args.width,
                    height=args.height,
                    automatic=args.auto,
                    browser=use_browser,
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
                control_mode=args.control_mode,
                enable_click=args.enable_click,
                enable_drag=args.enable_drag,
            )
        elif args.command == "collect-contact":
            from .contact_capture import collect_contact_data

            collect_contact_data(
                profile_dir=args.profile_dir,
                camera_index=args.camera,
                auto_train=not args.no_train,
                guided=not args.manual,
                prepare_seconds=args.prepare_seconds,
                record_seconds=args.record_seconds,
                redo_stage=args.redo_stage,
                label_filter=None,
            )
        elif hasattr(args, "collection_label"):
            from .contact_capture import collect_contact_data

            collect_contact_data(
                profile_dir=args.profile_dir,
                camera_index=args.camera,
                auto_train=not args.no_train,
                guided=True,
                prepare_seconds=args.prepare_seconds,
                record_seconds=args.record_seconds,
                label_filter=args.collection_label,
            )
        elif args.command == "train-contact":
            from .contact_capture import (
                fit_contact_model_for_profile,
                print_contact_training_report,
            )

            report = fit_contact_model_for_profile(args.profile_dir)
            print_contact_training_report(report)
        elif args.command == "reset-contact":
            from .contact_capture import archive_contact_data

            archive_dir = archive_contact_data(args.profile_dir, label=args.label)
            if archive_dir is None:
                print("No contact data or model exists; nothing to reset.")
            else:
                scope = "Contact data" if args.label is None else f"{args.label.upper()} data"
                print(f"{scope} cleared and archived at: {archive_dir}")
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"deskpad: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
