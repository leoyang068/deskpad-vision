import unittest

from deskpad_vision.cli import build_parser


class RunDirectionDefaultsTests(unittest.TestCase):
    def test_current_mirror_setup_inverts_horizontal_axis_only(self) -> None:
        args = build_parser().parse_args(["run"])

        self.assertTrue(args.invert_x)
        self.assertFalse(args.invert_y)
        self.assertEqual(args.control_mode, "index")

    def test_axis_inversion_and_legacy_control_modes_remain_selectable(self) -> None:
        args = build_parser().parse_args(
            ["run", "--no-invert-x", "--invert-y", "--control-mode", "pinch"]
        )

        self.assertFalse(args.invert_x)
        self.assertTrue(args.invert_y)
        self.assertEqual(args.control_mode, "pinch")


if __name__ == "__main__":
    unittest.main()
