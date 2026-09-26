import unittest

from deskpad_vision.gesture import PinchLatch, TouchMouseGesture
from deskpad_vision.pointer import NullPointerSink


class PinchLatchTests(unittest.TestCase):
    def test_uses_confirmation_and_hysteresis(self) -> None:
        latch = PinchLatch(on_threshold=0.3, off_threshold=0.5, confirmation_frames=2)
        self.assertFalse(latch.update(0.2))
        self.assertTrue(latch.update(0.2))
        self.assertTrue(latch.update(0.4))
        self.assertTrue(latch.update(0.6))
        self.assertFalse(latch.update(0.6))


class TouchMouseGestureTests(unittest.TestCase):
    def test_short_stationary_contact_becomes_click(self) -> None:
        sink = NullPointerSink()
        gesture = TouchMouseGesture()
        self.assertEqual(
            gesture.update(1.0, (100.0, 100.0), "left", True, False, True, sink, True),
            "TOUCH START",
        )
        self.assertIsNone(
            gesture.update(1.1, (104.0, 102.0), "left", True, False, True, sink, True)
        )
        self.assertEqual(
            gesture.update(1.2, (104.0, 102.0), "left", False, True, True, sink, True),
            "TAP",
        )
        self.assertEqual(sink.events, ["click"])

    def test_slide_does_not_become_click(self) -> None:
        sink = NullPointerSink()
        gesture = TouchMouseGesture(tap_max_travel=20.0)
        gesture.update(1.0, (0.0, 0.0), "left", True, False, True, sink, True)
        gesture.update(1.1, (30.0, 0.0), "left", True, False, True, sink, True)
        self.assertIsNone(
            gesture.update(1.2, (30.0, 0.0), "left", False, True, True, sink, True)
        )
        self.assertEqual(sink.events, [])

    def test_hold_then_move_uses_drag_events(self) -> None:
        sink = NullPointerSink()
        gesture = TouchMouseGesture(drag_hold_seconds=0.5)
        gesture.update(1.0, (50.0, 50.0), "right", True, False, True, sink, True, True)
        self.assertEqual(
            gesture.update(1.6, (51.0, 50.0), "right", True, False, True, sink, True, True),
            "DRAG START",
        )
        self.assertEqual(sink.events, ["down"])
        self.assertEqual(
            gesture.update(1.8, (60.0, 50.0), "right", False, True, True, sink, True, True),
            "DRAG END",
        )
        self.assertEqual(sink.events, ["down", "up"])

    def test_invalid_geometry_cancels_drag_and_releases_button(self) -> None:
        sink = NullPointerSink()
        gesture = TouchMouseGesture(drag_hold_seconds=0.1)
        gesture.update(1.0, (50.0, 50.0), "right", True, False, True, sink, True, True)
        gesture.update(1.2, (50.0, 50.0), "right", True, False, True, sink, True, True)

        self.assertEqual(
            gesture.update(1.3, None, None, False, False, False, sink, True, True),
            "DRAG CANCEL",
        )
        self.assertEqual(sink.events, ["down", "up"])


if __name__ == "__main__":
    unittest.main()
