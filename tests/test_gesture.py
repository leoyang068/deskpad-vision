import unittest

from deskpad_vision.gesture import PinchLatch


class PinchLatchTests(unittest.TestCase):
    def test_uses_confirmation_and_hysteresis(self) -> None:
        latch = PinchLatch(on_threshold=0.3, off_threshold=0.5, confirmation_frames=2)
        self.assertFalse(latch.update(0.2))
        self.assertTrue(latch.update(0.2))
        self.assertTrue(latch.update(0.4))
        self.assertTrue(latch.update(0.6))
        self.assertFalse(latch.update(0.6))


if __name__ == "__main__":
    unittest.main()
