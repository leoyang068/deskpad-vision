import unittest

from deskpad_vision.pointer import NullPointerSink, RelativePointerMapper


class PointerMapperTests(unittest.TestCase):
    def test_requires_clutch_and_reanchors_on_zone_change(self) -> None:
        sink = NullPointerSink()
        mapper = RelativePointerMapper(sink=sink, gain=2.0, smoothing_alpha=1.0)
        self.assertIsNone(mapper.update((10, 10), "left", clutch=False))
        self.assertEqual(mapper.update((10, 10), "left", clutch=True), (0.0, 0.0))
        self.assertEqual(mapper.update((13, 8), "left", clutch=True), (6.0, -4.0))
        self.assertEqual(mapper.update((500, 500), "right", clutch=True), (0.0, 0.0))

    def test_rejects_large_jump(self) -> None:
        sink = NullPointerSink()
        mapper = RelativePointerMapper(sink=sink, max_jump=20.0, smoothing_alpha=1.0)
        mapper.update((10, 10), "right", clutch=True)
        self.assertIsNone(mapper.update((100, 100), "right", clutch=True))


if __name__ == "__main__":
    unittest.main()
