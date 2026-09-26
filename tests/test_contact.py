from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from deskpad_vision.cli import build_parser
from deskpad_vision.contact import (
    CONTACT_FEATURE_NAMES,
    ContactCollectionStage,
    ContactLatch,
    ContactState,
    ContactZoneGrace,
    GuidedContactProtocol,
    LogisticContactModel,
    ZonedContactModel,
    physical_side_for_image_zone,
    samples_to_training_arrays,
    train_contact_model,
)
from deskpad_vision.contact_capture import (
    _effective_lift_image_zone,
    _guided_action_lines,
    _load_training_samples,
    _matching_sample_paths,
    _missing_training_groups,
    archive_contact_data,
)


class ContactTests(unittest.TestCase):
    def test_top_level_collection_commands_lock_one_distinct_label(self) -> None:
        parser = build_parser()
        for label in ("air", "touch", "slide", "lift"):
            args = parser.parse_args([f"collect-{label}"])
            self.assertEqual(args.command, f"collect-{label}")
            self.assertEqual(args.collection_label, label)

    def test_physical_side_is_explicitly_mirrored_from_image_zone(self) -> None:
        self.assertEqual(physical_side_for_image_zone("left"), "right")
        self.assertEqual(physical_side_for_image_zone("right"), "left")
        self.assertIsNone(physical_side_for_image_zone(None))

    def test_contact_latch_rejects_short_probability_spike(self) -> None:
        latch = ContactLatch(on_threshold=0.75, off_threshold=0.40, touch_frames=4)

        states = [latch.update(value) for value in (0.1, 0.9, 0.9, 0.2, 0.95, 0.2)]

        self.assertEqual(states, [ContactState.AIR] * len(states))
        self.assertFalse(latch.active)

    def test_contact_latch_confirms_touch_and_release_across_multiple_frames(self) -> None:
        latch = ContactLatch(
            on_threshold=0.70,
            off_threshold=0.35,
            touch_frames=3,
            release_frames=2,
            release_display_frames=2,
        )

        self.assertEqual(
            [latch.update(0.9) for _ in range(3)],
            [ContactState.AIR, ContactState.AIR, ContactState.TOUCH],
        )
        self.assertEqual(latch.update(0.5), ContactState.TOUCH)
        self.assertEqual(latch.update(0.2), ContactState.TOUCH)
        self.assertEqual(latch.update(0.2), ContactState.RELEASE)
        self.assertEqual(latch.update(0.1), ContactState.RELEASE)
        self.assertEqual(latch.update(0.1), ContactState.AIR)

    def test_contact_latch_invalid_geometry_immediately_returns_to_air(self) -> None:
        latch = ContactLatch(touch_frames=1)
        self.assertEqual(latch.update(0.99), ContactState.TOUCH)
        self.assertEqual(latch.update(0.99, valid=False), ContactState.AIR)
        self.assertFalse(latch.active)

    def test_contact_zone_grace_keeps_side_during_short_lift(self) -> None:
        grace = ContactZoneGrace(max_seconds=0.5, max_plane_distance=100.0)
        self.assertEqual(grace.update(1.0, "left", (100.0, 100.0), False), "left")
        self.assertEqual(grace.update(1.2, None, (140.0, 100.0), True), "left")
        self.assertIsNone(grace.update(1.3, None, (240.0, 100.0), True))

    def test_lift_collection_accepts_bounded_off_zone_parallax(self) -> None:
        instruction = GuidedContactProtocol(
            prepare_seconds=0.0,
            stages=(ContactCollectionStage("right", "lift"),),
        )
        instruction.update(1.0, valid_sample=False)
        status = instruction.status(1.1)
        zone, extended = _effective_lift_image_zone(
            status,
            None,
            (320.0, 250.0),
            np.asarray((100.0, 100.0), dtype=np.float64),
        )
        self.assertEqual(zone, "left")
        self.assertTrue(extended)
        far_zone, far_extended = _effective_lift_image_zone(
            status,
            None,
            (900.0, 900.0),
            np.asarray((100.0, 100.0), dtype=np.float64),
        )
        self.assertIsNone(far_zone)
        self.assertFalse(far_extended)

    def test_guided_collection_uses_countdown_and_pauses_on_invalid_frames(self) -> None:
        protocol = GuidedContactProtocol(
            prepare_seconds=1.0,
            record_seconds=0.2,
            stages=(ContactCollectionStage("right", "air"),),
        )

        self.assertEqual(protocol.status(10.0).mode, "prepare")
        self.assertAlmostEqual(protocol.status(10.4).remaining_seconds, 0.6)
        self.assertEqual(protocol.update(11.0, valid_sample=False).mode, "record")
        self.assertAlmostEqual(protocol.update(11.1, valid_sample=False).remaining_seconds, 0.2)
        self.assertGreater(protocol.update(11.2, valid_sample=True).remaining_seconds, 0.0)
        self.assertEqual(protocol.update(11.31, valid_sample=True).mode, "done")

    def test_training_fits_separable_examples_and_model_round_trips(self) -> None:
        rng = np.random.default_rng(4)
        feature_count = len(CONTACT_FEATURE_NAMES)
        negative = rng.normal(0.0, 0.18, size=(90, feature_count))
        positive = rng.normal(0.0, 0.18, size=(90, feature_count))
        negative[:, :4] -= 1.0
        positive[:, :4] += 1.0
        vectors = np.vstack([negative, positive])
        labels = np.concatenate([np.zeros(len(negative)), np.ones(len(positive))])

        model, report = train_contact_model(vectors, labels)

        self.assertLessEqual(report.false_positive_rate, 0.02)
        self.assertGreaterEqual(report.recall, 0.95)
        self.assertGreaterEqual(report.accuracy, 0.95)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contact-model.json"
            model.save(path)
            loaded = LogisticContactModel.load(path)
        self.assertEqual(loaded.feature_names, CONTACT_FEATURE_NAMES)
        self.assertEqual(loaded.physical_hand, "right")
        self.assertEqual(loaded.threshold, model.threshold)
        np.testing.assert_allclose(loaded.weights, model.weights)

    def test_zoned_contact_model_round_trip(self) -> None:
        feature_count = len(CONTACT_FEATURE_NAMES)
        zone_model = LogisticContactModel(
            feature_names=CONTACT_FEATURE_NAMES,
            mean=(0.0,) * feature_count,
            scale=(1.0,) * feature_count,
            weights=(0.1,) * feature_count,
            bias=0.0,
            threshold=0.8,
            reference_sha256="reference",
        )
        model = ZonedContactModel(
            zone_models={"left": zone_model, "right": zone_model},
            reference_sha256="reference",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zoned-model.json"
            model.save(path)
            loaded = ZonedContactModel.load(path)

        self.assertEqual(set(loaded.zone_models), {"left", "right"})
        self.assertEqual(loaded.zone_models["left"].threshold, 0.8)
        self.assertEqual(loaded.physical_hand, "right")

    def test_sample_labels_map_touch_and_slide_to_positive(self) -> None:
        base = {name: 0.25 for name in CONTACT_FEATURE_NAMES}
        samples = [
            {**base, "label": "air"},
            {**base, "label": "lift"},
            {**base, "label": "touch"},
            {**base, "label": "slide"},
            {**base, "label": "paused"},
        ]

        vectors, labels = samples_to_training_arrays(samples)

        self.assertEqual(vectors.shape, (4, len(CONTACT_FEATURE_NAMES)))
        self.assertEqual(labels.tolist(), [0.0, 0.0, 1.0, 1.0])

    def test_incomplete_contact_session_is_never_used_for_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile_dir = Path(directory)
            session_dir = profile_dir / "contact-data" / "unfinished"
            session_dir.mkdir(parents=True)
            (session_dir / "metadata.json").write_text(
                json.dumps({"reference_sha256": "current", "complete": False}),
                encoding="utf-8",
            )
            (session_dir / "samples.jsonl").write_text("{}\n", encoding="utf-8")

            self.assertEqual(_matching_sample_paths(profile_dir, "current"), [])

    def test_correction_session_replaces_only_matching_side_and_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "01"
            correction = root / "02"
            first.mkdir()
            correction.mkdir()
            (first / "metadata.json").write_text(
                json.dumps({"complete": True, "replaces": []}),
                encoding="utf-8",
            )
            (first / "samples.jsonl").write_text(
                "\n".join(
                    (
                        json.dumps({"physical_zone": "left", "label": "lift", "value": "bad"}),
                        json.dumps({"physical_zone": "left", "label": "air", "value": "keep"}),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            (correction / "metadata.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "replaces": [{"physical_zone": "left", "label": "lift"}],
                    }
                ),
                encoding="utf-8",
            )
            (correction / "samples.jsonl").write_text(
                json.dumps({"physical_zone": "left", "label": "lift", "value": "good"})
                + "\n",
                encoding="utf-8",
            )

            samples = _load_training_samples(
                [first / "samples.jsonl", correction / "samples.jsonl"]
            )

        self.assertEqual([sample["value"] for sample in samples], ["keep", "good"])

    def test_guided_collection_uses_right_hand_on_both_physical_sides(self) -> None:
        left_status = GuidedContactProtocol(
            stages=(ContactCollectionStage("left", "touch"),)
        ).status(1.0)
        right_status = GuidedContactProtocol(
            stages=(ContactCollectionStage("right", "touch"),)
        ).status(1.0)

        self.assertIn("始终使用右手食指", _guided_action_lines(left_status)[0])
        self.assertIn("始终使用右手食指", _guided_action_lines(right_status)[0])

    def test_training_requires_every_label_on_both_sides(self) -> None:
        samples = [
            {"physical_zone": side, "label": label}
            for side in ("right", "left")
            for label in ("air", "touch", "slide", "lift")
            for _ in range(25)
        ]
        self.assertEqual(_missing_training_groups(samples), [])
        incomplete = [sample for sample in samples if sample["label"] != "slide"]
        self.assertEqual(
            _missing_training_groups(incomplete),
            [("right", "slide"), ("left", "slide")],
        )

    def test_contact_reset_moves_data_and_model_to_recoverable_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile_dir = Path(directory)
            data_dir = profile_dir / "contact-data"
            data_dir.mkdir()
            (data_dir / "sample.txt").write_text("data", encoding="utf-8")
            (profile_dir / "contact-model.json").write_text("{}", encoding="utf-8")

            archive_dir = archive_contact_data(profile_dir)

            self.assertIsNotNone(archive_dir)
            assert archive_dir is not None
            self.assertFalse(data_dir.exists())
            self.assertFalse((profile_dir / "contact-model.json").exists())
            self.assertTrue((archive_dir / "contact-data" / "sample.txt").exists())
            self.assertTrue((archive_dir / "contact-model.json").exists())

    def test_contact_reset_can_archive_only_one_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile_dir = Path(directory)
            for label in ("air", "lift"):
                session_dir = profile_dir / "contact-data" / label
                session_dir.mkdir(parents=True)
                (session_dir / "metadata.json").write_text(
                    json.dumps({"requested_label": label, "counts": {label: 10}}),
                    encoding="utf-8",
                )
                (session_dir / "samples.jsonl").write_text("{}\n", encoding="utf-8")

            archive_dir = archive_contact_data(profile_dir, label="lift")

            self.assertIsNotNone(archive_dir)
            assert archive_dir is not None
            self.assertTrue((profile_dir / "contact-data" / "air").exists())
            self.assertFalse((profile_dir / "contact-data" / "lift").exists())
            self.assertTrue((archive_dir / "contact-data" / "lift").exists())


if __name__ == "__main__":
    unittest.main()
