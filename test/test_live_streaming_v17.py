import unittest

from scripts.live_streaming_v17 import StreamingStabilizer, parser


def result(gloss: str, accepted: bool = True) -> dict[str, object]:
    return {"gloss": gloss, "accepted": accepted}


class StreamingStabilizerTests(unittest.TestCase):
    def test_emits_once_after_two_agreeing_windows(self):
        value = StreamingStabilizer(required_hits=2, release_hits=2)
        self.assertIsNone(value.update(result("YOU")))
        self.assertEqual(value.update(result("YOU")), "YOU")
        self.assertIsNone(value.update(result("YOU")))

    def test_new_stable_label_emits_without_neutral_gap(self):
        value = StreamingStabilizer(required_hits=2, release_hits=2)
        value.update(result("YOU"))
        value.update(result("YOU"))
        self.assertIsNone(value.update(result("NEED")))
        self.assertEqual(value.update(result("NEED")), "NEED")

    def test_two_rejections_rearm_same_label(self):
        value = StreamingStabilizer(required_hits=2, release_hits=2)
        value.update(result("SICK"))
        self.assertEqual(value.update(result("SICK")), "SICK")
        value.update(result("UNKNOWN", False))
        value.update(result("UNKNOWN", False))
        self.assertIsNone(value.update(result("SICK")))
        self.assertEqual(value.update(result("SICK")), "SICK")

    def test_stream_defaults_are_separate_and_low_latency(self):
        args = parser().parse_args([])
        self.assertEqual(args.mode, "cascade")
        self.assertEqual(args.window_seconds, 1.2)
        self.assertEqual(args.stride_seconds, 0.2)
        self.assertIn("live_streaming_v17", str(args.output_root))


if __name__ == "__main__":
    unittest.main()
