import unittest

from scripts.live_motion_valley_v17 import parser


class MotionValleyDefaultsTests(unittest.TestCase):
    def test_separate_path_uses_low_motion_video_boundaries(self):
        args = parser().parse_args([])
        self.assertTrue(args.end_on_low_motion)
        self.assertTrue(args.segment_video)
        self.assertEqual(args.mode, "cascade")
        self.assertEqual(args.quiet_motion, 0.010)
        self.assertEqual(args.quiet_seconds, 0.12)
        self.assertIn("live_motion_valley_v17", str(args.output_root))


if __name__ == "__main__":
    unittest.main()
