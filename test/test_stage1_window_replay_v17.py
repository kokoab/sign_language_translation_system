import unittest
from scripts import replay_revisable_transcription_v17 as replay


class RecordingContractTests(unittest.TestCase):
    def test_timestamp_mapping_preserves_gaps_and_excludes_damaged_tail(self):
        self.assertTrue(hasattr(replay, 'recording_times'))
        self.assertEqual(replay.recording_times([.1, 12.6, 12.7, 12.8], 3, 15, 1), [.1, 12.6])
        for times in ([0, 0, 1], [0, float('nan'), 1], [0, 1]):
            with self.assertRaises(ValueError):
                replay.recording_times(times, 3, 15, 0)

    def test_manifest_rejects_protected_or_unverified_truth(self):
        self.assertTrue(hasattr(replay, 'validate_recording'))
        for row in (
            dict(item_id='bad', video='data/test/a.mp4', role='validation'),
            dict(item_id='bad', video='a.mp4', role='diagnostic', reference=['HUNGRY']),
            dict(item_id='../bad', video='a.mp4', role='validation'),
        ):
            with self.assertRaises(ValueError):
                replay.validate_recording(row)

    def test_phase_shift_keeps_all_frames_and_reports_sparse_windows(self):
        self.assertTrue(hasattr(replay, 'phase_windows'))
        times = [0, .2, .4, 1.2, 1.4, 2.1]
        windows = replay.phase_windows(times, 1., .5)
        self.assertEqual(windows, [[0, 1, 2], [3, 4], [5]])


if __name__ == '__main__':
    unittest.main()
