"""Minimal checks for buffered ownership, resampling and opt-in shell routing."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from concurrent.futures import Future
import numpy as np
from scripts.live_renz_v17 import sign_regions, rgb_input
from scripts.app_shell_v17 import parser, warm


class BufferedRenzTest(unittest.TestCase):
    def test_adjacent_buffer_ownership_and_repeated_signs(self):
        probabilities = [0]*10 + [1]*4 + [0]*10
        all_regions = sign_regions(probabilities, 0, -1, 2)
        first = sign_regions(probabilities, 0, -1, .48)
        second = sign_regions(probabilities, 0, .48, 2)
        self.assertEqual(first + second, all_regions)
        self.assertEqual(len(all_regions), 2)
        self.assertEqual(all_regions, [(0., .48), (.48, .96)])
        self.assertEqual(sign_regions([1, 1], 0, -1, 2), [])

    def test_rgb_clock_color_and_aspect_preservation(self):
        frame = np.zeros((100, 200, 3), np.uint8); frame[:] = [0, 0, 255]
        obs = [SimpleNamespace(seconds=i/20, frame=frame) for i in range(21)]
        images = rgb_input(obs)
        self.assertEqual(images.shape, (26, 224, 224, 3))
        np.testing.assert_array_equal(images[0, 112, 112], [255, 0, 0])
        self.assertTrue((images[0, 0] == 0).all())

    def test_mode_exclusion_and_model_warming(self):
        args = parser().parse_args(['--renz-buffered'])
        self.assertTrue(args.renz_buffered)
        with patch('sys.stderr'), self.assertRaises(SystemExit):
            parser().parse_args(['--renz-buffered', '--familiar-ctc-checkpoint', 'x'])
        shell = SimpleNamespace(build_backdrop=lambda: None)
        with patch('scripts.live_renz_v17.BufferedRenz', return_value='renz'):
            warm(shell, args)
        self.assertEqual(shell.components, 'renz')
        self.assertTrue(shell.ready)

    def test_video_loop_records_commits_and_finalizes(self):
        from scripts.live_renz_v17 import run
        args = parser().parse_args(['--renz-buffered', '--video', 'fake.mp4'])
        frame = np.zeros((160, 240, 3), np.uint8)
        capture = MagicMock(); capture.isOpened.return_value = True; capture.get.return_value = 20.
        capture.read.side_effect = [(True, frame)] * 105 + [(False, None)]
        shell = SimpleNamespace(paused=False, attach=lambda *args: None, present=lambda *args: 255)
        model = MagicMock()
        model.predict.side_effect = lambda obs, start, end: dict(predictions=[dict(committed_gloss='HELLO')], owned_end=end, inference_seconds=.01)
        executor = MagicMock()
        def submit(fn, *args):
            future = Future(); future.set_result(fn(*args)); return future
        executor.submit.side_effect = submit
        with patch('scripts.live_renz_v17.cv2.VideoCapture', return_value=capture), patch('scripts.live_renz_v17.ThreadPoolExecutor', return_value=executor), patch('scripts.live_renz_v17.time.sleep'), patch('scripts.live_isolated_v17.SessionRecorder') as recorder, patch('active.v17.extract_v17.AppleVisionDetector'), patch('scripts.live_stage2_ctc_v17.observe_stage2_frame', side_effect=lambda frame, seconds, *args: SimpleNamespace(frame=frame, seconds=seconds)):
            run(args, shell, model)
        recorder.return_value.add.assert_called_once_with(dict(committed_gloss='HELLO'))
        recorder.return_value.finish_video.assert_called_once()
        recorder.return_value.close.assert_called_once()
        capture.release.assert_called_once()


if __name__ == '__main__': unittest.main()
