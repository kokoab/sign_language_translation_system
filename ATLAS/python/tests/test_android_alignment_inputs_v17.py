"""An explicit extraction list must not expand to historical defaults."""
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import extract_mediapipe_full_v17 as extraction


class ExplicitExtractionJobs(unittest.TestCase):
    def test_coordinator_preserves_explicit_membership(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            job = {'landmark': str(root / 'only.npz'), 'crops': str(root / 'only.crops.npz')}
            jobs_file = root / 'reviewed.json'
            jobs_file.write_text(json.dumps([job]))
            args = SimpleNamespace(output=root, pose_model='pose_landmarker_lite',
                                   jobs_file=str(jobs_file), shard=None, workers=1,
                                   max_restarts=0, sources=['local_train'], limit=0)
            child = Mock()
            child.poll.return_value = 0
            with patch.object(extraction, 'guard_root') as guard, \
                    patch.object(extraction, 'isolated_jobs', side_effect=AssertionError('Default expansion')), \
                    patch.object(extraction.subprocess, 'Popen', return_value=child) as launch, \
                    patch.object(extraction.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
                extraction.run_isolated(args)
            guard.assert_called_once()
            command = launch.call_args.args[0]
            dispatched = Path(command[command.index('--jobs-file') + 1])
            self.assertEqual(json.loads(dispatched.read_text()), [job])


if __name__ == '__main__':
    unittest.main()
