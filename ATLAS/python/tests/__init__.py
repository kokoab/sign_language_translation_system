"""ATLAS tests with quiet library output and visible unittest results."""

from contextlib import contextmanager
import io
import os
from pathlib import Path
import sys
import unittest


@contextmanager
def _quiet_library_output():
    """Silence Python/native output temporarily, keeping original descriptors."""
    sys.stdout.flush()
    sys.stderr.flush()
    with os.fdopen(os.dup(1), "w", buffering=1) as stdout, \
            os.fdopen(os.dup(2), "w", buffering=1) as stderr, \
            open(os.devnull, "w") as silent:
        try:
            os.dup2(silent.fileno(), 1)
            os.dup2(silent.fileno(), 2)
            yield {1: stdout, 2: stderr}
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout.fileno(), 1)
            os.dup2(stderr.fileno(), 2)


class _QuietLibrarySuite(unittest.TestSuite):
    def run(self, result, debug=False):
        original_stream = getattr(result, "stream", None)
        with _quiet_library_output() as streams:
            try:
                descriptor = original_stream.fileno()
            except (AttributeError, io.UnsupportedOperation):
                descriptor = None
            if descriptor in streams:
                result.stream = type(original_stream)(streams[descriptor])
            try:
                return super().run(result, debug=debug)
            finally:
                if original_stream is not None:
                    result.stream = original_stream


def load_tests(loader, tests, pattern):
    """Unittest discovery hook: silence dependency imports and test execution."""
    directory = Path(__file__).resolve().parent
    with _quiet_library_output():
        suite = loader.discover(
            start_dir=str(directory),
            pattern=pattern or "test_*.py",
            top_level_dir=str(directory.parents[2]),
        )
    return _QuietLibrarySuite(suite)
