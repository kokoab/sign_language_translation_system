"""Detached fixed boundary experiment; one notification after training and replay."""
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from active.v17.approved_phrase_data_v17 import digest
from scripts.train_temporal_boundary_v17 import REPORT, atomic, train
from scripts.evaluate_temporal_boundary_v17 import learned, write_report


if __name__ == '__main__':
    started = time.perf_counter()
    files = ['active/v17/temporal_boundary_v17.py', 'scripts/train_temporal_boundary_v17.py',
             'scripts/evaluate_temporal_boundary_v17.py', 'scripts/live_boundary_v17.py',
             'scripts/live_reel_stage1_v17.py', 'scripts/live_isolated_v17.py',
             'scripts/live_stage2_ctc_v17.py', 'scripts/app_shell_v17.py']
    pins = {p: digest(ROOT / p) for p in files}
    atomic(REPORT / 'execution_pins.json', pins)
    message = 'ASL boundary experiment failed; inspect completion.json.'
    try:
        train()
        if any(digest(ROOT / p) != sha for p, sha in pins.items()):
            raise ValueError('code changed during detached training; replay not started')
        learned()
        if any(digest(ROOT / p) != sha for p, sha in pins.items()):
            raise ValueError('code changed during paired replay; results need review')
        write_report()
        atomic(REPORT / 'completion.json', dict(status='complete', elapsed_seconds=time.perf_counter() - started,
                                               promoted=False, next_action='Review REPORT.md and learned_results.json before any runtime promotion or contextual-identity recipe.'))
        subprocess.run([str(ROOT / 'venv/bin/python'), 'scripts/index_large_artifacts_v17.py'], cwd=ROOT, check=True)
        message = 'ASL boundary training and whole-video replay complete. Results ready; default Reel unchanged.'
    except BaseException:
        atomic(REPORT / 'completion.json', dict(status='failed', elapsed_seconds=time.perf_counter() - started, traceback=traceback.format_exc()))
        raise
    finally:
        subprocess.run(['osascript', '-e', 'display notification ' + json.dumps(message) + ' with title "SLT temporal boundary"'], check=False)
