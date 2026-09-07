"""Matched familiar-development replays. Never reads test/sealed examples."""
import json
from pathlib import Path
import statistics
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from active.v17.train_stage_2_v17 import edit_distance

ROOT = Path(__file__).resolve().parent / 'final'
ROOT.mkdir(parents=True, exist_ok=True)
SOURCES = REPO / 'artifacts/reports/live_stage2_ctc_v17_timebase15_eval_v1'
EXPECTED = [r['phrase'].split() for r in json.loads((SOURCES / 'report.json').read_text())['rows']]
# Only these five already-consumed development/reference sources are allowed.
VIDEOS = [
    ('GOOD MORNING', 'data/raw_videos/PHRASES/GOOD_MORNING/GOOD_MORNING_11.mp4'),
    ('HELLO HOW YOU', 'data/raw_videos/PHRASES/HELLO_HOW_YOU/0f3336e6.mp4'),
    ('MY NAME', 'data/raw_videos/PHRASES/MY_NAME/bab3adfa.mp4'),
    ('THANKYOU FRIEND', 'data/raw_videos/PHRASES/THANKYOU_FRIEND/fa5f33d6.mp4'),
    ('TOMORROW SCHOOL GO', 'data/raw_videos/PHRASES/TOMORROW_SCHOOL_GO/973e8bb0.mp4'),
]
assert [p.split() for p, _ in VIDEOS] == EXPECTED
rows = []
for index, (phrase, video) in enumerate(VIDEOS):
    for lane, script in [('baseline', 'live_reel_stage1_v17.py'),
                         ('continuous', 'live_reel_continuous_v17.py')]:
        destination = ROOT / lane / f'{index:02d}'
        destination.mkdir(parents=True, exist_ok=True)
        histories = list(destination.glob('*/history.json'))
        if not histories:
            command = [sys.executable, str(REPO / 'scripts' / script),
                       '--video', str(REPO / video), '--no-display', '--no-speech',
                       '--no-finish-gesture', '--naturalizer', 'literal',
                       '--finish-at-eof', '--realtime-video',
                       '--output-root', str(destination),
                       '--expected-sequence', *phrase.split()]
            print(f'{lane}: {phrase}', flush=True)
            with (destination / 'run.log').open('w') as log:
                subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=180)
            histories = list(destination.glob('*/history.json'))
        assert len(histories) == 1
        history = json.loads(histories[0].read_text())
        assert 'finished_utc' in history
        events = history['events']
        selected = next(e for e in events if e['type'] == 'finished_sequence_selected')
        stage2_events = [e for e in events if e['type'] == 'stage2_sequence_update']
        previews = [e for e in events if e['type'] == 'provisional_gloss']
        commits = [p for p in history['predictions'] if p.get('committed_gloss')]
        if lane == 'continuous':
            assert all(e['after_finish'] for e in stage2_events)
            assert selected['selected'] == selected['stage1']
        verifications = [p['full_verifier']['latency_ms']['total']
                         for p in history['predictions'] if p.get('full_verifier')]
        reference = phrase.split()
        row = {
            'lane': lane, 'phrase': phrase, 'source': video,
            'history': str(histories[0].relative_to(REPO)),
            'hypothesis': selected['selected'], 'edits': edit_distance(reference, selected['selected']),
            'stage2_candidate': selected['stage2'],
            'stage2_candidate_edits': edit_distance(reference, selected['stage2']) if stage2_events else None,
            'first_preview_elapsed_s': previews[0]['elapsed_seconds'] if previews else None,
            'first_preview_gloss': previews[0]['gloss'] if previews else None,
            'first_commit_elapsed_s': commits[0]['completed_elapsed_seconds'] if commits else None,
            'finish_decode_ms': selected['finish_decode_ms'],
            'verifier_median_ms': statistics.median(verifications) if verifications else None,
            'retained_frames': sum(e['retained_frames'] for e in events if e['type'] == 'committed_buffer_retained'),
            'capture': history['capture_stats'],
            'stage2_updates': len(stage2_events),
        }
        rows.append(row)
        (ROOT / 'comparison.json').write_text(json.dumps({
            'scope': 'five familiar development recordings; paced saved video, not webcam or independent accuracy',
            'test_accessed': False, 'rows': rows,
        }, indent=2) + '\n')
        print(json.dumps(row), flush=True)
