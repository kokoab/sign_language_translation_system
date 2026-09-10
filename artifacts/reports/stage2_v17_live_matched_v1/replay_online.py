"""Replay the complete frozen exact-variant development subset, without tuning."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    name = 'adapted_online_replays' if args.checkpoint else 'online_replays'
    rows = json.loads((ROOT / 'artifacts/reports/stage2_v17_live_lock_diagnosis_v1/matched_manifest.json').read_text())
    results = []
    for item in rows:
        key = item['item_id'].replace(':', '_').replace('.mp4', '')
        target = OUT / name / key
        target.mkdir(parents=True, exist_ok=True)
        cmd = [str(ROOT/'venv/bin/python'), 'scripts/live_reel_continuous_v17.py',
               '--video', item['video'], '--sequence-preview', '--realtime-video', '--finish-at-eof', '--no-display', '--no-speech',
               '--naturalizer', 'literal', '--output-root', str(target)]
        if args.checkpoint:
            cmd.extend(['--stage2-live-checkpoint', str(args.checkpoint)])
        with (target/'process.log').open('w') as log:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        row = dict(item_id=item['item_id'], reference=item['reference'], command=cmd, exit_code=proc.returncode)
        if proc.returncode == 0:
            history = sorted(target.glob('*/history.json'))[-1]
            data = json.loads(history.read_text())
            finish = [e for e in data['events'] if e['type']=='finished_sequence_selected']
            assert len(finish)==1, (key, finish)
            row.update(history=str(history.relative_to(ROOT)), finish=finish[0], capture_stats=data['capture_stats'])
        results.append(row)
        (OUT/(name + '.json')).write_text(json.dumps(results, indent=2)+'\n')
        print(key, row.get('finish',{}).get('stage2'), 'exit',proc.returncode,flush=True)
        assert proc.returncode==0, key

if __name__=='__main__': main()
