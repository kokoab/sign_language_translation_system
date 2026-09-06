#!/usr/bin/env python3
"""Export an unreviewed YOU/NEED SignWriting sequence for CWASA Anna."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active.v17.avatar_rig_v17 import signwriting_pilot_sigml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entries', type=Path, required=True,
                        help='JSON containing one selected dictionary entry per gloss')
    parser.add_argument('--glosses', nargs='+', required=True,
                        help='ordered pilot glosses; this does not translate English')
    parser.add_argument('--output', type=Path, required=True, help='new output directory')
    args = parser.parse_args()
    candidates = json.loads(args.entries.read_text())['entries']
    selected = {}
    for entry in candidates:
        gloss = entry['gloss']
        if gloss in selected:
            parser.error(f'multiple candidates for {gloss}; select an exact variant first')
        selected[gloss] = entry
    glosses = [gloss.upper() for gloss in args.glosses]
    missing = set(glosses) - selected.keys()
    if missing:
        parser.error('no selected SignWriting entry for: ' + ', '.join(sorted(missing)))
    entries = [selected[gloss] for gloss in glosses]
    try:
        sigml = signwriting_pilot_sigml(entries)
    except ValueError as error:
        parser.error(str(error))
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        args.entries, Path(__file__),
        Path(__file__).resolve().parents[1] / 'active/v17/avatar_rig_v17.py')}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'sequence.fsw').write_text('\n'.join(e['fsw'] for e in entries) + '\n')
    (args.output / 'sequence.sigml').write_text(sigml + '\n')
    (args.output / 'report.json').write_text(json.dumps(dict(
        glosses=glosses, entries=entries, hashes=hashes, avatar='anna',
        human_accepted=False, training_eligible=False,
        limitations=[
            'Only the two tested YOU/NEED symbol pairs are supported; other notation is rejected',
            'Dictionary variants and generated signing still need fluent video review',
            'Location, thumb pose and timing are assumptions; index bend calibrated for Anna',
            'This exports notation; it does not translate arbitrary English or validate continuous signing',
        ]), indent=2) + '\n')
    print(args.output / 'sequence.sigml')


if __name__ == '__main__':
    main()
