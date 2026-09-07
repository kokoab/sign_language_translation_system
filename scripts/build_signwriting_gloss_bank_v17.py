#!/usr/bin/env python3
"""Build executable SignWriting cores and report unsupported Citizen100 notation."""
from pathlib import Path
import argparse
import json
import sys
import unicodedata

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active.v17.avatar_rig_v17 import (RetargetedAvatar, parse_signwriting_signbox,
    signwriting_pilot_symbols, animate_signwriting_pilot, bone_length_metrics,
    _fallback_rest_hand, _solve_elbow)
from scripts.render_rigged_avatar_v17 import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-bank', type=Path,
        default=Path('artifacts/reports/avatar_gloss_bank_v17_v1'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest_path = Path('active/v17/citizen100_manifest.json')
    lexicon_path = Path('active/v17/citizen100_phonology.json')
    dictionary_path = Path('artifacts/reports/signwriting_avatar_pilot_v17_v1/dictionary_100_coverage.json')
    manifest = json.loads(manifest_path.read_text())
    lexicon = json.loads(lexicon_path.read_text())
    reference = json.loads((args.reference_bank/'report.json').read_text())
    if reference['manifest_sha256'] != sha256(manifest_path) or lexicon['manifest_sha256'] != sha256(manifest_path):
        raise ValueError('frozen manifest mismatch')
    if reference['lexicon_sha256'] != sha256(lexicon_path):
        raise ValueError('frozen phonology mismatch')
    dictionary = json.loads(dictionary_path.read_text())
    candidates = {r['canonical_label']: r['candidates'] for r in dictionary['rows']}
    sources = {r['entry']['gloss']: r for r in reference['items']}
    # Explicitly selected manual variant; never choose a dictionary entry by list order.
    profiles = {
        'YES': dict(fsw='AS20320S23004M513x518S20320493x481S23004488x500', frames=36,
            parameters=dict(wrist_position=[-.18,1.10,.28], wrist_flex_degrees=55)),
        'YOUR': dict(fsw='AS15a28S26500M507x523S15a28494x496S26500493x477', frames=24,
            parameters=dict(wrist_position=[-.13,1.15,.27], travel_metres=.08),
            variant_notes=['Selected train video has fingers together, matching dictionary S15a',
                           'Frozen handshape label is 5; this discrepancy needs fluent review',
                           'Left-hand dictionary pose normalized to right-handed signing']),
        'GOODBYE': dict(fsw='M524x520S27206505x480S14c20477x483', frames=36,
            parameters=dict(wrist_position=[-.22,1.20,.20], wrist_flex_degrees=20),
            variant_notes=['Dictionary wrist-wave candidate; selected source also adds finger hinging',
                           'This candidate does not reproduce that additional finger motion']),
        'HELLO': dict(fsw='M536x518S30007482x483S15a11513x482S26500516x459S20500504x465', frames=24,
            parameters=dict(wrist_position=[-.16,1.38,.16], travel_metres=.08),
            variant_notes=['Selected forehead-contact salute candidate matches source phase order',
                           'Head rim and touch symbols set the initial placement; manual motion moves away'])}
    selected = {g: p['fsw'] for g,p in profiles.items()}
    selected.update({g: sources[g]['entry']['fsw'] for g in ('YOU', 'NEED')})
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(format='slt_avatar_gloss_bank_v17', symbol_motion=True,
        manifest_sha256=sha256(manifest_path), lexicon_sha256=sha256(lexicon_path),
        dictionary_sha256=sha256(dictionary_path), reference_report_sha256=sha256(args.reference_bank/'report.json'),
        code_sha256=sha256(Path(__file__)), rig_code_sha256=sha256(Path('active/v17/avatar_rig_v17.py')),
        items=[], unsupported=[], notation_inventory=[], training_eligible=False, human_accepted=False,
        limitations=['Parsing notation does not establish animation support or lexical accuracy',
                    'New manual motions are review candidates; joint angles, placement and timing are assumptions',
                    'No facial grammar synthesis; source comparison timing is approximate',
                    'Unsupported glosses have no source-motion fallback'])
    for cls in manifest['classes']:
        gloss = cls['canonical_label']
        phonology = {}
        for attribute in lexicon['attributes']:
            index = attribute['targets_by_class_index'][cls['class_index']]
            phonology[attribute['name']] = attribute['values'][index] if 0 <= index < len(attribute['values']) else None
        inventory = dict(gloss=gloss, raw_gloss=cls['citizen_raw_gloss'],
            asl_lex_code=cls['citizen_asl_lex_code'], locked_phonology=phonology, candidates=[])
        for candidate in candidates[gloss]:
            row = dict(fsw=candidate['fsw'], terms=candidate['terms'])
            try:
                row['notation'] = parse_signwriting_signbox(candidate['fsw'])
                for symbol in row['notation']['symbols']:
                    symbol['name'] = unicodedata.name(chr(0x1d800 + int(symbol['base'][1:], 16)-0x100))
            except ValueError as error:
                row['parse_error'] = str(error)
            inventory['candidates'].append(row)
        report['notation_inventory'].append(inventory)
        if gloss not in selected:
            report['unsupported'].append(dict(gloss=gloss, reason='no selected executable notation variant'))
            continue
        fsw = selected[gloss]
        if fsw not in {c['fsw'] for c in candidates[gloss]}:
            raise ValueError(f'{gloss}: selected notation absent from pinned dictionary')
        expected = dict(sign_type='OneHanded', major_location='Neutral', contact='0', wrist_twist='0',
            handshape={'YOU':'1', 'NEED':'bent_1', 'YES':'s', 'YOUR':'5',
                       'GOODBYE':'5', 'HELLO':'closed_b'}[gloss],
            repeated_movement='1' if gloss in ('NEED','YES','GOODBYE') else '0')
        if gloss == 'HELLO':
            expected.update(major_location='Head', minor_location='Forehead', contact='1')
        if any(phonology[key] != value for key, value in expected.items()):
            raise ValueError(f'{gloss}: pinned phonology disagrees with supported manual variant')
        repeated = phonology['repeated_movement'] == '1'
        hand, movement = signwriting_pilot_symbols(fsw, repeated=repeated)
        source = sources[gloss]
        if (source['class_index'], source['raw_gloss'], source['asl_lex_code']) != (
                cls['class_index'], cls['citizen_raw_gloss'], cls['citizen_asl_lex_code']):
            raise ValueError(f'{gloss}: source identity mismatch')
        old_path = args.reference_bank/source['audit']
        if sha256(old_path) != source['audit_sha256'] or sha256(Path(source['source'])) != source['source_sha256']:
            raise ValueError(f'{gloss}: reference hash mismatch')
        path = args.output/f'{gloss.lower()}_audit.npz'
        if gloss in ('YOU', 'NEED'):
            if source['motion_origin'] != 'signwriting_pilot':
                raise ValueError('approved pilot core required')
            path.write_bytes(old_path.read_bytes())
            item = dict(source)
        else:
            # ponytail: selected manual primitives; unsupported contact/nonmanuals stay explicit.
            profile = profiles[gloss]
            count, fps = profile['frames'], 30
            hands = np.repeat(np.stack([_fallback_rest_hand(s) for s in range(2)])[None], count, axis=0)
            hands[:, 1] = animate_signwriting_pilot(_fallback_rest_hand(1), hand, movement,
                                                   frames=count, **profile['parameters'])
            shoulders = np.repeat(np.array([[[.1677125,1.285,0],[-.1677125,1.285,0]]], np.float32), count, axis=0)
            elbows = np.empty_like(shoulders)
            for i in range(count):
                for side in range(2):
                    elbows[i,side], wrist = _solve_elbow(shoulders[i,side], hands[i,side,0], side)
                    hands[i,side] += wrist-hands[i,side,0]
            states = np.tile(np.array(['rest-uncertain', 'symbolic']), (count,1))
            rig = RetargetedAvatar(shoulders, elbows, hands, states, np.zeros((count,2), bool))
            with np.load(old_path) as data:
                # Reference frames align by phase only; no pose or trajectory is read.
                source_indices = np.rint(np.linspace(data['source_indices'][0], data['source_indices'][-1], count)).astype(int)
            np.savez_compressed(path, shoulders=shoulders, elbows=elbows, symbol_hands=hands,
                                hand_states=states, source_indices=source_indices)
            item = {k: source[k] for k in ('source', 'source_sha256', 'class_index', 'raw_gloss', 'asl_lex_code')}
            item.update(entry=dict(gloss=gloss, fsw=fsw), motion_origin='signwriting_pilot',
                        fps=fps, core_start_frame=0, core_stop_frame=count, **bone_length_metrics(rig),
                        hand_participation=[False,True], human_accepted=False,
                        motion_parameters=profile['parameters'], variant_notes=profile.get('variant_notes',[]),
                        motion_geometry_source='static avatar anatomy; no source-video trajectory')
        item.update(audit=path.name, audit_sha256=sha256(path), hand_symbol=hand, movement_symbol=movement)
        report['items'].append(item)
    report.update(coverage=len(report['items']), requested_classes=len(manifest['classes']),
                  complete=not report['unsupported'])
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('coverage', 'requested_classes', 'complete')}, indent=2))


if __name__ == '__main__':
    main()
