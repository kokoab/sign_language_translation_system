"""Trace existing development OOV identities through saved training sources, without alias guesses."""
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.finalize_asllrp_other_ctc_manifest_v17 import sha256


def run():
    folder = ROOT / 'artifacts/reports/annotation_identity_audit_v17_20260921'
    cores = json.loads((folder / 'development_cores.json').read_text())
    events = json.loads((folder / 'event_ledger.json').read_text())
    ncslgr_path = ROOT / 'active/v17/ncslgr_supervised_manifest_v17.json'
    ncslgr = {r['source_item_id']: r for r in json.loads(ncslgr_path.read_text())['rows']}
    sources, hashes = [], {}
    for name in ('stage2_v17_grounded_phrases_fixed_20260921', 'stage2_v17_flores_other_20260921'):
        for archive in sorted((ROOT / 'data/local' / name / 'train').glob('*/*.npz')):
            with np.load(archive, allow_pickle=False) as d:
                m = json.loads(str(d['metadata_json'].item()))
            assert m['role'] == 'train'
            if m['source'] == 'ncslgr_strict':
                row = ncslgr[m['source_item_id']]
                assert row['role'] == 'train'
                tokens = [e['raw_gloss'] for e in row['events']]
            elif m['source'] == 'flores_other':
                tokens = m['raw_gloss'].split()
            else:
                continue
            hashes[str(archive.relative_to(ROOT))] = sha256(archive)
            sources.append(dict(source=m['source'], identity=m['source_item_id'],
                                tokens={t.strip('.,!?;:"()[]{}').upper() for t in tokens}))
    identities = []
    for code in sorted({c['asllex_code'] for c in cores if c['status'] == 'oov'}):
        selected = [c for c in cores if c['asllex_code'] == code and c['status'] == 'oov']
        variants = {c['variant'] for c in selected}
        confirmed = [e for e in events if e['role'] == 'train' and e['asllex_code'] == code]
        potential = [dict(source=s['source'], identity=s['identity'], matched_raw_labels=sorted(variants & s['tokens']))
                     for s in sources if variants & s['tokens']]
        identities.append(dict(asllex_code=code, variants=sorted(variants), development_cores=len(selected),
                               confirmed_asllrp_training_occurrences=len(confirmed),
                               exact_string_candidates=potential,
                               exposure_status='confirmed_seen' if confirmed else 'unresolved_cross_corpus_identity',
                               globally_unseen_certified=False))
    checkpoints = []
    for seed in (17321, 17322):
        for arm in ('without_flores', 'with_flores'):
            path = ROOT / f'artifacts/models/flores_other_mps_retry_v17_20260921/{arm}_{seed}/best_model.pth'
            p = torch.load(path, map_location='cpu', weights_only=False)
            base_path, init_path = Path(p['base_checkpoint']), Path(p['initial_head'])
            assert sha256(base_path) == p['base_checkpoint_sha256']
            assert sha256(init_path) == p['initial_head_sha256']
            base = torch.load(base_path, map_location='cpu', weights_only=False)
            provenance = base['training_data_provenance']
            for key in ('citizen_manifest', 'semlex_manifest'):
                assert sha256(ROOT / provenance[key]) == provenance[key + '_sha256']
            initial = torch.load(init_path, map_location='cpu', weights_only=False)
            checkpoints.append(dict(checkpoint=str(path.relative_to(ROOT)), sha256=sha256(path), arm=arm,
                                    base_checkpoint=str(base_path), base_sha256=sha256(base_path),
                                    base_training_provenance=provenance,
                                    initial_head=str(init_path), initial_sha256=sha256(init_path),
                                    initial_head_keys=sorted(initial),
                                    initial_provenance_limit='State/config only; baseline initializer per saved run script, no embedded training lineage.',
                                    protected_test_accessed=False))
    summary = dict(oov_cores=sum(r['development_cores'] for r in identities), identities=len(identities),
                   confirmed_seen_cores=sum(r['development_cores'] for r in identities if r['confirmed_asllrp_training_occurrences']),
                   unresolved_cores=sum(r['development_cores'] for r in identities if not r['confirmed_asllrp_training_occurrences']),
                   unresolved_identities=sum(not r['confirmed_asllrp_training_occurrences'] for r in identities),
                   unresolved_with_string_candidates=[r['asllex_code'] for r in identities
                       if not r['confirmed_asllrp_training_occurrences'] and r['exact_string_candidates']],
                   globally_unseen_certified=0, source_training_archives=dict(Counter(s['source'] for s in sources)),
                   training_started=False, acquisition_started=False, protected_test_accessed=False)
    result = dict(summary=summary, identities=identities, checkpoints=checkpoints,
                  evidence_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
                      (folder/'development_cores.json', folder/'event_ledger.json', ncslgr_path)},
                  source_archive_sha256=hashes,
                  policy='Raw-string overlap is a candidate only, never an official cross-corpus identity link. '
                         'No-match cannot certify absence, including unresolved annotations, incidental signs and blank clips. '
                         'Flores candidates apply only to with_flores heads. All cores remain reused development.')
    (folder / 'exposure_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    run()
