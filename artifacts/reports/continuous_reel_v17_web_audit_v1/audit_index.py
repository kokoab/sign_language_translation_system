"""Conservative exact-raw-label candidates from the official NCSLGR index.

Text matches are candidates, not cross-corpus lexical-variant approval or timed spans.
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
manifest = json.loads((REPO / 'active/v17/citizen100_manifest.json').read_text())
locked = {c['citizen_raw_gloss']: c['canonical_label'] for c in manifest['classes']}
archive = ROOT / 'video_index.zip'
with zipfile.ZipFile(archive) as z:
    text = z.read('video_index-20120129/files_by_xml_file.csv').decode('utf-8-sig')
rows = list(csv.DictReader(io.StringIO(text, newline='')))
utterances = {}
conflicts = []
for row in rows:
    key = (row['XML file'], row['Occurs in utterance id'])
    if key in utterances and utterances[key]['main gloss'] != row['main gloss']:
        conflicts.append(key)
    utterances.setdefault(key, row)
assert not conflicts
small = {'ncslgr10a.xml', 'ncslgr10b.xml', 'ncslgr10c.xml', 'ncslgr10d.xml'}
def tokenize(value):
    # Keep quoted multi-word classifier/gesture descriptions within their token.
    if value.count('"') % 2:
        return []
    return re.findall(r'(?:[^\s"]|"[^"]*")+', value)
assert tokenize('HOME DCL"SCHOOL GO" TIME') == ['HOME', 'DCL"SCHOOL GO"', 'TIME']
def spans(tokens):
    run = []
    result = []
    for token in [*tokens, None]:
        if token in locked:
            run.append(token)
        else:
            if len(run) >= 2:
                result.append(run)
            run = []
    return result
assert spans(['HOME', 'unsupported', 'SCHOOL']) == []
candidates = []
for (collection, utterance), row in utterances.items():
    if collection in small:
        continue
    tokens = tokenize(row['main gloss'])
    for span in spans(tokens):
        candidates.append({
            'collection': collection, 'utterance': utterance,
            'raw_glosses': span, 'candidate_canonical_labels': [locked[t] for t in span],
            'full_main_gloss': row['main gloss'],
            'training_eligible': False,
            'reason': 'index lacks per-sign timing and participant IDs; exact spelling alone does not approve lexical variants',
        })
summary = {
    'source': 'https://www.bu.edu/asllrp/ncslgr-for-download/video_index-20120129.zip',
    'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
    'archive_bytes': archive.stat().st_size,
    'index_rows_including_views': len(rows), 'unique_utterances': len(utterances),
    'collections': len({k[0] for k in utterances}),
    'existing_subset_utterances': sum(k[0] in small for k in utterances),
    'outside_existing_subset_utterances': sum(k[0] not in small for k in utterances),
    'outside_subset_candidate_spans': len(candidates),
    'outside_subset_candidate_parent_utterances': len({(c['collection'], c['utterance']) for c in candidates}),
    'distinct_candidate_sequences': len({tuple(c['raw_glosses']) for c in candidates}),
    'candidate_raw_classes': sorted({t for c in candidates for t in c['raw_glosses']}),
    'matching': 'case-sensitive exact Citizen raw gloss; no alias, normalization, or numeric variant merging',
    'timing_and_signer_coverage_verified': False, 'training_eligible': False,
    'protected_video_or_annotations_accessed': False,
}
(ROOT / 'ncslgr_index_audit.json').write_text(json.dumps(summary, indent=2) + '\n')
(ROOT / 'ncslgr_phrase_candidates.json').write_text(json.dumps(candidates, indent=2) + '\n')
print(json.dumps(summary, indent=2))
