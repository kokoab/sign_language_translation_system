"""Package the v2 caption engine for the desktop app (no training data needed).

Writes artifacts/models/live_captions_v17_20261014/: the five state dicts, a
compact vocab.json (vocabularies, legal forms, allowed insertion/form pairs) and
manifest.json with hashes and provenance.
"""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'artifacts/models/live_captions_v17_20261014'
MODELS = ['tagger_k1.pt', 'tagger_k1b.pt', 'tagger_k0.pt', 'mover.pt', 'labeler.pt']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    D = json.loads((HERE / 'data.json').read_text())
    pairs = {}
    for r in D['train']:
        for p, f in zip(r['pre'], r['form']):
            if p != -100 and f != -100:
                pairs[(p, f)] = pairs.get((p, f), 0) + 1
    vocab = {'glosses': D['glosses'], 'pre_vocab': D['pre_vocab'], 'post_vocab': D['post_vocab'],
             'form_vocab': D['form_vocab'], 'legal_forms': D['legal_forms'],
             'allowed_pairs': sorted([p, f] for (p, f), c in pairs.items() if c >= 5)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'vocab.json').write_text(json.dumps(vocab))
    for m in MODELS:
        shutil.copy2(HERE / m, OUT / m)
    manifest = {
        'name': 'live_captions_v17_20261014',
        'description': 'Live gloss-to-English captions: k0 draft, k1 live (two-model average), '
                       'clause mover + labeler. Experiment evidence: '
                       'experiments/streaming_translation/caption_tagger/REPORT_V2.md',
        'live_models': ['tagger_k1.pt', 'tagger_k1b.pt'], 'draft_model': 'tagger_k0.pt',
        'mover': 'mover.pt', 'labeler': 'labeler.pt',
        'architecture': {'tagger': {'d': 192, 'layers': 3, 'heads': 4},
                         'mover': {'d': 160, 'layers': 3, 'heads': 4}},
        'policy': {'end_of_turn_seconds': 3.5, 'draft': 'no_post', 'revise': 'meaning',
                   'keep_confidence': 0.6, 'pronoun_probability': 0.5, 'context_signs': 0},
        'training_corpus_sha256': '62081b4db0295e216082b610bc5dfef3b101433a45bb46dfbd3f65c5b9278cce',
        'sha256': {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.name != 'manifest.json'},
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest['sha256'], indent=1))


if __name__ == '__main__':
    main()
