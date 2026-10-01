"""Dev-only (ASLLRP signer Cory) comparison of feature extractions, e.g. Vision at 640 vs 1280 px."""
import glob, json, re, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from active.v17.letter_ctc_v17 import CLASSES, LETTERS, cer_counts, greedy, load_checkpoint
from scripts.evaluate_letter_ctc_v17 import FEAT, assign, run_model


def dev_score(model, feature_dir, device='mps'):
    manifest = {Path(c['clipFilename']).stem: c for c in json.loads((FEAT / 'manifest.json').read_text())['clips']}
    e = n = exact = words = empty = 0
    for f in sorted(glob.glob(str(Path(feature_dir) / '*.npz'))):
        clip = manifest[Path(f).stem]
        if clip['signerId'] != 'Cory':
            continue
        z = np.load(f); t = z['times']
        _, toks = greedy(run_model(model, z, device))
        tokens = [(CLASSES[c], t[min(len(t) - 1, (a + b) // 2)]) for c, a, b in toks if CLASSES[c] in LETTERS]
        hyp, _ = assign(tokens, clip['words'])
        for w, h in zip(clip['words'], hyp):
            ref = re.sub('[^A-Z]', '', w['gloss'][3:].upper())
            de, dn = cer_counts(h, ref); e += de; n += dn; exact += h == ref; words += 1; empty += h == ''
    return dict(words=words, cer=round(e / n, 4), exact=round(exact / words, 4), empty=round(empty / words, 3))


if __name__ == '__main__':
    for ck in sys.argv[1:]:
        model, _ = load_checkpoint(ck, 'mps')
        print(Path(ck).parent.name, '640:', dev_score(model, FEAT / 'features'),
              '1280:', dev_score(model, ROOT / 'artifacts/generated/letter_ctc_v17/asllrp_dev_det1280'), flush=True)
