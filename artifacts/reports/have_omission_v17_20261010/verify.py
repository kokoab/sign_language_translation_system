"""Small reproducibility check for this experiment's saved evidence."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    return json.loads((HERE / name).read_text())


def main():
    for path, expected in load('source_hashes.json').items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    ablation = load('ablation.json')['results']
    assert load('take11_pts.json')['records'][0]['hypothesis'] == ablation['baseline']['hypothesis']
    assert ablation['noearly_big']['hypothesis'] == ablation['noearly_all']['hypothesis']
    for name, count in [('baseline', 5), ('noearly_big', 6)]:
        assert ablation[name]['hypothesis'].count('HAVE') == count
    live = load('exact_live_controls.json')['results']
    assert live['baseline']['hypothesis'].count('ANGRY') == 0
    assert live['noearly_big']['hypothesis'].count('ANGRY') == 1
    reg = load('regression_pair_val_tune_citizen_val.json')
    for name in ['baseline', 'noearly_big', 'have_cost2']:
        assert len(reg['pair_val:' + name]['rows']) == 103
        assert len(reg['citizen_val:' + name]['rows']) == 378
        assert len(reg['tune:' + name]['rows']) == 89
        assert reg['tune:' + name]['summary'] == reg['tune:baseline']['summary']
    assert reg['pair_val:have_cost2']['summary']['false_have'] == 1
    assert reg['pair_val:baseline']['summary']['false_have'] == 0
    oracle = load('oracle.json')['rows']
    assert {r['region'] for r in oracle} == set(range(6))
    assert all(max(r['have'] for r in oracle if r['region'] == region) >= .4 for region in range(6))
    print('PASS: native/trace word parity, ablation counts, regression sizes, oracle regions, production-source hashes')


if __name__ == '__main__':
    main()
