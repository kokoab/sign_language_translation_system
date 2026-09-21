"""Pinned admission and fail-closed training gate for the repaired phrase dataset."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / 'active/v17/approved_phrase_manifest_20260921_v2.json'
APPROVED_ROOT = ROOT / 'data/local/approved_phrases_v17_20260921_v2'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_manifest(path=DEFAULT_MANIFEST):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get('format') != 'approved_phrase_data_v17':
        raise ValueError('unsupported dataset manifest')
    expected = {str((ROOT / r['path']).absolute()):r['sha256'] for r in manifest['admitted']}
    if len(expected) != len(manifest['admitted']):
        raise ValueError('duplicate manifest membership')
    actual = set()
    for value in manifest['roots'].values():
        root = ROOT / value
        if not root.is_dir():
            raise ValueError(f'missing approved root: {root}')
        actual.update(str(p.absolute()) for p in root.rglob('*.npz'))
    if actual != set(expected):
        raise ValueError('approved dataset membership differs from manifest')
    for name, sha in {**expected, **manifest['evidence_sha256']}.items():
        if digest(ROOT / name) != sha:
            raise ValueError(f'dataset/evidence hash mismatch: {name}')
    return dict(manifest=str(path.resolve()), sha256=digest(path), archives=len(expected),
                training_ready=manifest['training_ready'], blockers=manifest['blockers'])


def require_training_manifest(args):
    path = Path(getattr(args, 'dataset_manifest', DEFAULT_MANIFEST))
    manifest = json.loads(path.read_text())
    if manifest.get('training_entrypoint'):
        raise ValueError('recipe-scoped manifest must use its dedicated training entrypoint')
    for key, value in manifest['roots'].items():
        if Path(getattr(args, key)).resolve() != (ROOT / value).resolve():
            raise ValueError(f'{key} differs from approved dataset manifest; historical roots are not admitted')
    if getattr(args, 'supplement_root', None) is not None:
        raise ValueError('unapproved supplement: Flores identities remain unresolved')
    if getattr(args, 'full_local_root', None) is not None:
        raise ValueError('unapproved full-local supplement')
    provenance = verify_manifest(path)
    if not provenance['training_ready']:
        raise ValueError('approved dataset is not training-ready: ' + '; '.join(provenance['blockers']))
    return provenance


if __name__ == '__main__':
    print(json.dumps(verify_manifest(), indent=2))
