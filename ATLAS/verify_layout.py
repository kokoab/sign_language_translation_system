"""Verify the canonical ATLAS source and its historical compatibility links."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'ATLAS/source_manifest.json').read_text())
    for row in manifest['files']:
        old = ROOT / row['original']
        new = ROOT / row['canonical']
        if not old.is_symlink() or old.resolve() != new.resolve() or not new.is_file():
            raise RuntimeError('Broken canonical source link: ' + row['original'])
        if new.suffix == '.py':
            ast.parse(new.read_text(), filename=str(new))
    old_mobile = Path(manifest['mobile_original'])
    new_mobile = ROOT / 'ATLAS/mobile_app/slt_mobile_app'
    if not old_mobile.is_symlink() or old_mobile.resolve() != new_mobile.resolve():
        raise RuntimeError('Broken mobile compatibility link')
    for path in ('pubspec.yaml', 'ios/Runner.xcworkspace',
                 'ios/Runner/LiveReel/LiveReelModels.swift',
                 'android/app/src/main/kotlin/com/example/slt_mobile_app/live/LiveModels.kt'):
        if not (new_mobile / path).exists():
            raise RuntimeError('Missing mobile project file: ' + path)
    print('ATLAS layout OK: ' + str(len(manifest['files'])) + ' source/resource links and mobile project')


if __name__ == '__main__':
    main()
