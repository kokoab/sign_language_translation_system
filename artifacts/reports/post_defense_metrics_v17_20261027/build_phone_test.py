"""Build the post-defense replay test from a private internal-disk mirror; canonical sources are never edited."""
from pathlib import Path
import subprocess, json, hashlib, os

D = Path(__file__).resolve().parent
SOURCE = D.parents[2] / 'ATLAS/mobile_app/slt_mobile_app'
MIRROR = Path.home() / 'slt_post_defense_mirror/slt_mobile_app'
DERIVED = Path.home() / 'slt_post_defense_build'
DEVICE = '00008110-00111D1A0130A01E'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
canonical_tests = SOURCE / 'ios/RunnerTests/RunnerTests.swift'
before = {p: sha(SOURCE / 'ios/Runner/LiveReel' / p) for p in os.listdir(SOURCE / 'ios/Runner/LiveReel') if p.endswith('.swift')}
before['RunnerTests.swift'] = sha(canonical_tests)

MIRROR.mkdir(parents=True, exist_ok=True)
subprocess.run(['rsync', '-a', '--delete', '--exclude', '/build/', '--exclude', '/.dart_tool/', '--exclude', '/android/',
                '--exclude', '/ios/Flutter/Generated.xcconfig', '--exclude', '/ios/Flutter/flutter_export_environment.sh',
                f'{SOURCE}/', f'{MIRROR}/'], check=True)
tests = MIRROR / 'ios/RunnerTests/RunnerTests.swift'
tests.write_text(tests.read_text() + (D / 'PostDefenseReplayTest.swift').read_text())
subprocess.run(['flutter', 'pub', 'get'], cwd=MIRROR, check=True, stdout=subprocess.DEVNULL)
subprocess.run(['flutter', 'build', 'ios', '--release', '--config-only'], cwd=MIRROR, check=True, stdout=subprocess.DEVNULL)
command = ['xcodebuild', '-workspace', 'ios/Runner.xcworkspace', '-scheme', 'Runner', '-configuration', 'Release',
           '-destination', f'id={DEVICE}', '-derivedDataPath', str(DERIVED), '-allowProvisioningUpdates',
           '-only-testing:RunnerTests/RunnerTests/testPostDefenseReplay', 'DEVELOPMENT_TEAM=2BJ6GHCUJ2', 'CODE_SIGN_IDENTITY=Apple Development', 'ENABLE_TESTABILITY=YES', 'build-for-testing']
with (D / 'phone_build.log').open('w') as log:
    done = subprocess.run(command, cwd=MIRROR, stdout=log, stderr=subprocess.STDOUT)
after = {p: sha(SOURCE / 'ios/Runner/LiveReel' / p) for p in before if p != 'RunnerTests.swift'}
after['RunnerTests.swift'] = sha(canonical_tests)
assert after == before, 'canonical source changed during build'
(D / 'phone_build_status.json').write_text(json.dumps(dict(
    returncode=done.returncode, command=command, mirror=str(MIRROR), derived=str(DERIVED),
    canonical_source_sha256_unchanged=before, mirror_tests_sha256=sha(tests),
    test_source_sha256=sha(D / 'PostDefenseReplayTest.swift')), indent=2) + '\n')
print('build returncode', done.returncode)
raise SystemExit(done.returncode)
