"""Run the staged ABBA replays on the iPhone 13 one plan at a time and pull each result as it completes.

Usage: run_phone.py [--skip-setup] [run-name ...]   (default: every run in runs.json not yet pulled complete)
"""
import sys, json, subprocess, time, os
from pathlib import Path

D = Path(__file__).resolve().parent
DEVICE = '00008110-00111D1A0130A01E'
BUNDLE = 'com.kokoab.sltMobileApp'
STAGE = Path.home() / 'slt_post_defense_staging/post_defense'
PRODUCTS = Path.home() / 'slt_post_defense_build/Build/Products'
RESULTS = D / 'phone_results'; RESULTS.mkdir(exist_ok=True)
LOGS = D / 'phone_logs'; LOGS.mkdir(exist_ok=True)


def run(cmd, name, check=True, timeout=None):
    with (LOGS / (name + '.log')).open('w') as out:
        out.write(json.dumps(cmd) + '\n'); out.flush()
        return subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, check=check, timeout=timeout)


def unlocked():
    p = LOGS / 'lock.json'
    run(['xcrun', 'devicectl', 'device', 'info', 'lockState', '--device', DEVICE, '--json-output', str(p)], 'lock')
    return json.loads(p.read_text())['result'].get('passcodeRequired') is False


def copy_to(src, dst, name):
    run(['xcrun', 'devicectl', 'device', 'copy', 'to', '--device', DEVICE, '--source', str(src), '--destination', dst,
         '--domain-type', 'appDataContainer', '--domain-identifier', BUNDLE], name)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    runs = json.loads((D / os.environ.get('RUNS', 'runs.json')).read_text())
    if args:
        runs = [r for r in runs if r['run'] in args]
    if not unlocked():
        raise SystemExit('iPhone is locked: unlock it, keep it awake, then rerun.')
    xctestrun = next(PRODUCTS.glob('*.xctestrun'))
    if '--skip-setup' not in sys.argv:
        run(['xcrun', 'devicectl', 'device', 'install', 'app', '--device', DEVICE, str(PRODUCTS / 'Release-iphoneos/Runner.app')], 'install_test_app')
        if '--app-only' not in sys.argv:   # recordings and FP32 models persist in Documents across app updates
            copy_to(STAGE / 'fp32_models', 'Documents/post_defense/fp32_models', 'copy_models')
            copy_to(STAGE / 'videos', 'Documents/post_defense/videos', 'copy_videos')
    status = json.loads((D / 'phone_run_status.json').read_text()) if (D / 'phone_run_status.json').exists() else {}
    for plan in runs:
        name = plan['run']
        out = RESULTS / (name + '.json')
        if out.exists() and json.loads(out.read_text()).get('complete'):
            continue
        if not unlocked():
            status[name] = 'skipped_locked'; break
        (LOGS / 'plan.json').write_text(json.dumps(plan))
        copy_to(LOGS / 'plan.json', 'Documents/post_defense/plan.json', 'copy_plan_' + name)
        started = time.time()
        result = run(['xcodebuild', 'test-without-building', '-xctestrun', str(xctestrun), '-destination', 'id=' + DEVICE,
                      '-only-testing:RunnerTests/RunnerTests/testPostDefenseReplay',
                      '-resultBundlePath', str(LOGS / f'{name}_{int(started)}.xcresult')], 'test_' + name, check=False)
        run(['xcrun', 'devicectl', 'device', 'copy', 'from', '--device', DEVICE, '--source', f'Documents/post_defense/results/{name}.json',
             '--destination', str(out), '--domain-type', 'appDataContainer', '--domain-identifier', BUNDLE], 'pull_' + name, check=False)
        complete = out.exists() and json.loads(out.read_text()).get('complete')
        status[name] = dict(returncode=result.returncode, complete=bool(complete), minutes=round((time.time() - started) / 60, 2))
        (D / 'phone_run_status.json').write_text(json.dumps(status, indent=2) + '\n')
        print(name, status[name], flush=True)
        if not complete:
            break


if __name__ == '__main__':
    main()
