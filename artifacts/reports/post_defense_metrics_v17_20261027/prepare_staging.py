"""Stage recordings, the FP32 model folder and run plans for the iPhone 13 replay.

FP32 arm: LiveModelLocator.packageDirectory points at fp32_models/. The engine loads the boundary and
recognizer by their production names, so the FP32 exports are stored under those names there; the
alias table and every package hash are recorded in staging_manifest.json.
"""
import sys, json, shutil, hashlib
from pathlib import Path

D = Path(__file__).resolve().parent
R = D.parents[2]
sys.path.insert(0, str(R))
from scripts import segmental_lab_v17 as lab

S = Path.home() / 'slt_post_defense_staging/post_defense'
MODELS = R / 'ATLAS/mobile_app/slt_mobile_app/ios/Runner/LiveReel/Models'
RUNNER = R / 'ATLAS/mobile_app/slt_mobile_app/ios/Runner'


def tree_sha(p):
    h = hashlib.sha256()
    for f in sorted(x for x in Path(p).rglob('*') if x.is_file() and not x.name.startswith('._')):
        h.update(str(f.relative_to(p)).encode()); h.update(f.read_bytes())
    return h.hexdigest()


if S.exists():
    shutil.rmtree(S)
(S / 'videos').mkdir(parents=True)
fp32 = S / 'fp32_models'; fp32.mkdir()
aliases = {
    'AVBoundaryStudentV17L6FP16.mlpackage': D / 'fp32_models/AVBoundaryStudentV17L6FP32.mlpackage',
    'AVBoundaryStudentV17L6LettersFP16.mlpackage': D / 'fp32_models/AVBoundaryStudentV17L6LettersFP32.mlpackage',
    'SpanRecognizerV17ConfidenceDualB8FP16.mlpackage': D / 'fp32_models/SpanRecognizerV17ConfidenceDualB8FP32.mlpackage',
    'MobileCLIP2S0ImageEncoderV17FP32.mlpackage': R / 'artifacts/coreml/MobileCLIP2S0ImageEncoderV17FP32.mlpackage',
    # English models are unchanged FP32 production packages.
    'Stage3T5EncoderV17.mlpackage': MODELS / 'Stage3T5EncoderV17.mlpackage',
    'Stage3T5DecoderStepV17.mlpackage': MODELS / 'Stage3T5DecoderStepV17.mlpackage',
}
manifest = {'fp32_models': {}, 'fp16_bundle': {}, 'resources': {}, 'sets': {}}
ignore = shutil.ignore_patterns('._*', '.DS_Store')
for name, src in aliases.items():
    shutil.copytree(src, fp32 / name, ignore=ignore)
    manifest['fp32_models'][name] = dict(source=str(src.relative_to(R)), tree_sha256=tree_sha(fp32 / name))
for name in ['AVBoundaryStudentV17L6FP16', 'AVBoundaryStudentV17L6LettersFP16', 'SpanRecognizerV17ConfidenceDualB8FP16',
             'MobileCLIP2S0ImageEncoderV17FP16', 'Stage3T5EncoderV17', 'Stage3T5DecoderStepV17']:
    manifest['fp16_bundle'][name] = tree_sha(MODELS / (name + '.mlpackage'))
for f in sorted(MODELS.iterdir()):
    if f.is_file() and not f.name.startswith('._'):
        shutil.copy2(f, fp32 / f.name); manifest['resources'][f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
for name in ['stage3_mobile_naturalizer_manifest_v17.json']:
    shutil.copy2(RUNNER / name, fp32 / name); manifest['resources'][name] = hashlib.sha256((RUNNER / name).read_bytes()).hexdigest()

sets = {}
for s in ['tune', 'test']:
    clips = []
    for i, r in enumerate(lab.rows_for(s)):
        file = f'{s}_{i:03d}.mp4'
        shutil.copy2(R / r['video_path'], S / 'videos' / file)
        clips.append(dict(id=r['source_item_id'], file=file, reference=r['target_sequence'], subset=r.get('subset'),
                          source=r['video_path'], sha256=hashlib.sha256((S / 'videos' / file).read_bytes()).hexdigest()))
    sets[s] = clips
live = R / 'artifacts/reports/sept30_fix_port_v17_20261011/phone/live_reel_videos/20261011_220739_03.mp4'
shutil.copy2(live, S / 'videos/live_000.mp4')
sets['live'] = [dict(id='live:20261011_220739_03', file='live_000.mp4', reference=[], subset='phone_camera_1280x720',
                     source=str(live.relative_to(R)), sha256=hashlib.sha256(live.read_bytes()).hexdigest())]
manifest['sets'] = sets

runs = []
for s in ['tune', 'test', 'live']:
    for p, arm in enumerate(['fp16', 'fp32', 'fp32', 'fp16']):        # ABBA
        runs.append(dict(arm=arm, run=f'{s}_{arm}_p{p}', set=s, clips=sets[s]))
(D / 'runs.json').write_text(json.dumps(runs, indent=1) + '\n')
(D / 'staging_manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
print('staged', S, {k: len(v) for k, v in sets.items()}, 'runs', len(runs))
