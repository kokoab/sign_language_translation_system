"""Reproduce read-only runtime diagnostics and the requested local video gallery.

Run with venv/bin/python. Writes only this report; no inference, training, or test data.
"""
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from scripts.live_stage2_ctc_v17 import collapse_ctc_path, roll_ctc_prefix


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,nb_frames:format=duration',
        '-of', 'json', str(path)]))


def logits(path):
    out = np.full((len(path), 3), -10., np.float32)
    out[np.arange(len(path)), path] = 10.
    return out


assert collapse_ctc_path(logits([1, 1, 1]), 3)[0] == (1,)
assert collapse_ctc_path(logits([1, 0, 1]), 3)[0] == (1, 1)
tokens, positions = collapse_ctc_path(logits([1] * 64), 64)
locked, retained, shifted = roll_ctc_prefix([], ['A'], list(positions))
next_tokens, _ = collapse_ctc_path(logits([1] * 64), 64)
rolled = locked + ['A' for _ in next_tokens]
assert rolled == ['A', 'A'] and not retained and not shifted

session_path = 'artifacts/reports/live_reel_stage1_v17/20260915_220852_978390/history.json'
history = read(session_path)
times = np.array(history['video_source_timestamps_seconds'])
assert np.all(np.diff(times) >= 0)
session_video = Path(history['video'])
session_probe = probe(session_video)
events = [e for e in history['events'] if e['type'] == 'stage2_sequence_update']
early_duplicate = next(e for e in events if e.get('hypothesis') == ['I', 'I'])
assert early_duplicate['window_count'] == 2 and not early_duplicate['after_finish']
selector = history['models']['general_selector']
assert sha(selector['path']) == selector['sha256']
manifest_path = 'artifacts/reports/o5s5_augmented_v17_20260914/evaluation_manifest.json'
evaluation_path = 'artifacts/reports/joint_ctc_aligned_v17_20260915/evaluation.json'
manifest = {r['source_item_id']: r for r in read(manifest_path)['rows']}
evaluation = read(evaluation_path)
predictions = {r['item_id']: r for r in evaluation['rows']}
vocab = {r['class_index']: r['canonical_label'] for r in read('active/v17/citizen100_manifest.json')['classes']}
cards = []


def export(card, source, filters, seek=None, duration=None):
    dest = HERE / 'media' / (card['id'] + '.mp4')
    command = ['ffmpeg', '-v', 'error', '-y']
    if seek is not None:
        command += ['-ss', str(seek)]
    command += ['-i', str(source)]
    if duration is not None:
        command += ['-t', str(duration)]
    command += ['-an', '-vf', filters, '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '21', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(dest)]
    # ponytail: preserve existing exports; delete this report's clip to regenerate it.
    if not dest.exists():
        subprocess.run(command, check=True)
    card.update(media='media/' + dest.name, source_video=str(source), media_sha256=sha(dest))
    info = probe(dest)
    card['export_duration_seconds'] = float(info['format']['duration'])
    assert card['export_duration_seconds'] > 0
    if 'frame_source_times' in card:
        assert int(info['streams'][0]['nb_frames']) == len(card['frame_source_times'])
    cards.append(card)


for name, start, end, title, note in [
    ('webcam_i', 24., 27.2, 'One chest-point episode → I I',
     'Sampled-frame visual review: one chest-point gesture, then hand lowering. Intended ASL gloss is unverified. The log already contains I I after two accepted windows, before Finish or context rollover.'),
    ('webcam_hello', 12., 18.5, 'Two separate salutations → HELLO HELLO',
     'Repetition control: two visibly separate salutation gestures, separated by rest. HELLO HELLO should not automatically be removed. This is a visual observation, not an expert linguistic annotation.'),
    ('webcam_eat', 35., 39., 'Chest-point then hand-to-mouth → EAT EAT',
     'Visual review: a chest-point episode followed by a hand-to-mouth episode. The log changes MY → EAT → EAT EAT. Intended gloss sequence remains unverified; classification and segmentation may both contribute.')]:
    first, last = np.searchsorted(times, [start, end]).tolist()
    selected_events = [e for e in events if start <= e['end_seconds'] <= end]
    intervals = [dict(start_seconds=e['start_seconds'], end_seconds=e['end_seconds'],
                      label=('accepted window → ' + (' '.join(e['hypothesis']) or '∅')) if e['accepted'] else 'rejected window',
                      token_positions=e.get('token_positions'), window_count=e.get('window_count'))
                 for e in selected_events]
    export(dict(id=name, title=title, note=note, model='Recorded live selector; no new inference',
                reference='No expert ground truth. See visual observation above.',
                prediction=' → '.join(' '.join(e['hypothesis']) or '∅' for e in selected_events if e['accepted']),
                annotation_status='Rows below are decoder windows, NOT sign annotations.',
                intervals=intervals, events=selected_events, source_annotation=session_path,
                frame_source_times=times[first:last].tolist(), frame_rate=15, offset=0,
                source_frame_range=[first, last], source_video_sha256=sha(session_video)),
           session_video, f'trim=start_frame={first}:end_frame={last},setpts=PTS-STARTPTS')

for key, name in [
    ('asllrp:15718738.mp4:span00', 'asllrp_night_time'),
    ('asllrp:4236779.mp4:span00', 'asllrp_time_friend'),
    ('asllrp_other_ctc:29626:span00', 'asllrp_morning'),
    ('local:HELLO_HOW_YOU:818e7c56', 'local_hello_how_you')]:
    row, pred = manifest[key], predictions[key]
    assert row['role'] == 'validation'
    source = ROOT / row['video_path']
    assert sha(source) == row['video_sha256']
    export(dict(id=name, title=' '.join(row['reference']) + ' — development failure',
                note='Existing annotations and saved offline predictions. A selected failure example, not an unbiased sample. OTHER means outside the locked vocabulary; it is not silence.',
                model='Research epoch 42; NOT the webcam model',
                reference=' '.join(row['reference']), prediction=' '.join(pred['final']) or '∅',
                annotation_status=('Manifest marks reference intervals verified.' if row['verified_reference_intervals'] else
                                   'Manifest does NOT mark intervals verified; no new expert annotation check.') +
                                  (' Clip-level reference only; no word timings supplied.' if not row['intervals'] else ''),
                intervals=row['intervals'], offset=0, source_annotation=manifest_path,
                original_reference_including_oov=row['original_reference_including_oov'],
                source_item_id=key, role=row['role'], signer=row.get('signer_id'),
                source_video_sha256=row['video_sha256'], saved_evaluation=pred),
           source, 'scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1')

lg_source = next(r for r in read('artifacts/reports/o5s5_citizen100_v17/sources.json') if r['source_item_id'] == 'O5S5_002_LG')
lg = next(r for r in read('artifacts/reports/o5s5_augmented_v17_20260914/combined_supervision.json')['rows'] if r['source_item_id'] == 'O5S5_002_LG')
assert lg['role'] == 'validation' and not lg['all_signs_annotated'] and not lg['background_training_eligible']
assert sha(ROOT / lg_source['video_path']) == lg_source['video_sha256']
for core in evaluation['raw_core']['validation']['rows'][:2]:
    _, event, start, end = core['identity'].split(':')
    original = lg['intervals'][int(event)]
    assert original['label'] == vocab[core['target']]
    start, end = float(start) - .65, float(end) + .65
    annotations = [r for r in lg['intervals'] if r['end_seconds'] >= start and r['start_seconds'] <= end]
    name = 'o5s5_lg_' + original['label'].lower()
    export(dict(id=name, title='LG: ' + original['label'] + ' → ' + vocab[core['prediction']],
                note='The prediction uses only the exact annotated sign core; this export includes 0.65 seconds of context on each side. All listed timings are original source-video seconds. Unannotated regions are unknown, not blank.',
                model='Research epoch 42 POOLED core classifier; NOT continuous CTC or webcam output',
                reference=original['label'], prediction=vocab[core['prediction']], intervals=annotations,
                annotation_status='Partial O5S5 annotation; original ID gloss shown in parentheses. No new expert correction.',
                offset=start, source_annotation=lg_source['eaf_path'],
                source_item_id=lg['source_item_id'], role='validation', signer='LG',
                source_video_sha256=lg_source['video_sha256'], saved_evaluation=core,
                exact_core_interval=original), ROOT / lg_source['video_path'],
           'scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1', seek=start, duration=end-start)

evidence = dict(format='stage2_research_review_v1', session_history=session_path,
    session_history_sha256=sha(ROOT / session_path), live_models=history['models'],
    early_duplicate_event=early_duplicate, session_sequence_updates=len(events),
    rejected_windows=sum(not e['accepted'] for e in events),
    rendering_examples=[{k:u.get(k) for k in ('glosses','input_glosses','sentence','rendering_mode')} for u in history['utterances']],
    ctc_checks={'held_path_A_A_A':['A'], 'separated_path_A_blank_A':['A','A'],
                'synthetic_rolling_continuous_A_path':rolled,
                'rolling_limit':'Code counterexample, not attribution of the early webcam duplicate.'},
    webcam_recorded_frames=int(session_probe['streams'][0]['nb_frames']),
    webcam_timestamp_count=len(times),
    timestamp_caveat='Webcam video is CFR 15 fps; source timestamps are irregular. Gallery maps frame indices through the saved timestamps. The two trailing unmapped frames are excluded.',
    evaluation_path=evaluation_path, evaluation_sha256=sha(ROOT / evaluation_path),
    research_checkpoint=evaluation['checkpoint'], research_checkpoint_sha256=evaluation['checkpoint_sha256'],
    protected_test_accessed=False, cards=cards)
(HERE / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')

sections = []
for card in cards:
    esc = html.escape
    rows = ''.join(f'<tr data-start="{r["start_seconds"]}" data-end="{r["end_seconds"]}"><td><button type="button">{r["start_seconds"]:.3f}–{r["end_seconds"]:.3f}s</button></td><td>{esc(r["label"])}'+
                   (f' ({esc(r["id_gloss"])})' if 'id_gloss' in r else '')+'</td></tr>' for r in card['intervals'])
    sections.append(f'''<article id="{card['id']}"><h2>{esc(card['title'])}</h2>
<p class="model">{esc(card['model'])}</p><video controls preload="metadata" src="{card['media']}"></video>
<p><label>Playback speed <select aria-label="Playback speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option></select></label> · <output>Source time</output></p>
<p><b>Reference:</b> {esc(card['reference'])}<br><b>Output:</b> {esc(card['prediction'])}</p>
<p>{esc(card['note'])}</p><p><b>Annotation status:</b> {esc(card['annotation_status'])}</p>
<table><thead><tr><th>Source seconds — click to seek</th><th>Annotation / recorded window</th></tr></thead><tbody>{rows}</tbody></table>
<details><summary>Provenance</summary><p>Video: <code>{esc(card['source_video'])}</code></p><p>Annotation: <code>{esc(card['source_annotation'])}</code></p></details></article>''')
page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stage 2 video evidence</title>
<style>body{font:17px/1.5 system-ui,sans-serif;margin:32px auto;padding:0 20px;max-width:1150px;color:#172536;background:#f3f5f7}h1{line-height:1.15}article{background:white;border:1px solid #d5dce2;border-radius:12px;padding:24px;margin:28px 0}video{width:100%;max-height:470px;background:#182330}table{border-collapse:collapse;width:100%}th,td{padding:8px;text-align:left;border-bottom:1px solid #ddd}tr.active{background:#d8f3e3}button,select{font:inherit;padding:5px;cursor:pointer}code{overflow-wrap:anywhere}a{color:#185fad}.model{color:#765119;font-weight:600}output{font-variant-numeric:tabular-nums}</style>
<h1>Held signs, real repetitions, and difficult corpus clips</h1>
<p>Nine selected examples. Webcam observations are provisional; corpus annotations are preserved as supplied. Nothing here establishes the signer's intended meaning without human review. Slow playback helps inspect short signs; original timing remains visible.</p>
<p><a href="README.md">Research report</a> · <a href="evidence.json">Full annotations and provenance (JSON)</a></p>
<nav>'''+ ' · '.join(f'<a href="#{c["id"]}">{html.escape(c["id"])}</a>' for c in cards)+'</nav>'+''.join(sections)
page += '<script id="cards" type="application/json">'+json.dumps(cards).replace('<','\\u003c')+'</script>'
page += '''<script>
const cards=JSON.parse(document.getElementById('cards').textContent);
for(const c of cards){
 const a=document.getElementById(c.id),v=a.querySelector('video'),rows=[...a.querySelectorAll('tbody tr')];
 const sourceTime=()=>c.frame_source_times ? c.frame_source_times[Math.min(c.frame_source_times.length-1,Math.floor(v.currentTime*c.frame_rate))] : v.currentTime+c.offset;
 a.querySelector('select').onchange=e=>v.playbackRate=Number(e.target.value);
 for(const row of rows)row.querySelector('button').onclick=()=>{
  const target=Number(row.dataset.start);
  if(c.frame_source_times){const index=c.frame_source_times.findIndex(t=>t>=target);v.currentTime=(index<0?c.frame_source_times.length-1:index)/c.frame_rate;}
  else v.currentTime=Math.max(0,target-c.offset);
 };
 v.ontimeupdate=()=>{const t=sourceTime();a.querySelector('output').textContent='Source time '+t.toFixed(3)+'s';for(const row of rows)row.classList.toggle('active',t>=Number(row.dataset.start)&&t<=Number(row.dataset.end));};
}
</script></html>'''
(HERE / 'videos.html').write_text(page)
assert len(cards) == 9 and len({c['id'] for c in cards}) == 9
for card in cards:
    assert (HERE / card['media']).is_file()
    assert all(r['end_seconds'] >= r['start_seconds'] >= 0 for r in card['intervals'])
print(json.dumps({'clips':len(cards),'ctc_assertions':'passed','early_duplicate_windows':early_duplicate['window_count'],
                  'rejected_windows':evidence['rejected_windows'],'protected_test_accessed':False}))
