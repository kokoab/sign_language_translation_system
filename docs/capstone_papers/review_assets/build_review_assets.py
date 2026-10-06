"""Build local review figures from recorded evidence; never runs model evaluation."""
from pathlib import Path
import hashlib
import json
import re
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BLUE, TEAL, GRAY, ORANGE = '#245b85', '#218276', '#718096', '#b86b28'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 160, 'svg.fonttype': 'none'})
SOURCES = {}


def read(path):
    p = ROOT / path
    raw = p.read_bytes()
    SOURCES[path] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def save(fig, name, note='', *, tight=True):
    # Explanatory notes belong in the Markdown document, never in the image.
    if tight:
        fig.tight_layout(rect=(0, 0, 1, .97))
    for ext in ('png', 'svg'):
        fig.savefig(HERE / f'{name}.{ext}', bbox_inches='tight')
    plt.close(fig)


def bars(ax, labels, values, title, xlabel, percent=False, highlight=-1):
    colors = [GRAY] * len(values)
    if highlight is not None:
        colors[highlight] = TEAL
    ax.barh(range(len(values)), values, color=colors, height=.6)
    ax.set_yticks(range(len(values)), labels)
    ax.invert_yaxis()
    ax.set_title(title, loc='left', fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, 110 if percent else max(values) * 1.23)
    for i, v in enumerate(values):
        ax.text(v + (1 if percent else max(values) * .025), i,
                f'{v:.2f}' if v < 100 else f'{v:.0f}', va='center', fontsize=9)
    ax.grid(axis='x', alpha=.15)
    ax.set_axisbelow(True)


def family_chart(families):
    selected = ['BiLSTM', 'Temporal CNN', 'Flat Transformer',
                'Part-wise + global Squeezeformer']
    rows = [next(r for r in families if r['display_name'] == name) for name in selected]
    names = ['BiLSTM', 'Temporal CNN', 'Flat Transformer', 'ATLAS (Squeezeformer)']
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    bars(axes[0], names, [r['validation']['top1'] for r in rows],
         'Recognition model families', 'Top-1 (%)', True)
    bars(axes[1], names, [r['latency']['median_ms'] for r in rows],
         'Time per prediction', 'Median milliseconds')
    save(fig, 'families')


def language_screening_chart(candidates):
    tags = ['t5_tiny_v2_kv_fp16', 'flan_t5_base_kv_fp16']
    rows = [next(r for r in candidates if r['tag'] == tag) for tag in tags]
    names = ['ATLAS tiny T5 / KV FP16', 'FLAN-T5-base / KV FP16']
    fig, ax = plt.subplots(figsize=(9, 3.1))
    bars(ax, names, [r['est_session_p90_28_ms'] for r in rows],
         'Language-model execution screening',
         'Estimated milliseconds for 28 output tokens', highlight=0)
    save(fig, 'language_latency')


def phone_profile_chart(profiles):
    fig, ax = plt.subplots(figsize=(11, 4.3))
    bars(ax, ['Mixed precision / CPU + GPU recognition',
              'FP16 models / CPU + GPU recognition',
              'FP16 models / Neural Engine enabled for recognition'],
         [r['median'] for r in profiles[:3]],
         'Combined recognition processing on iPhone 13',
         'Median preparation-and-recognition time per frame (ms)')
    save(fig, 'phone_speed')


def comparison_charts():
    old = ROOT / 'artifacts/reports/capstone1_v17_revision_checklist_v1/README.md'
    SOURCES[str(old.relative_to(ROOT))] = hashlib.sha256(old.read_bytes()).hexdigest()
    # Transcribed from the named tables in the historical reviewed report.
    extractor = {'labels': ['MediaPipe', 'Apple Vision'], 'top1': [89.95, 93.12],
                 'seconds': [1.230, .678], 'clips_quality': 300, 'clips_accuracy': 378}
    modality = {'labels': ['Hand images', 'Landmarks', 'Combined'],
                'top1': [80.69, 95.50, 96.30], 'top5': [94.71, 98.68, 99.21],
                'clips': 378}
    latency = read('artifacts/reports/capstone1_v17_revision_checklist_v1/architecture_latency_benchmark.json')
    keys = ['graph_part_replacement', 'wider_flat_squeezeformer_d384',
            'flat_squeezeformer_d256', 'partwise_global_squeezeformer']
    arch = {'labels': ['Graph replacement', 'Wider flat', 'Flat', 'Part-wise + global'],
            'top1': [78.31, 95.24, 95.77, 96.83],
            'median_ms': [latency['results'][k]['median_ms'] for k in keys]}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    bars(axes[0], extractor['labels'], extractor['top1'], 'Matched classifier accuracy', 'Top-1 (%)', True)
    bars(axes[1], extractor['labels'], extractor['seconds'], 'Extraction time', 'Median seconds per clip')
    save(fig, 'extractors', 'Recorded landmark comparison; extraction timing measured on the development computer.')
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    bars(ax, modality['labels'], modality['top1'], 'Complementary recognition inputs', 'Top-1 (%)', True)
    save(fig, 'modalities', 'Recognition inputs evaluated on the same validation recordings.')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bars(axes[0], arch['labels'], arch['top1'], 'Landmark architecture accuracy', 'Top-1 (%)', True)
    bars(axes[1], arch['labels'], arch['median_ms'], 'Time per prediction', 'Median milliseconds')
    save(fig, 'architecture', 'Base-model comparison; single-thread CPU timings with prepared landmark input.')
    family = read('artifacts/reports/capstone1_v17_revision_checklist_v1/stage1_family_benchmark/result.json')
    fam = [{k: r[k] for k in ('display_name', 'parameters', 'best_epoch', 'validation', 'latency')}
           for r in family['results']]
    family_chart(fam)
    swift = read('artifacts/reports/phone_speed_v17_20260929/score_test_C.json')
    baseline = read('artifacts/reports/boundary_expanded_eval_v17_20260922/evaluation.json')
    row = next(r for r in baseline['runs'] if r['arm'] == 'frozen_pretrained_bio')
    records = row['records']
    wer = sum(r['metrics'][k] for r in records for k in ('substitutions','deletions','insertions')) / sum(r['metrics']['references'] for r in records)
    stream = [dict(label='Boundary-guided interval classification', overall={'wer':wer}),
              dict(label='ATLAS streaming recognition', overall=swift['overall'])]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars(ax, [r['label'] for r in stream], [100*r['overall']['wer'] for r in stream],
         'Streaming recognition configurations', 'Word error rate (%) — lower is better', True)
    save(fig, 'streaming', 'Same recorded sequences; complete recognition pipelines. Processing conditions: ATLAS_MEASUREMENT_NOTES.md.')
    english = {'labels': ['Whole-sequence generation', 'Incremental generation'],
               'fully_right': [60,60], 'wrong': [6,8], 'bleu': [75,74],
               'negation_drops': [0,0], 'sessions':300, 'judge':'DeepSeek'}
    p = ROOT / 'artifacts/reports/stage3_multisentence_bakeoff_v17_20260929/REPORT.md'
    SOURCES[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    bars(axes[0], english['labels'], english['fully_right'], 'Automatically judged fully correct', 'Sessions (%)', True)
    bars(axes[1], english['labels'], english['wrong'], 'Automatically judged wrong', 'Sessions (%) — lower is better', True)
    save(fig, 'english_quality', 'Automatic English assessment using generated references and a language-model evaluator.')
    lm = read('artifacts/reports/stage3_latency_bench_v17_20260929/summary.json')
    # Same rows as the benchmark report; model configurations differ intentionally.
    specs = [('t5_tiny_v2_kv_fp16', 'Tiny T5 / KV FP16'),
             ('t5_small_kv_fp16', 'T5-small / KV FP16'),
             ('flan_t5_small_kv_fp16', 'FLAN-T5-small / KV FP16'),
             ('flan_t5_base_kv_fp16', 'FLAN-T5-base / KV FP16')]
    lm_rows = []
    for tag, label in specs:
        matches = [r for r in lm if r['tag'] == tag]
        if not matches:
            continue
        r = min(matches, key=lambda x: x['est_session_p90_28_ms'])
        lm_rows.append(dict(label=label, **r))
    language_screening_chart(lm_rows)
    log = ROOT / 'artifacts/reports/phone_speed_v17_20260929/profile_1.log'
    profiles = []
    with log.open() as f:
        for line in f:
            m = re.match(r'PROFILE (\S+) frames=(\d+) total_median_ms=([\d.]+) total_p90_ms=([\d.]+)', line)
            if m:
                profiles.append(dict(name=m[1], frames=int(m[2]), median=float(m[3]), p90=float(m[4])))
    hasher = hashlib.sha256()
    with log.open('rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    SOURCES[str(log.relative_to(ROOT))] = hasher.hexdigest()
    chosen = [profiles[i] for i in (0, 1, 2)]
    phone_profile_chart(chosen)
    return dict(extractor=extractor, modality=modality, architecture=arch, families=fam,
                streaming=stream, english=english, language_latency=lm_rows, phone_profiles=profiles)


def training_charts():
    p = 'artifacts/generated/kaggle_stage1_partwise_kokoab_pull_v1/stage1_v17_partwise_v2/history.json'
    hist = read(p)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [r['epoch'] for r in hist]
    axes[0].plot(epochs, [r['train_loss'] for r in hist], label='Training objective', color=BLUE)
    axes[0].plot(epochs, [r['loss'] for r in hist], label='Validation loss', color=TEAL)
    axes[0].set_ylabel('Recorded loss'); axes[0].legend(frameon=False)
    axes[1].plot(epochs, [r['top1'] for r in hist], color=BLUE, label='Validation top-1')
    axes[1].plot(epochs, [r['top5'] for r in hist], color=TEAL, label='Validation top-5')
    axes[1].set_ylabel('Accuracy (%)'); axes[1].set_ylim(0, 103); axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xlabel('Epoch'); ax.axvline(100, color=ORANGE, linestyle='--', alpha=.8)
    fig.suptitle('Landmark recognizer: training and validation')
    save(fig, 'recognition_training', 'Landmark recognition training; selected checkpoint marked.')
    spans = read('artifacts/models/span_recognizer_v17_local_a/history.json')['history']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    trained = [r for r in spans if r['epoch'] > 0]
    axes[0].plot([r['epoch'] for r in trained], [r['loss'] for r in trained], '-o', color=BLUE)
    axes[0].set_ylabel('Training objective'); axes[0].set_xlabel('Epoch')
    axes[1].plot([r['epoch'] for r in spans], [r['tune']['wer'] * 100 for r in spans], '-o', color=TEAL)
    axes[1].set_ylabel('Tuning word error rate (%)'); axes[1].set_xlabel('Epoch (0 = initial model)')
    fig.suptitle('Span recognizer: recorded adaptation history')
    save(fig, 'span_training', 'Sign-interval adaptation; sequence errors measured with the training study’s fixed decoder settings.')
    hist = read('artifacts/reports/stage3_multisentence_bakeoff_v17_20260929/tiny/training_result.json')['history']
    fig, ax = plt.subplots(figsize=(8, 4))
    for key, label, color in [('train_loss', 'Training loss', BLUE), ('validation_loss', 'Validation loss', TEAL)]:
        ax.plot([r['step'] for r in hist], [r[key] for r in hist], '-o', color=color, label=label)
    ax.set_xlabel('Optimizer step'); ax.set_ylabel('Recorded loss'); ax.legend(frameon=False)
    ax.set_title('T5-efficient-tiny: training and validation')
    save(fig, 'english_training', 'Recorded training and validation loss for the English-generation model.')
    return {'recognition_epochs': len(epochs), 'span_observations': len(spans), 'english_observations': len(hist)}


def flow(name, title, nodes, edges, figsize, limits=(12, 12)):
    """Render paper-ready vector/bitmap diagrams with explicit manual layout."""
    mermaid = ['flowchart TD']
    for key, (_, _, label, kind) in nodes.items():
        mermaid.append(f'    {key}' + ('{' if kind == 'decision' else '[')
                       + json.dumps(label) + ('}' if kind == 'decision' else ']'))
    for a, b, label in edges:
        mermaid.append(f'    {a} -->' + (f'|{json.dumps(label)}| ' if label else ' ') + b)
    (HERE / f'{name}.mmd').write_text('\n'.join(mermaid) + '\n')
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, limits[0]); ax.set_ylim(-.25, limits[1]); ax.axis('off')
    for key, (x, y, label, kind) in nodes.items():
        w, h = (3.5, 1.45) if kind == 'decision' else (3.35, 1.0)
        if kind == 'decision':
            patch = Polygon([(x-w/2,y),(x,y+h/2),(x+w/2,y),(x,y-h/2)],
                            facecolor='#fff1dd', edgecolor=ORANGE, linewidth=1.2)
        else:
            patch = FancyBboxPatch((x-w/2,y-h/2), w,h,boxstyle='round,pad=0.04,rounding_size=.10',
                                   facecolor='#e8f1f7' if kind != 'output' else '#e7f3ef',
                                   edgecolor=BLUE if kind != 'output' else TEAL, linewidth=1.2)
        ax.add_patch(patch)
        ax.text(x,y,'\n'.join(textwrap.wrap(label, 28 if kind != 'decision' else 22)),
                ha='center',va='center',fontsize=9.5)
    for a,b,label in edges:
        x,y,_,kind = nodes[a]; xx,yy,_,kk = nodes[b]
        dx, dy = xx-x, yy-y
        def offset(dx, dy, kind):
            w, h = (1.75, .725) if kind == 'decision' else (1.675, .5)
            if kind == 'decision':
                t = 1 / (abs(dx)/w + abs(dy)/h)
            else:
                t = min(w/abs(dx) if dx else float('inf'), h/abs(dy) if dy else float('inf'))
            return dx*t, dy*t
        ox, oy = offset(dx, dy, kind)
        tx, ty = offset(-dx, -dy, kk)
        start=(x+ox, y+oy); stop=(xx+tx, yy+ty)
        ax.add_patch(FancyArrowPatch(start,stop,arrowstyle='-|>',mutation_scale=12,
                                    linewidth=1.2,color='#4b5966'))
        if label:
            ax.text((start[0]+stop[0])/2+.1,(start[1]+stop[1])/2+.12,label,
                    fontsize=8,ha='center',bbox=dict(fc='white',ec='none',pad=1))
    fig.suptitle(title,fontsize=14,fontweight='bold')
    save(fig,name)


def diagrams():
    flow('system_flow','ATLAS: Live recognition and English output',{
        'camera':(6,11.2,'Start Live camera','box'),
        'vision':(6,9.9,'Landmark extractor: locate signing landmarks','box'),
        'features':(6,8.6,'Landmark motion and MobileCLIP2 hand-image features','box'),
        'recognition':(6,7.3,'Streaming Sign Recognition','box'),
        'buffer':(6,6,'Recognized glosses','box'),
        'english':(6,4.7,'T5-efficient-tiny: incremental English generation','box'),
        'finish':(6,3.4,'Finish button or held-open-palms gesture','box'),
        'finalize':(6,2.1,'Finalize English text','box'),
        'output':(6,.8,'English text and local speech output','output')},
        [('camera','vision',''),('vision','features',''),('features','recognition',''),
         ('recognition','buffer',''),('buffer','english',''),('english','finish',''),
         ('finish','finalize',''),('finalize','output','')],(8,13))
    flow('app_flow','ATLAS: mobile application navigation',{
        'open':(8,23,'Open ATLAS','box'),
        'home':(8,21,'Home: choose an activity','decision'),
        'live':(2,18.5,'Live: camera preview','box'),
        'start':(2,16.8,'Start recognition','box'),
        'sign':(2,15.1,'Sign and view words; Reset clears input','box'),
        't5':(2,13.4,'T5-efficient-tiny: incremental English generation','box'),
        'spoken':(2,8.3,'English text and local speech output','output'),
        'finish':(2,11.7,'Finish button or held-open-palms gesture','box'),
        'english':(2,10,'Finalize English text','output'),
        'continue':(2,6.6,'Continue signing (repeat) or Stop','box'),
        'saved':(2,4.9,'Session saved on the device','box'),
        'livehome':(2,3.2,'Home / another activity','output'),
        'glosses':(6,18.5,'Glosses','box'),
        'browse':(6,16.5,'Browse and select a sign','box'),
        'example':(6,14.5,'Watch the sign demonstration','box'),
        'back':(6,12.5,'Back to Glosses or Home','output'),
        'practice':(10,18.5,'Practice: select 5, 10 or 20 signs','box'),
        'targets':(10,16.5,'Start a random practice set','box'),
        'reference':(10,14.5,'View target sign and demonstration','box'),
        'attempt':(10,12.5,'Sign for the camera','box'),
        'correct':(10,10.5,'Matches target?','decision'),
        'retry':(14,10.5,'Try again or Skip','box'),
        'next':(10,8.5,'Advance to next sign','box'),
        'complete':(10,6.5,'Round complete?','decision'),
        'more':(14,6.5,'Show next target and repeat the attempt','box'),
        'result':(10,4.5,'Show score and skipped signs','output'),
        'again':(10,2.5,'Practice again or return Home','box'),
        'history':(14,18.5,'History','box'),
        'sessions':(14,16.5,'View saved session dates, duration and text','box'),
        'historyhome':(14,13.2,'Home / another activity','output')},
        [('open','home',''),('home','live','Live'),('home','glosses','Glosses'),
         ('home','practice','Practice'),('home','history','History'),
         ('live','start',''),('start','sign',''),('sign','t5',''),('t5','finish',''),('finish','english',''),
         ('english','spoken',''),('spoken','continue',''),('continue','saved','Stop'),('saved','livehome',''),
         ('glosses','browse',''),('browse','example',''),('example','back','Back'),
         ('example','reference','Practice this sign'),('practice','targets',''),('targets','reference',''),
         ('reference','attempt',''),('attempt','correct',''),('correct','next','Yes'),
         ('correct','retry','No'),('retry','attempt','Retry'),('retry','next','Skip'),
         ('next','complete',''),('complete','more','No'),('complete','result','Yes'),('result','again',''),
         ('history','sessions',''),('sessions','historyhome','')],(18,22),limits=(16,24))
    flow('runtime_decisions','ATLAS: recognition decisions during signing',{
        'input':(4,11,'Landmarks and hand-image features','box'),
        'boundary':(4,9.4,'Boundary estimates and candidate intervals','box'),
        'scores':(4,7.8,'Squeezeformer recognition and combined scores','box'),
        'decode':(4,6.2,'Segmental decoder selects sign and rest intervals','box'),
        'accept':(4,4.35,'Commit conditions met?','decision'),
        'wait':(9,4.35,'Keep observing the next frames','box'),
        'words':(4,2.5,'Commit output and manage gloss buffer','output'),
        'language':(4,.9,'Update English; Finish finalizes remaining text','output')},
        [('input','boundary',''),('boundary','scores',''),('scores','decode',''),
         ('decode','accept',''),('accept','wait','No'),('accept','words','Yes'),
         ('words','language','')],(10,11))
    flow('engineering_decisions','ATLAS: engineering selection rationale',{
        'goal':(6,11,'On-device sign recognition and English output','box'),
        'visual':(3,9,'Compare landmark extractors','box'),
        'language':(9,9,'Evaluate English-output quality','box'),
        'apple':(3,7,'Landmarks with complementary hand-image features','box'),
        'tiny':(9,7,'T5-efficient-tiny with incremental generation','box'),
        'temporal':(3,5,'Compare temporal recognition architectures','box'),
        'coreml':(9,5,'Export models for on-device inference','box'),
        'selected':(3,3,'Multimodal Squeezeformer and segmental decoding','box'),
        'app':(6,1,'Integrated mobile application','output')},
        [('goal','visual',''),('goal','language',''),('visual','apple','Local comparison'),
         ('language','tiny','English quality'),('apple','temporal',''),('tiny','coreml',''),
         ('temporal','selected','Local comparison'),('selected','app',''),('coreml','app','')],(11,10))


def sequence_timeline():
    source = 'artifacts/reports/phone_speed_v17_20260929/score_test_C.json'
    record_id = 'local:HELLO_HOW_YOU:ff187c3f'
    record = next(r for r in read(source)['records'] if r['id'] == record_id)
    assert record['reference'] == record['hypothesis'] == ['HELLO', 'HOW', 'YOU']
    words = record['words']
    colors = [BLUE, TEAL, ORANGE]
    import cv2
    video = 'data/raw_videos/PHRASES FIXED/HELLO_HOW_YOU/ff187c3f.mp4'
    SOURCES[video] = hashlib.sha256((ROOT / video).read_bytes()).hexdigest()
    capture = cv2.VideoCapture(str(ROOT / video))
    if not capture.isOpened():
        raise ValueError(f'Cannot read timeline video: {video}')
    fps = capture.get(cv2.CAP_PROP_FPS)
    fig = plt.figure(figsize=(11, 6.5))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], hspace=.6)
    ax = fig.add_subplot(grid[0, :])
    ax.set_title('HELLO HOW YOU: recognition timeline', loc='left',
                 fontsize=13, fontweight='bold', pad=16)
    ax.set_xlim(0, 3.2)
    ax.set_ylim(-.4, 1.65)
    ax.set_yticks([1, 0], ['Boundary detector\n+ decoder', 'Recognition model'])
    ax.tick_params(axis='y', length=0, pad=12)
    ax.set_xlabel('Time in recording (seconds)')
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', alpha=.15)
    ax.set_axisbelow(True)
    stills = []
    for index, (word, color) in enumerate(zip(words, colors)):
        start, end = word['start_seconds'], word['end_seconds']
        mid = (start + end) / 2
        ax.barh(1, end-start, left=start, height=.3, color=color,
                alpha=.25, edgecolor=color)
        ax.text(mid, 1.25, f'{start:.2f}–{end:.2f} s', ha='center',
                va='bottom', fontsize=10, color='#354458')
        ax.annotate('', xy=(mid, .28), xytext=(mid, .8),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.3))
        ax.barh(0, end-start, left=start, height=.42, color=color)
        ax.text(mid, 0, word['gloss'], ha='center', va='center',
                color='white', fontweight='bold', fontsize=10)
        frame_index = round(mid * fps)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f'Cannot decode frame {frame_index} from {video}')
        still_ax = fig.add_subplot(grid[1, index])
        still_ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        still_ax.set_title(f"{word['gloss']} · {frame_index / fps:.2f} s",
                           fontsize=10, color=color, fontweight='bold')
        still_ax.axis('off')
        stills.append(dict(gloss=word['gloss'], frame_index=frame_index,
                           seconds=frame_index / fps))
    capture.release()
    save(fig, 'hello_how_you_timeline', tight=False)
    return dict(source=source, record_id=record_id,
                intervals=[{k: w[k] for k in ('gloss', 'start_seconds', 'end_seconds')}
                           for w in words],
                axis='recording seconds; selected model intervals',
                video=video, video_fps=fps, stills=stills)



if __name__ == '__main__':
    metrics = comparison_charts()
    metrics['training'] = training_charts()
    diagrams()
    metrics['sequence_timeline'] = sequence_timeline()
    metrics['source_sha256'] = SOURCES
    (HERE / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(f'Wrote {len(list(HERE.glob("*.png")))} PNG/SVG figure pairs and metrics.json')
