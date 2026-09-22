"""Bounded runtime-matched Reel head adaptation; frozen BIO, no live promotion."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, subprocess, sys, time, traceback
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from active.v17.approved_phrase_data_v17 import digest, verify_manifest
from scripts.train_temporal_boundary_v17 import atomic, CONFIDENT, SOURCE, COMBINED, event_key, safe_path
from scripts.evaluate_temporal_boundary_v17 import arguments, observations, edit_counts, summarize
OUT = ROOT / 'artifacts/reports/reel_context_adapt_v17_20260922'
CACHE = ROOT / 'artifacts/cache/reel_context_adapt_v17_20260922'
MODELS = ROOT / 'artifacts/models/reel_context_adapt_v17_20260922'
LOCAL = ROOT / 'artifacts/reports/boundary_local_calibration_v17_20260922'
RECIPE = ROOT / 'active/v17/reel_context_recipe_20260922.json'
PROPOSAL = ROOT / 'artifacts/models/stage1_v17_phrase_adapt_reel_v2/best_model.pth'
VERIFIER = ROOT / 'artifacts/models/stage1_v17_unified_phrase_activity_adapt_reel_v2/best_model.pth'
FEATURES = ('proposal_features', 'landmark_features', 'hand_features', 'landmark_logits', 'hand_logits')


def crop_bounds(event, neighbours, context):
    start, end = event['start_seconds'], event['end_seconds']
    if not np.isfinite([start, end]).all() or end <= start:
        raise ValueError('invalid annotated interval')
    others = [e for e in neighbours if e is not event and
              (e['start_seconds'], e['end_seconds']) != (start, end)]
    if any(e['start_seconds'] < end and e['end_seconds'] > start for e in others):
        return None
    left = max([0.] + [e['end_seconds'] for e in others if e['end_seconds'] <= start])
    right = min([end + context] + [e['start_seconds'] for e in others if e['start_seconds'] >= end])
    return max(left, start - context), min(right, end + context)


def accepted_candidate(candidate, baseline):
    return (candidate['wer'] < baseline['wer'] and candidate['correct'] >= baseline['correct']
            and candidate['insertions'] <= baseline['insertions']
            and candidate['retained_baseline_correct_events'] >= .98 * baseline['correct'])


def models():
    from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
    from active.v17.export_unified_multimodal_coreml_v17 import load_model
    p = torch.load(PROPOSAL, map_location='cpu', weights_only=False)
    proposal = SLTStage1V17(Stage1V17Config(**p['model_config']))
    proposal.load_state_dict(p['model_state_dict'], strict=True)
    verifier, v = load_model(VERIFIER)
    if p['label_to_index'] != v['label_to_index']:
        raise ValueError('proposal/verifier label mismatch')
    return proposal.eval(), verifier.eval(), p, v


def manifest():
    canonical = verify_manifest()
    confident, source, combined = [json.loads(p.read_text()) for p in (CONFIDENT, SOURCE, COMBINED)]
    assert digest(SOURCE) == confident['provenance']['source_manifest_sha256']
    source_rows = {(r['source'], r['role'], r['source_item_id']): r for r in source['rows']}
    admitted_o5 = {(r['role'], r['source_item_id'], round(r['start_seconds'], 3),
                    round(r['end_seconds'], 3), r['canonical_label'])
                   for r in combined['records'] if r['source'] == 'o5s5'}
    local = json.loads((LOCAL / 'manifest.json').read_text())
    reserved = {r['video_sha256'] for rows in local['phases'].values() for r in rows}
    unique, physical_roles, signer_roles = {}, defaultdict(set), defaultdict(set)
    for r in sorted(confident['rows'], key=lambda r: (r['source'] != 'asllrp_contiguous', r['identity'])):
        if r['source'] not in ('asllrp_contiguous', 'asllrp_other_ctc', 'o5s5'):
            continue
        parent = r['source_item_id'] if r['source'] == 'o5s5' else r['source_item_id'].split(':')[1]
        physical_roles[parent].add(r['role'])
        signer_roles[(r['source'] == 'o5s5', r['signer_id'])].add(r['role'])
        if r['target_kind'] != 'known' or not r['source_crop_complete']:
            continue
        key = event_key(r, source_rows)
        if r['source'] == 'o5s5' and key not in admitted_o5:
            continue
        if key in unique:
            if unique[key]['role'] != r['role']:
                raise ValueError('duplicate event crosses roles')
            continue
        source_row = source_rows[(r['source'], r['role'], r['source_item_id'])]
        if source_row['video_sha256'] in reserved:
            raise ValueError('reserved calibration/confirmation video in training source')
        if crop_bounds(r, source_row['intervals'], .1) is None:
            continue
        unique[key] = {**r, 'video_sha256': source_row['video_sha256'],
                       'neighbours': source_row['intervals'], 'physical_parent': parent}
    if any(len(v) != 1 for v in physical_roles.values()) or any(len(v) != 1 for v in signer_roles.values()):
        raise ValueError('continuous source parent/signer crosses original roles')
    rows = [r for r in unique.values() if r['role'] == 'train']
    if not rows:
        raise ValueError('no admitted training events')
    files = [CONFIDENT, SOURCE, COMBINED, LOCAL/'manifest.json', LOCAL/'calibration.json',
             LOCAL/'confirmation.json', PROPOSAL, VERIFIER, Path(__file__), OUT/'PLAN.md',
             ROOT/'scripts/live_isolated_v17.py', ROOT/'scripts/live_reel_stage1_v17.py',
             ROOT/'scripts/live_boundary_v17.py']
    recipe = dict(format='reel_context_adaptation_v1', canonical=canonical, training_ready=True,
                  files={str(p.relative_to(ROOT)): digest(p) for p in files}, events=rows,
                  event_counts=dict(Counter(r['source'] for r in rows)), seed=17621,
                  contexts=[0., .1, .2], maximum_epochs=80, patience=10, learning_rate=1e-4,
                  replay_fraction=.5, kl_weight=2., batch_size=128,
                  scope='Reviewed complete positive intervals only; no equal-partition local targets, no blank/OOV labels. Original source roles preserved.',
                  adaptation='Proposal classifier and verifier fusion head only; all temporal/image encoders frozen. No architecture change.',
                  selection='Whole-video local calibration WER; no correct-count loss/no extra insertions/98% retained, isolated validation <=1 percentage point loss per model/domain. Epoch0 fallback.',
                  limitations='Familiar-signer reused development. Baseline lineage includes historical local adaptation. Not a new signer generalization claim.',
                  promoted=False)
    atomic(RECIPE, recipe)
    return recipe


def verify(recipe):
    assert verify_manifest()['sha256'] == recipe['canonical']['sha256']
    for name, sha in recipe['files'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('pinned file changed: ' + name)


@torch.inference_mode()
def encode(provider, proposal, verifier):
    x = torch.from_numpy(provider['landmarks']).float()
    _, pf = proposal(x, return_embeddings=True)
    ll, lf = verifier.landmark_model(x, return_embeddings=True)
    hf = verifier.hand_model.forward_features(torch.from_numpy(provider['hand_embeddings']).float(),
         torch.from_numpy(provider['hand_valid']) > .5, torch.from_numpy(provider['hand_boxes']).float())
    hl = verifier.hand_model.classifier(hf)
    return {k: t.detach().cpu()[0].clone() for k, t in zip(FEATURES, (pf, lf, hf, ll, hl))}


class Capture:
    def __init__(self, inner):
        self.inner, self.provider, self.output = inner, None, None
    def predict(self, provider):
        self.provider = {k: np.array(v, copy=True) for k, v in provider.items()}
        self.output = self.inner.predict(provider)
        return self.output


def prepare_window(obs, reel, proposal, verifier):
    # Invoke the actual classifier preprocessing; capture its existing Core ML inputs.
    reel.orientation.provider = reel.full.stage1.provider = None
    p, v = reel.classify(obs), reel.verify(obs)
    if reel.orientation.provider is None or reel.full.stage1.provider is None:
        return None
    provider = reel.full.stage1.provider
    np.testing.assert_array_equal(provider['landmarks'], reel.orientation.provider['landmarks'])
    sample = encode(provider, proposal, verifier)
    with torch.inference_mode():
        pl, vl = forward(proposal.classifier, verifier.fusion_head, {k: x[None] for k, x in sample.items()})
    core_p = np.asarray(next(iter(reel.orientation.output.values()))).reshape(-1)
    core_v = np.asarray(reel.full.stage1.output['var_2927']).reshape(-1)
    # FP16 Core ML verifier may differ slightly; decisions are checked separately.
    np.testing.assert_allclose(pl.numpy()[0], core_p, atol=.06, rtol=.02)
    np.testing.assert_allclose(vl.numpy()[0], core_v, atol=.06, rtol=.02)
    return dict(features=sample, proposal=p, verifier=v, proposal_teacher=pl[0].clone(), verifier_teacher=vl[0].clone())


def pack(samples):
    if not samples:
        raise ValueError('empty prepared data')
    return {k: torch.stack([s['features'][k] for s in samples]) for k in FEATURES}


def forward(p, v, x):
    return p(x['proposal_features']), v(*(x[k] for k in FEATURES[1:]))


def decision(logits, old, labels, args):
    probability = logits.softmax(-1).numpy()
    order = np.argsort(probability)[::-1]
    score, margin = float(probability[order[0]]), float(probability[order[0]] - probability[order[1]])
    reasons = [r for r in old['rejection_reasons'] if r not in ('low_score', 'low_margin')]
    if score < args.minimum_score: reasons.append('low_score')
    if margin < args.minimum_margin: reasons.append('low_margin')
    return dict(candidate_gloss=labels[int(order[0])], model_score=score, accepted=not reasons,
                rejection_reasons=reasons)


def replay(pl, vl, prepared, labels, args):
    from scripts.live_reel_stage1_v17 import VerifiedCommitLock
    results = []
    for record in prepared['records']:
        hyp = []
        for index in record['indices']:
            old = prepared['samples'][index]
            p, v = decision(pl[index], old['proposal'], labels, args), decision(vl[index], old['verifier'], labels, args)
            agreement = p['model_score'] if p['candidate_gloss'] == v['candidate_gloss'] else 0.
            if (p['accepted'] and v['accepted'] and
                VerifiedCommitLock(args.commit_hits, args.instant_commit_score).update(v['candidate_gloss'],
                    max(v['model_score'], agreement), proposal=p['candidate_gloss'],
                    proposal_score=p['model_score'], minimum_score=args.commit_score)):
                hyp.append(v['candidate_gloss'])
        results.append(dict(id=record['id'], reference=record['reference'], hypothesis=hyp,
                            metrics=edit_counts(record['reference'], hyp)))
    return results


def prepare(recipe):
    from scripts.live_reel_stage1_v17 import ReelCascadeClassifier
    from active.v17.extract_v17 import AppleVisionDetector
    started = time.perf_counter()
    proposal, verifier, pc, vc = models()
    args = arguments('unused', OUT/'unused'); args.no_motion_trim = True
    reel = ReelCascadeClassifier(args)
    reel.args.no_motion_trim = reel.full.args.no_motion_trim = True
    reel.orientation, reel.full.stage1 = Capture(reel.orientation), Capture(reel.full.stage1)
    detector = AppleVisionDetector(args.minimum_point_confidence)
    labels = [k for k, v in sorted(pc['label_to_index'].items(), key=lambda kv: kv[1])]
    if labels != reel.labels: raise ValueError('runtime label order differs')
    CACHE.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for event in recipe['events']: grouped[event['video_path']].append(event)
    samples, identities, skipped, pins = [], [], [], {}
    for number, (video, events) in enumerate(grouped.items()):
        row = events[0]
        if digest(safe_path(video)) != row['video_sha256']: raise ValueError('video changed: ' + video)
        pins[video] = row['video_sha256']
        archive = safe_path(row['archive_path'])
        with np.load(archive, allow_pickle=False) as z: meta = json.loads(str(z['metadata_json'].item()))
        if meta['video_sha256'] != row['video_sha256'] or meta['source_item_id'] != row['source_item_id']:
            raise ValueError('source archive/video mismatch')
        pins[row['archive_path']] = digest(archive)
        obs = observations(row, args, detector)
        for event in events:
            assert event['label'] in pc['label_to_index']
            seen = set()
            for context in recipe['contexts']:
                bounds = crop_bounds(event, event['neighbours'], context)
                if bounds in seen: continue
                seen.add(bounds)
                chosen = [o for o in obs if bounds[0] <= o.seconds <= bounds[1]]
                sample = prepare_window(chosen, reel, proposal, verifier) if len(chosen) >= 4 else None
                if sample is None:
                    skipped.append(dict(identity=event['identity'], context=context, reason='insufficient detected frames')); continue
                samples.append(sample); identities.append(dict(identity=event['identity'], context=context,
                    target=pc['label_to_index'][event['label']], video=video, bounds=bounds))
        if (number + 1) % 25 == 0:
            print(f'prepare continuous {number+1}/{len(grouped)} videos; {len(samples)} windows', flush=True)
    train = dict(x=pack(samples), y=torch.tensor([r['target'] for r in identities]),
                 pt=torch.stack([s['proposal_teacher'] for s in samples]),
                 vt=torch.stack([s['verifier_teacher'] for s in samples]), identities=identities)
    torch.save(train, CACHE/'continuous.pt')
    atomic(OUT/'continuous_audit.json', dict(windows=len(samples), distinct_events=len(set(r['identity'] for r in identities)),
           skipped=skipped, files=pins, seconds=time.perf_counter()-started))
    prepare_replay(proposal, verifier, pc, vc, pins)
    phase_rows = json.loads((LOCAL/'manifest.json').read_text())['phases']
    for phase in ('calibration', 'confirmation'):
        saved = json.loads((LOCAL/(phase+'.json')).read_text())['records']['frozen_min3']
        rows = {r['source_item_id']: r for r in phase_rows[phase]}
        phase_samples, records = [], []
        for record in saved:
            row = rows[record['id']]
            assert digest(safe_path(row['video_path'])) == row['video_sha256']
            pins[row['video_path']] = row['video_sha256']
            obs = observations(row, args, detector); indices = []
            for prediction in record['predictions']:
                chosen = [o for o in obs if prediction['start_seconds']-.1 <= o.seconds <= prediction['end_seconds']+.1]
                sample = prepare_window(chosen, reel, proposal, verifier) if len(chosen) >= 4 else None
                if sample is None:
                    if prediction.get('committed_gloss'): raise ValueError('baseline committed missing input')
                    continue
                indices.append(len(phase_samples)); phase_samples.append(sample)
            records.append(dict(id=record['id'], reference=record['reference'], indices=indices))
        prepared = dict(samples=phase_samples, records=records, x=pack(phase_samples), baseline=saved)
        # Confirmation features are prepared but never used for checkpoint selection.
        if phase == 'calibration':
            with torch.inference_mode(): pl, vl = forward(proposal.classifier, verifier.fusion_head, prepared['x'])
            actual = replay(pl, vl, prepared, labels, args)
            if [r['hypothesis'] for r in actual] != [r['hypothesis'] for r in saved]:
                atomic(OUT/'baseline_mismatch.json', dict(actual=actual, expected=saved))
                raise ValueError('Torch/runtime baseline recognition mismatch; training blocked')
            atomic(OUT/'baseline_parity.json', dict(videos=len(actual), summary=summarize(actual, saved), passed=True))
        torch.save(prepared, CACHE/(phase+'.pt'))
    for p in CACHE.glob('*.pt'): pins[str(p.relative_to(ROOT))] = digest(p)
    atomic(OUT/'cache_manifest.json', dict(files=pins, preparation_seconds=time.perf_counter()-started))
    print('preparation complete', time.perf_counter()-started, flush=True)


def prepare_replay(proposal, verifier, pc, vc, pins):
    from active.v17.extract_hand_rgb_supplement_v17 import selection_items
    from active.v17.extract_hand_rgb_semlex_val_v17 import validation_items
    from active.v17.train_stage_1_v17 import load_v17_archive
    from active.v17.schema_v17 import V17Config, schema_fingerprint
    # Existing cached verifier encoders must be tensor-identical, not merely same architecture.
    for name, current in [('landmark', verifier.landmark_model), ('hand', verifier.hand_model)]:
        path = ROOT/vc['source_checkpoints'][name]['path']
        assert digest(path) == vc['source_checkpoints'][name]['sha256']
        old = torch.load(path, map_location='cpu', weights_only=False)['model_state_dict']
        for k, value in current.state_dict().items(): torch.testing.assert_close(value, old[k], rtol=0, atol=0)
        pins[str(path.relative_to(ROOT))] = digest(path)
    for domain in ('citizen', 'semlex'):
        for split in ('train', 'val'):
            cache = ROOT/f'artifacts/generated/unified_multimodal_student_v17/{domain}_{split}.npz'
            assert digest(cache) == vc['training_data_provenance']['cache_sha256'][f'{domain}_{split}']
            pins[str(cache.relative_to(ROOT))] = digest(cache)
            with np.load(cache, allow_pickle=False) as z:
                data = {k: torch.from_numpy(z[k].copy()).float() for k in FEATURES[1:]}
                targets, ids = z['targets'].copy(), z['item_ids'].copy()
            mapping = None
            if domain == 'semlex':
                path = ROOT/('data/local/semlex_citizen100_train_audit/full_clean_train_candidates.json' if split == 'train'
                            else 'data/local/semlex_citizen100_val_audit/selection_plan.json')
                items, _ = selection_items(path, 'semlex') if split == 'train' else validation_items(path)
                mapping = {f'{i.label}/{i.item_id}': i.landmark_path for i in items}
                pins[str(path.relative_to(ROOT))] = digest(path)
            pooled = []
            for start in range(0, len(ids), 64):
                batch = []
                for index in range(start, min(start+64, len(ids))):
                    item = str(ids[index]); label = item.split('/')[0]
                    assert pc['label_to_index'][label] == int(targets[index])
                    path = mapping[item] if mapping is not None else ROOT/f'data/local/citizen100_v17/landmarks/{split}/{item}.v17.npz'
                    path = path.resolve(); safe_path(str(path))
                    batch.append(load_v17_archive(path, schema_fingerprint(V17Config())))
                    pins[str(path.relative_to(ROOT))] = digest(path)
                with torch.inference_mode(): _, values = proposal(torch.stack(batch), return_embeddings=True)
                pooled.append(values.clone())
            data['proposal_features'] = torch.cat(pooled)
            with torch.inference_mode(): pt, vt = forward(proposal.classifier, verifier.fusion_head, data)
            torch.save(dict(x=data, y=torch.tensor(targets).long(), pt=pt.clone(), vt=vt.clone(), ids=ids.tolist()), CACHE/f'{domain}_{split}.pt')
            print('prepared isolated', domain, split, len(ids), flush=True)


def load_cache(name):
    return torch.load(CACHE/(name+'.pt'), map_location='cpu', weights_only=False)


def train(recipe):
    verify(recipe)
    cm = json.loads((OUT/'cache_manifest.json').read_text())
    for name, sha in cm['files'].items():
        if digest(ROOT/name) != sha: raise ValueError('cache/source changed: '+name)
    torch.manual_seed(recipe['seed']); np.random.seed(recipe['seed'])
    proposal, verifier, pc, vc = models()
    p, v = proposal.classifier, verifier.fusion_head
    labels = [k for k, _ in sorted(pc['label_to_index'].items(), key=lambda kv: kv[1])]
    args = arguments('unused', OUT/'unused')
    cal = load_cache('calibration'); baseline = summarize(cal['baseline'], cal['baseline'])
    domains = {d: load_cache(d+'_val') for d in ('citizen', 'semlex')}
    base_acc = {d: [float((x.argmax(-1)==c['y']).float().mean()) for x in (c['pt'], c['vt'])] for d,c in domains.items()}
    continuous = load_cache('continuous'); isolated = [load_cache(d+'_train') for d in domains]
    data = {k: torch.cat([c['x'][k] for c in [continuous]+isolated]) for k in FEATURES}
    y, pt, vt = [torch.cat([c[k] for c in [continuous]+isolated]) for k in ('y','pt','vt')]
    n = len(continuous['y']); weights = torch.cat([torch.full((n,),.5/n), torch.full((len(y)-n,),.5/(len(y)-n))])
    optimizer = torch.optim.AdamW(list(p.parameters())+list(v.parameters()), lr=recipe['learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, recipe['maximum_epochs'], eta_min=1e-6)
    MODELS.mkdir(parents=True, exist_ok=True)
    history=[]; best_wer=baseline['wer']; best_epoch=0; stale=0; best_trained=None; started=time.perf_counter()
    for epoch in range(1, recipe['maximum_epochs']+1):
        tick=time.perf_counter(); p.train(); v.train(); losses=[]
        indices=torch.multinomial(weights, max(2*n, len(y)), replacement=True)
        for batch in indices.split(recipe['batch_size']):
            pl,vl=forward(p,v,{k:t[batch] for k,t in data.items()})
            loss=F.cross_entropy(pl,y[batch],label_smoothing=.05)+F.cross_entropy(vl,y[batch],label_smoothing=.05)
            loss+=recipe['kl_weight']*(F.kl_div(F.log_softmax(pl/2,-1),F.softmax(pt[batch]/2,-1),reduction='batchmean')+
                                       F.kl_div(F.log_softmax(vl/2,-1),F.softmax(vt[batch]/2,-1),reduction='batchmean'))*4
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(p.parameters())+list(v.parameters()),1.)
            optimizer.step(); losses.append(float(loss.detach()))
        scheduler.step(); p.eval();v.eval()
        with torch.inference_mode():
            pl,vl=forward(p,v,cal['x']); records=replay(pl,vl,cal,labels,args); summary=summarize(records,cal['baseline'])
            accuracy={d:[float((z.argmax(-1)==c['y']).float().mean()) for z in forward(p,v,c['x'])] for d,c in domains.items()}
        isolated_ok=all(accuracy[d][j]>=base_acc[d][j]-.01 for d in domains for j in (0,1))
        eligible=accepted_candidate(summary,baseline) and isolated_ok
        row=dict(epoch=epoch,loss=float(np.mean(losses)),summary=summary,isolated_accuracy=accuracy,
                 eligible=eligible,seconds=time.perf_counter()-tick)
        history.append(row); print(json.dumps(row),flush=True)
        state=dict(epoch=epoch,proposal_head=copy.deepcopy(p.state_dict()),verifier_head=copy.deepcopy(v.state_dict()),
                   recipe_sha256=digest(RECIPE),summary=summary,isolated_accuracy=accuracy)
        if best_trained is None or summary['wer']<best_trained['summary']['wer']:
            best_trained=state; torch.save(state,MODELS/'best_trained_heads.pth'); stale=0
        else: stale+=1
        if eligible and summary['wer']<best_wer:
            best_wer=summary['wer'];best_epoch=epoch;torch.save(state,MODELS/'selected_heads.pth')
        atomic(OUT/'history.json',dict(baseline=baseline,isolated_baseline=base_acc,epochs=history,selected_epoch=best_epoch))
        if epoch==1:
            atomic(OUT/'timing.json',dict(first_epoch_seconds=row['seconds'],maximum_training_minutes=row['seconds']*recipe['maximum_epochs']/60,
                   likely_patience_minutes=row['seconds']*(recipe['patience']+1)/60,preparation_seconds=cm['preparation_seconds']))
        if stale>=recipe['patience']: break
    confirmation=None
    if best_epoch:
        state=torch.load(MODELS/'selected_heads.pth',map_location='cpu',weights_only=False)
        p.load_state_dict(state['proposal_head']);v.load_state_dict(state['verifier_head']);p.eval();v.eval()
        confirmed=load_cache('confirmation')
        with torch.inference_mode(): pl,vl=forward(p,v,confirmed['x'])
        records=replay(pl,vl,confirmed,labels,args); confirmation=summarize(records,confirmed['baseline'])
        atomic(OUT/'confirmation.json',dict(records=records,summary=confirmation,baseline=summarize(confirmed['baseline'],confirmed['baseline'])))
        # Save ordinary full checkpoints; no app/default/Core ML replacement.
        pc['model_state_dict']=proposal.state_dict();pc['reel_context_adaptation']=state
        vc['head_state_dict']=verifier.fusion_head.state_dict();vc['reel_context_adaptation']=state
        torch.save(pc,MODELS/'proposal_candidate.pth');torch.save(vc,MODELS/'verifier_candidate.pth')
    result=dict(status='complete',selected_epoch=best_epoch,epochs=len(history),training_seconds=time.perf_counter()-started,
                baseline=baseline,best_trained=best_trained['summary'],confirmation=confirmation,promoted=False)
    atomic(OUT/'completion.json',result)
    (OUT/'REPORT.md').write_text('# Reel context adaptation\n\n'+('Candidate selected; no deployment.' if best_epoch else 'No candidate passed retention gates; frozen models retained.')+
       '\n\nRuntime-matched reviewed sign crops; proposal classifier and verifier fusion adapted. Frozen encoders, frozen BIO windows, unchanged acceptance. '
       'Citizen/SemLex train replay and validation retention; protected tests untouched.\n\n'+
       f"Completed {len(history)} epochs in {result['training_seconds']/60:.2f} minutes after {cm['preparation_seconds']/60:.2f} minutes preparation. Selected epoch {best_epoch}.\n\n"+
       '| Readout | WER | Correct | Extra words |\n|---|---:|---:|---:|\n'+
       ''.join(f"| {name} | {s['wer']:.2%} | {s['correct']} | {s['insertions']} |\n" for name,s in [('Baseline',baseline),('Best trained (not necessarily eligible)',best_trained['summary'])])+
       '\nSee history.json for retention and isolated-domain results; completion.json for confirmation. This is reused familiar-signer development, not independent generalization evidence. '
       'No distillation or deployment. Temporal backbone adaptation remains untested in this bounded head phase.\n')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--prepare',action='store_true');parser.add_argument('--train',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    try:
        recipe=json.loads(RECIPE.read_text()) if RECIPE.exists() else manifest()
        verify(recipe)
        if args.run or args.prepare:
            if (OUT/'cache_manifest.json').exists(): raise FileExistsError('prepared cache exists; use --train')
            prepare(recipe)
        if args.run or args.train: train(recipe)
        subprocess.run(['osascript','-e','display notification "Reel adaptation finished. Read the report." with title "SLT"'],check=False,capture_output=True)
    except Exception:
        atomic(OUT/'failure.json',dict(traceback=traceback.format_exc()))
        subprocess.run(['osascript','-e','display notification "Reel adaptation stopped. Read failure.json." with title "SLT"'],check=False,capture_output=True)
        raise


if __name__=='__main__': main()
