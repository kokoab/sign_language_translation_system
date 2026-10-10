from pathlib import Path
import json,hashlib
O=Path(__file__).resolve().parent;R=O.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
c=O.parent/'temporal_repair_contract_v17_20261011/contract.json'
recipe=dict(format='slt_temporal_repair_training_recipe_v2',status='declared_before_training_pending_full_input_verification',recipe_training_ready=False,canonical_training_ready=False,
 contract_path=str(c),contract_sha256=sha(c),canonical_phrase_manifest_sha256=sha(R/'active/v17/approved_phrase_manifest_20260921_v2.json'),
 base=str(R/'artifacts/reports/go_split_root_cause_v17_20261009/run_frag25_reject/span_recognizer/best_model.pth'),
 authorization='User approved full native extraction and bounded training experiment after reviewing temporal repair preparation results.',
 admission='New recipe version; only pinned positive events from the reviewed contract plus exact prior-admitted isolated replay caches. Not a canonical training flag change or historical mixed trainer bypass.',
 exclusions=['phone development recordings','protected test media/features','unlabeled gaps and uncertain boundary labels','synthetic fragments/rejects','automatic forced-alignment labels'],
 training=dict(seeds=[27927,27928],head_learning_rates=[0.00003,0.0001],epochs=6,samples_per_epoch=6000,batch_size=64,optimizer='AdamW',weight_decay=.001,scheduler='CosineAnnealingLR(T_max=6)',temporal_share=.30,isolated_share=.70,
 sampling='Equal class mass within each source; equal source-domain isolated mass; temporal equal class then equal event then equal eligible view. Source roles preserved.',
 loss='CE(all samples) + KL(incumbent teacher on isolated replay, temperature=2, multiplied by T^2) + .01 * sum(mean squared head parameter deviation)',
 frozen='Both complete encoders in eval mode, byte-identical weights; precompute full temporal encoder outputs once, with direct-model parity checks.',gradient_clip=1.0),
 selection=dict(epoch_zero=True,retention='Every represented class in each Citizen/SemLex/local validation domain must retain >= incumbent correct count; all100Citizen classes required. No aggregate regression.',
 objective='Among eligible checkpoints: maximize number of validation source events correct on all eligible views, then minimize equal-event mean cross entropy; retain earlier on exact tie.',
 secondary='Save one best temporal diagnostic candidate ignoring retention, clearly ineligible if it fails; never promote it.',phone='Post-selection diagnosis only. Never select hyperparameters/checkpoints using the known phone failures.'),
 evaluation=dict(all_classes='isolated domain per-class counts and native570cached streaming cases',development='native MY/HAVE/September30 phase sweeps plus genuine LIKE controls; separate incumbent-runtime and contact/continuous-runtime arms',promotion='Fresh independent normal-phone evaluation required after all development gates; no automatic install from this experiment'),
 remaining_admission_checks=['all138parent outputs complete','finite schema-valid arrays','core hand/crop eligibility with exclusions recorded','no parent/signer split overlap within source','full training and feature-cache hash pins','executable trainer hash and negative gate tests'])
(O/'declared_recipe.json').write_text(json.dumps(recipe,indent=2)+'\n');print('Declared four bounded arms, before training; full inputs still pending.')
