from pathlib import Path
import json

O=Path(__file__).resolve().parent
load=lambda name:json.loads((O/name).read_text())
evaluation=load('evaluation_summary.json');training=load('training_results.json');material=load('materialization.json')['summary']
assert load('verification.json')['passed']
lines=['Temporal repair experiment — completed 2026-10-11',
       '','Decision: no candidate is approved for installation. The installed model and app sources are unchanged.',
       '', 'This experiment tests the accepted bounded repair: retain the installed encoders, adapt only the word fusion head using reviewed native temporal inputs, preserve the existing letter path, then check the whole vocabulary and known phone failures. It does not establish a general solution to MY/HAVE.',
       '', 'Completed work',
       f"- All 138 parent videos / {material['events_attempted']} events extracted and audited. {material['views_eligible']}/{material['views_requested']} phase/capture/context views eligible; 192 short-core views excluded; no event lost.",
       '- Training events: MY 28, HAVE 59, LIKE 48. Validation events: HAVE 4, LIKE 5; no independent MY validation. Multiple views of an event are correlated, not new examples.',
       '- Local MY and phone contact boundaries were visually reviewed by Codex, not independently confirmed by an ASL expert or the signer. Source-provided labels and roles were preserved.',
       '- Four declared runs: seeds 27927/27928 × learning rates 3e-5/1e-4, six epochs each. Both complete encoders frozen; 30% temporal / 70% isolated replay. All source data, roles, caches, code and recipe pinned.',
       '- 432 native video replays: three models × two runtime variants × 72 cases. Each model/runtime has 24 schedules per source video (8 auxiliary phases × 3 capture offsets).',
       '- 1710 cached streaming replays:570 per model, including all 100 Citizen classes. 177 genuine MY/LIKE controls:59 per model.',
       '', 'Training and checkpoint selection',
       'Model | Citizen | SemLex | local | Temporal validation events correct on every view',
]
for name in ['incumbent','selected','diagnostic']:
    row=training['baseline'] if name=='incumbent' else training[name]['row'];m=row['metrics']
    lines.append(f"{name} | {m['citizen_val']['correct']}/378 | {m['semlex_val']['correct']}/953 | {m['local_val']['correct']}/2626 | {m['temporal_validation']['all_views_correct_events']}/9")
lines += ['',f"Selected: {training['selected']['arm']}, epoch {training['selected']['row']['epoch']}; passes the predeclared per-class isolated retention rule.",
          f"Diagnostic: {training['diagnostic']['arm']}, epoch {training['diagnostic']['row']['epoch']}; ineligible: SemLex HELP and MORNING each lose one correct example, local DAY loses one. Aggregate gains do not erase these regressions.",
          '', 'Native known-phone development results',
          'Model / runtime | September30 exact sentence | first October MY core | HAVE reviewed-core target-only overlaps | HAVE immediately followed by BIG | runs with all seven HAVE→BIG contacts covered correctly']
for name in ['incumbent','selected','diagnostic']:
    for runtime in ['control','continuous_contact']:
        r=evaluation[name][runtime];s=r['sept30'];m=r['oct11'];h=r['have_take11']
        lines.append(f"{name} / {runtime} | {s['exact']}/{s['n']} | {m['target_only']}/{m['n']} | {h['target_only']}/{h['n']} | {h['target_then_following_sign']}/{h['n']} | {h['all_events_target_then_following_runs']}/24")
lines += ['', 'MY/HAVE counts match predicted word spans to conservative visually reviewed contact interiors. An overlap attributed to BIG means the decoder span covers the HAVE contact with BIG; it is not necessarily an additional emitted BIG. These counts are development diagnostics, not independent sign accuracy. All seven HAVE contacts are followed by BIG; the ambiguous aborted attempt and later uncertain MY/LIKE switch remain unlabeled. Full per-event outcomes are in evaluation_summary.json.',
          '', 'Whole-vocabulary streaming and genuine controls',
          'Model | Citizen exact | pair exact | lab tune exact | genuine MY exact | genuine LIKE exact | LIKE clips containing MY']
for name in ['incumbent','selected','diagnostic']:
    v=evaluation[name]['validation'];g=evaluation[name]['genuine_controls']
    lines.append(f"{name} | {v['citizen_val']['exact']}/378 | {v['pair_val']['exact']}/103 | {v['tune']['exact']}/89 | {g['MY']['exact']}/22 | {g['LIKE']['exact']}/37 | {g['LIKE']['contains_MY']}/37")
    lines.append(f"  Citizen classes: {v['citizen_class_changes']}; individual rows in {name}_validation.per_class.csv.")
lines += ['', 'Citizen uses its official validation split. SemLex/local retain their historical admitted roles and familiar-signer limitations; this is not a blanket signer-disjoint evaluation. The 570 cached checks use the control runtime and identical existing traces, including the previously appended held-tail frames on isolated clips. They establish matched regression results, not new live-camera accuracy. The contact runtime is a separate development prototype and is not granted broad safety from the control-runtime table. Pools overlap and must not be pooled as independent samples.',
          '', 'What the experiment establishes',
          '- Stronger adaptation introduces false HAVE outputs in 10/25 ANGRY, 5/36 TIRED and 6/38 HAPPY clips, versus zero in each incumbent set. These include insertions as well as substitutions; do not describe every occurrence as a one-for-one relabel. Pair exact drops 58→47/103.',
          '- Passing isolated per-class retention did not preserve streaming behavior: selected loses four exact Citizen clips, adding GO after BIG, MOTHER after DRINK, LIKE after FEEL, and YESTERDAY after THANKYOU. Its isolated Citizen predictions were unchanged. One MY NAME lab sentence improves, but the phone MY still requires its own gate.',
          '- Temporal supervision can substantially change HAVE: on the reviewed phone HAVE core, incumbent and selected classify all 72 views as ANGRY; the stronger diagnostic classifies 66 as HAVE and 6 as ANGRY.',
          '- The same training does not repair the reviewed phone MY inputs. All three checkpoints give 55 MY / 17 LIKE on 72 cropped-core views, and only 6 MY top1 decisions on 62 overlapping windows saved from the actual phone failure. These are correlated diagnostics, not 62 independent signs.',
          '- Holding the phone video, model, phase and capture offset fixed, changing only the allowed left context changes MY/LIKE in 7 of 24 schedules. MY wins are 16/24, 18/24 and 21/24 for the three context views. This directly demonstrates window-context sensitivity; it does not prove a boundary-only repair will be sufficient.',
          '- The incumbent already recognizes 98.5% of the admitted training MY views averaged equally by event; the diagnostic reaches 100%. Training success on these 28 events therefore does not demonstrate coverage of the failing phone MY/NAME transition.',
          '- Removing forced-alignment omissions was valid data repair, but this experiment does not support claiming those omissions alone caused the deployed regression or that head-only continuation fixes it. Short-window placement/capture phase and missing representative transition supervision remain material.',
          '- No admitted HAVE→BIG training pair, new MY FRIEND supervision or independent phone MY set was created. Known phone failures were used only after checkpoint selection, never as training examples.',
          '', 'Export and source preservation',
          '- Both26.557 MB FP16 packages preserve all 378 PyTorch top1 word/letter predictions. Original letter weights and their original word-score input branch stay fixed.',
          '- Actual candidate CoreML letter logits are bit-identical to the installed CoreML outputs on all 378 checked inputs, with zero threshold disagreements. This bounds the letter-preservation evidence to those inputs.',
          '- Both encoder state dictionaries are unchanged. Production Python and Swift hashes match the pre-experiment snapshots. No app installation or physical-phone accuracy/latency test is claimed.',
          '', 'Audit corrections and execution failures',
          '- Rounded decimal ffprobe timestamps invalidated the previous720/720 core-eligibility claim. Exact source-frame PTS/time_base gives 624/720 eligible pilot views and 240/288 for the four recovered MY events. Source frame annotations were preserved. Full extraction excludes 192 short MY views and retains all 144 events.',
          '- The stale 978-row SemLex validation replay cache contained 25 raw-video duplicates of training. The finalized source-role manifest already excludes those 25 from validation. This run uses 1,388 train / 953 validation, zero retained raw overlap; source caches unchanged. Earlier temporary train-exclusion plan was superseded before training.',
          '- Initial MPS watermark configuration failed and was corrected before feature-cache completion. O5S5 Doug video had 180-degree display metadata: explicit pixel rotation was validated; identity outputs remain exact. An initially upside-down OpenCV reference was corrected before accepting the orientation check.',
          '- Long-source extraction performance was corrected through report-local context reuse and one-pass raw-detection caching. Fresh tracking and both auxiliary request signatures remain per stream.432 pilot + 144 long-source views matched every metadata row and all four tensors bit-for-bit. One obsolete suspended extractor was terminated after its verified replacement completed; no jobs remain suspended.',
          '', 'Boundaries and next justified action',
          'Do not install the stronger checkpoint or repeat this same core-only head recipe as a claimed fix. The next repair must address the real proposed-window/transition mismatch, with independently reviewed MY→NAME/MY→FRIEND and HAVE→BIG timing, representative phone signing, and a fresh reserved phone evaluation. A class-independent temporal/boundary change must pass the same MY/LIKE and all 100 class controls; these results do not justify a blanket MY geometry override.',
          'Canonical 494 manifest and original preparation contract remain training_ready=false. The dedicated pinned v2 recipe admits only this completed bounded experiment. Official Citizen test and protected lab test were not used for training, inference or selection; protected metadata was used only for exclusion.',
          '', 'Evidence',
          'training_results.json; training_recipe.json; materialization.json; isolated_per_class.csv; evaluation_summary.json; *_validation.per_class.csv; cached_head_summary.json; development_core_summary.json; actual_fp16_letter_preservation.json; verification.json; code_checks.json.',
]
(O/'REPORT.txt').write_text('\n'.join(lines)+'\n')
print('Completed report written:',O/'REPORT.txt')
