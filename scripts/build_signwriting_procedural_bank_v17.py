#!/usr/bin/env python3
"""Build 100 manual cores from selected FSW notation and official hand photos."""
from pathlib import Path
import argparse
import json
import sys
import unicodedata

import numpy as np

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from active.v17.avatar_rig_v17 import bone_length_metrics,parse_signwriting_signbox
from active.v17.signwriting_motion_v17 import load_hand_templates,generate_manual_rig
from scripts.render_rigged_avatar_v17 import sha256


def candidate_score(parsed,phonology):
    symbols=parsed['symbols'];hands=[s for s in symbols if s['category']=='hand']
    motions=[s for s in symbols if s['category'] in ('movement','finger_movement')]
    contacts=any(s['category']=='contact' for s in symbols)
    names=['' if not s else unicodedata.name(chr(0x1d800+int(s['base'][1:],16)-0x100)) for s in motions]
    sides={int(s['key'][5],16)>=8 for s in hands}
    two=phonology.get('sign_type')!='OneHanded'
    score=(4 if hands and ((len(sides)>=2)==two) else -4)
    score+=3 if contacts==(phonology.get('contact')=='1') else -2
    major=phonology.get('major_location')
    score+=2 if (major=='Head')==any(s['category']=='head_face' for s in symbols) else 0
    if major=='Hand':score+=2 if len(hands)>=2 else -2
    movement=phonology.get('movement')
    keywords={'Straight':('STRAIGHT','FLICK','SQUEEZE'),'Curved':('CURVE','BEND','WRIST FLEX'),
              'Circular':('CIRCLE','ROTATION'),'BackAndForth':('ALTERNATING','MULTIPLE'),
              'X-shaped':('CROSS',),'None':()}.get(movement,())
    if movement=='None':score+=2 if not motions else -1
    elif any(any(key in name for key in keywords) for name in names):score+=2
    repeated=any(any(key in name for key in ('MULTIPLE','DOUBLE','TRIPLE','ALTERNATING')) for name in names)
    score+=2 if repeated==(phonology.get('repeated_movement')=='1') else -1
    if phonology.get('wrist_twist')=='1':score+=2 if any('ROTATION' in name for name in names) else -2
    return score-.03*len(symbols)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,default=Path('artifacts/reports/signwriting_gloss_bank_v17_v5b'))
    parser.add_argument('--reference-bank',type=Path,default=Path('artifacts/reports/avatar_gloss_bank_v17_v1'))
    parser.add_argument('--templates',type=Path,default=Path('artifacts/reports/iswa_hand_templates_v17_v3/templates.npz'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    inventory=json.loads((args.inventory/'report.json').read_text())
    reference=json.loads((args.reference_bank/'report.json').read_text())
    manifest_path=Path('active/v17/citizen100_manifest.json');manifest=json.loads(manifest_path.read_text())
    if inventory['manifest_sha256']!=sha256(manifest_path) or reference['manifest_sha256']!=sha256(manifest_path):
        raise ValueError('frozen manifest mismatch')
    views,bases=load_hand_templates(args.templates)
    references={x['entry']['gloss']:x for x in reference['items']}
    inv={x['gloss']:x for x in inventory['notation_inventory']}
    preserved={x['entry']['gloss']:x for x in inventory['items']}
    overrides={'NO':'AS13f10S22114M516x513S13f10487x498S22114484x487'}
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(format='slt_avatar_gloss_bank_v17',symbol_motion=True,procedural_signwriting=True,
        manifest_sha256=sha256(manifest_path),lexicon_sha256=inventory['lexicon_sha256'],
        dictionary_sha256=inventory['dictionary_sha256'],templates_sha256=sha256(args.templates),
        code_sha256=sha256(Path(__file__)),items=[],failures=[],training_eligible=False,human_accepted=False,
        limitations=['All cores are driven by selected FSW symbols; automatic selections need fluent review',
                     'Official single-view hand photos provide handshape depth estimates',
                     'Manual motion families are procedural; facial and mouth symbols are recorded but not animated',
                     'Source videos provide comparison frames only, never landmark trajectories'])
    for cls in manifest['classes']:
        gloss=cls['canonical_label'];row=inv[gloss];phonology=row['locked_phonology'];reference_item=references[gloss]
        try:
            if gloss in preserved:
                item=dict(preserved[gloss]);old=args.inventory/item['audit'];path=args.output/item['audit']
                path.write_bytes(old.read_bytes());item['audit_sha256']=sha256(path)
                item['selection_reason']='explicit_symbolic_profile_preserved'
            else:
                choices=[]
                for candidate in row['candidates']:
                    parsed=candidate['notation']
                    hand_bases={s['base'] for s in parsed['symbols'] if s['category']=='hand'}
                    if hand_bases and hand_bases<=set(bases):choices.append((candidate_score(parsed,phonology),candidate))
                if gloss in overrides:
                    candidate=next(c for c in row['candidates'] if c['fsw']==overrides[gloss]);reason='manual_override_no_head-only_variant'
                elif choices:
                    _,candidate=max(choices,key=lambda value:value[0]);reason='phonology_scored_dictionary_candidate'
                else:raise ValueError('no hand-template-compatible dictionary candidate')
                parsed=candidate['notation'];repeated=phonology.get('repeated_movement')=='1'
                names=[_name(s['base']) for s in parsed['symbols'] if s['category'] in ('movement','finger_movement')]
                frames=36 if repeated or any('CIRCLE' in name or 'ALTERNATING' in name for name in names) else 24
                rig=generate_manual_rig(parsed,phonology,views,bases,frames)
                old=args.reference_bank/reference_item['audit']
                if sha256(old)!=reference_item['audit_sha256'] or sha256(Path(reference_item['source']))!=reference_item['source_sha256']:
                    raise ValueError('reference comparison hash mismatch')
                with np.load(old) as data:
                    indices=np.rint(np.linspace(data['source_indices'][0],data['source_indices'][-1],frames)).astype(int)
                path=args.output/f'{gloss.lower()}_audit.npz'
                np.savez_compressed(path,shoulders=rig.shoulders,elbows=rig.elbows,symbol_hands=rig.hands,
                                    hand_states=rig.hand_states,source_indices=indices)
                item=dict(entry=dict(gloss=gloss,fsw=candidate['fsw']),source=reference_item['source'],
                    source_sha256=reference_item['source_sha256'],class_index=cls['class_index'],
                    raw_gloss=cls['citizen_raw_gloss'],asl_lex_code=cls['citizen_asl_lex_code'],fps=30,
                    core_start_frame=0,core_stop_frame=frames,motion_origin='signwriting_procedural',
                    hand_participation=[bool(np.any(rig.hand_states[:,s]=='symbolic')) for s in range(2)],
                    selection_reason=reason,candidate_score=candidate_score(parsed,phonology),
                    unanimated_symbols=[s for s in parsed['symbols'] if s['category'] in ('head_face','body','dynamics')],
                    audit=path.name,audit_sha256=sha256(path),**bone_length_metrics(rig))
            report['items'].append(item);print(gloss, item['selection_reason'],flush=True)
        except (ValueError,OSError,KeyError) as error:
            report['failures'].append(dict(gloss=gloss,error=str(error)));print(gloss,'FAILED',error,flush=True)
        report.update(coverage=len(report['items']),requested_classes=len(manifest['classes']))
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=not report['failures'] and report['coverage']==100
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['failures']:raise SystemExit(f'{len(report["failures"])} failures')


def _name(base):
    return unicodedata.name(chr(0x1d800+int(base[1:],16)-0x100))


if __name__=='__main__':main()
