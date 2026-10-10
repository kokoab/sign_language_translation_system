from pathlib import Path
import ast,hashlib,json
N=Path(__file__).resolve().parent
checks={}
for arm in ['localA','legacy_base','legacy_guards','rank']:
 for kind,n in [('validation',570),('genuine',59),('new_video',8)]:
  p=N/(arm+'_'+kind+'.json');checks[arm+'_'+kind]=p.exists() and len(json.loads(p.read_text())['records'])==n
for arm in ['base','guards']:
 p=N/('legacy_'+arm+'_have.json');checks['legacy_'+arm+'_have8']=p.exists() and len(json.loads(p.read_text())['records'])==8
p=N/'rank_have.json';checks['rank_have8']=p.exists() and len(json.loads(p.read_text())['records'])==8
p=N/'localA_phase.json';checks['phase72']=p.exists() and len(json.loads(p.read_text())['records'])==72
checks['current_new_video8']=len(json.loads((N/'current_new_video.json').read_text())['records'])==8
commands=[json.loads(p.read_text()) for p in N.glob('*_command.json')]
checks['fresh_outputs']=all(p.exists() and p.stat().st_mtime>=c.stat().st_mtime for c in N.glob('*_command.json') for p in [c.with_name(c.name.replace('_command.json','.json'))])
checks['matched_letter_threshold']=bool(commands) and all(c['environment']['LETTER_THRESHOLD']=='0.5' for c in commands)
pins=json.loads((N/'production_pins.json').read_text());checks['production_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in pins.items())
for p in N.glob('*.py'):ast.parse(p.read_text())
checks['scripts_parse']=True
checks['invalid_runs_quarantined']=(N/'invalid_letter_threshold_095').is_dir()
(N/'verification.json').write_text(json.dumps({'experiment_complete':all(checks.values()),'checks':checks,'release_approved':False,'reason':'Paired-sign failures remain; regression results must not be treated as a shipping approval.'},indent=2)+'\n')
print(json.dumps(checks,indent=2));assert all(checks.values())
