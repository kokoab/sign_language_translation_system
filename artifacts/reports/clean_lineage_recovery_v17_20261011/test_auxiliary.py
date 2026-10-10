from pathlib import Path
import shutil,subprocess,os,json
O=Path(__file__).resolve().parent;P=O.parent/'lineage_runtime_investigation_v17_20261011';d=O/'auxiliary';d.mkdir(exist_ok=True)
for p in (O/'continuous_contact').glob('*.swift'):shutil.copy2(p,d/p.name)
p=d/'LiveReelEngine.swift';s=p.read_text();assert 'let vision = LiveVision()' in s
s=s.replace('let vision = LiveVision()','let vision = LiveVision(auxiliaryInterval: Int(ProcessInfo.processInfo.environment["AUX_INTERVAL"] ?? "8") ?? 8)');p.write_text(s)
cmd=['xcrun','swiftc','-O','-o',str(O/'auxiliary_probe')]+[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','main.swift']]
with (O/'auxiliary_build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
cases=json.load(open(P/'cases.json'))
for interval in [1,4]:
 subset=[r for r in cases if r['phase']<interval]
 casefile=O/f'auxiliary_{interval}_cases.json';casefile.write_text(json.dumps(subset))
 with (O/f'auxiliary_{interval}.log').open('w') as f:subprocess.run([str(O/'auxiliary_probe'),str(O/'resources'),str(casefile),str(O/f'auxiliary_{interval}.json')],env=dict(os.environ,FINISH='0',LETTER_THRESHOLD='0.7',AUX_INTERVAL=str(interval)),stdout=f,stderr=subprocess.STDOUT,check=True)
 print('auxiliary',interval,'complete',flush=True)
# Repeat the two actual phone processing schedules used by the previous load experiment.
casefile=O/'schedule_cases.json';casefile.write_text(json.dumps([r for r in cases if r['id']=='oct11_my_name_phase0']))
for name in ['candidate_paced','shared_candidate_paced']:
 times=O.parent/'my_have_phone_v17_20261011'/f'{name}_times.json'
 with (O/(name+'_aux1.log')).open('w') as f:subprocess.run([str(O/'auxiliary_probe'),str(O/'resources'),str(casefile),str(O/(name+'_aux1.json'))],env=dict(os.environ,FINISH='0',LETTER_THRESHOLD='0.7',AUX_INTERVAL='1',LIVE_TIMES=str(times)),stdout=f,stderr=subprocess.STDOUT,check=True)
 print(name,'complete',flush=True)
