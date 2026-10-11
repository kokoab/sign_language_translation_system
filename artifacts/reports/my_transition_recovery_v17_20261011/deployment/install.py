from pathlib import Path
import json,subprocess
D=Path(__file__).resolve().parent
v=json.loads((D/'release_verification.json').read_text());assert v['all_pass']
def run(cmd,name):
 (D/(name+'_command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
 with (D/(name+'.log')).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 print(name,p.returncode,flush=True)
 if p.returncode:print((D/(name+'.log')).read_text()[-2500:])
 p.check_returncode()
device='00008110-00111D1A0130A01E';bundle='com.kokoab.sltMobileApp'
run(['xcrun','devicectl','device','install','app','--device',device,'--json-output',str(D/'installation.json'),v['app']],'install')
run(['xcrun','devicectl','device','process','launch','--terminate-existing','--device',device,'--json-output',str(D/'launch.json'),bundle],'launch')
run(['xcrun','devicectl','device','info','processes','--device',device,'--json-output',str(D/'processes.json')],'processes')
