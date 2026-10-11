"""Run the prepared diagnostic only when unlocked; restore the normal app afterward."""
from pathlib import Path
import subprocess,json,time
D=Path(__file__).resolve().parent
DEVICE='00008110-00111D1A0130A01E';BUNDLE='com.kokoab.sltMobileApp'
def run(cmd,name,check=True):
 (D/(name+'_command.json')).write_text(json.dumps(cmd,indent=2))
 with (D/(name+'.log')).open('w') as out:return subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,check=check)
lock=D/'phone_lock_check.json'
run(['xcrun','devicectl','device','info','lockState','--device',DEVICE,'--json-output',str(lock)],'phone_lock_check')
x=json.loads(lock.read_text())
def lock_values(obj):
 if isinstance(obj,dict):
  return ([obj['passcodeRequired']] if 'passcodeRequired' in obj else [])+[v for item in obj.values() for v in lock_values(item)]
 if isinstance(obj,list):return [v for item in obj for v in lock_values(item)]
 return []
values=lock_values(x)
if not values or any(v is not False for v in values):raise SystemExit('iPhone locked or lock state unknown: diagnostic was not installed or launched. Unlock then rerun.')
assert json.loads((D/'phone_build_status.json').read_text())['returncode']==0
build=Path('/Users/frnzlo/slt_my_transition_recovery_build/Build/Products/Runner_iphoneos27.0-arm64.xctestrun')
normal=Path('/Users/frnzlo/slt_ordered_my_like_gap175_build/Build/Products/Release-iphoneos/Runner.app')
assert build.exists() and normal.exists()
try:
 result=run(['xcodebuild','test-without-building','-xctestrun',str(build),'-destination','id='+DEVICE,'-only-testing:RunnerTests/RunnerTests/testMyTransitionRecoveryComparison','-resultBundlePath',str(D/('phone_comparison_'+str(int(time.time()))+'.xcresult'))],'phone_comparison',False)
 for index,arm in enumerate(['baseline','candidate','candidate','baseline']):
  name=f'my_transition_{arm}_{index}.json'
  run(['xcrun','devicectl','device','copy','from','--device',DEVICE,'--source','Documents/'+name,'--destination',str(D/name),'--domain-type','appDataContainer','--domain-identifier',BUNDLE],'pull_'+name,False)
 (D/'phone_test_status.json').write_text(json.dumps(dict(returncode=result.returncode),indent=2))
finally:
 run(['xcrun','devicectl','device','install','app','--device',DEVICE,str(normal)],'restore_normal_app')
 run(['xcrun','devicectl','device','process','launch','--device',DEVICE,BUNDLE],'restore_normal_launch',False)
