from pathlib import Path
import json, os, subprocess, time

O = Path(__file__).resolve().parent
for name in ['selected', 'diagnostic']:
    model = f'SpanRecognizerV17Temporal{name.title()}B8FP16'
    link = O/'resources'/(model+'.mlpackage')
    if not link.exists(): link.symlink_to((O/(model+'.mlpackage')).resolve(), target_is_directory=True)
    for suffix, binary, cases, count in [
        ('validation', O/'validation_probe', O.parent/'my_have_joint_experiment_v17_20261011/validation_traces.json', 570),
        ('genuine_controls', O/'control_probe', O/'genuine_control_cases.json', 59),
    ]:
        start = time.time()
        output = O/f'{name}_{suffix}.json'
        with (O/f'{name}_{suffix}.log').open('w') as log:
            subprocess.run([str(binary),str(O/'resources'),str(cases),str(output)],env=dict(os.environ,MODEL=model,LETTER_THRESHOLD='0.5',FINISH='0'),stdout=log,stderr=subprocess.STDOUT,check=True)
        assert len(json.loads(output.read_text())['records']) == count
        print(name,suffix,'complete',round(time.time()-start,1),flush=True)
