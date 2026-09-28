import json, select, subprocess, sys, time
from pathlib import Path
root=Path('/Volumes/secret/SLT/SLT'); report=root/'artifacts/reports/stage3_composition_v17_20260929'
pid=json.loads((report/'launch.json').read_text())['pid']
if not (report/'completion.json').exists():
    q=select.kqueue()
    try:
        q.control([select.kevent(pid, filter=select.KQ_FILTER_PROC, flags=select.KQ_EV_ADD|select.KQ_EV_ONESHOT, fflags=select.KQ_NOTE_EXIT)],0,0)
        q.control(None,1,None)
    except ProcessLookupError:
        pass
    finally:
        q.close()
completion=json.loads((report/'completion.json').read_text())
assert completion['status']=='complete', completion
rows=json.loads((report/'probes.json').read_text())
def norm(s):return ' '.join(''.join(c.lower() for c in s if c.isalnum() or c.isspace()).split())
expected={'HELLO MY FRIEND HOW YOU':'Hello, my friend. How are you?',
          'HELLO GOOD MORNING HOW YOU FRIEND':'Hello, good morning. How are you, friend?'}
for g,e in expected.items():
    r=next(r for r in rows if r['origin']=='user_requested' and ' '.join(r['glosses'])==g)
    assert norm(r['candidate'])==norm(e), r
print('Both real user phrase checks pass. Preparing Core ML export.',flush=True)
checkpoint=root/'artifacts/models/stage3_composition_v17_20260929'
p=checkpoint/'stage3_input_contract.json';c=json.loads(p.read_text());c.update(reviewed_templates_enabled=False,utterance_segmentation='model');p.write_text(json.dumps(c,indent=2)+'\n')
output=root/'artifacts/coreml/stage3_composition_v17_20260929'
argv=['active.v17.export_stage3_t5_coreml_v17','--checkpoint',str(checkpoint),'--output-dir',str(output),'--corpus',str(report/'train.jsonl'),'--parity-split','train','--rows','200']
code='import sys,runpy; sys.modules["tensorflow"]=None; sys.argv='+repr(argv)+'; runpy.run_module("active.v17.export_stage3_t5_coreml_v17",run_name="__main__")'
started=time.time()
with (report/'export.log').open('w') as log:
    result=subprocess.run([str(root/'venv/bin/python'),'-u','-c',code],cwd=root,stdout=log,stderr=subprocess.STDOUT)
(report/'export_completion.json').write_text(json.dumps(dict(exit_code=result.returncode,seconds=time.time()-started),indent=2)+'\n')
assert result.returncode==0, 'Core ML export failed; see export.log'
print('Core ML export and training-row numerical parity complete.',flush=True)
