"""One-shot macOS process-exit notification, then final cache validation. No polling."""
from contextlib import closing
import json
import select
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).parent
ROOT=HERE.parents[2]


def main(pid):
    with closing(select.kqueue()) as queue:
        event=select.kevent(pid,filter=select.KQ_FILTER_PROC,
                            flags=select.KQ_EV_ADD|select.KQ_EV_ONESHOT,fflags=select.KQ_NOTE_EXIT)
        try:
            queue.control([event],0,0)
        except ProcessLookupError:
            pass
        else:
            print('Registered process-exit event for '+str(pid),flush=True)
            queue.control(None,1,None)
    completion=json.loads((HERE/'preparation_completion.json').read_text())
    if completion.get('status')!='complete':
        raise RuntimeError('Extraction did not complete; no full-validation claim')
    subprocess.run([str(ROOT/'venv/bin/python'),str(HERE/'validate.py')],cwd=ROOT,check=True)
    result=json.loads((HERE/'validation.json').read_text())
    if result['status']!='passed':raise RuntimeError('Full validation incomplete or failed')
    return dict(status='passed',validated=result['validated'],training_started=False)

if __name__=='__main__':
    try:
        result=main(int(sys.argv[1]))
        message='All pretrained boundary poses extracted and validated. Ready to discuss training.'
    except Exception as exc:
        result=dict(status='failed',error=type(exc).__name__+': '+str(exc),training_started=False)
        message='Pose validation needs attention; inspect validation_completion.json.'
    (HERE/'validation_completion.json').write_text(json.dumps(result,indent=2)+'\n')
    subprocess.run(['osascript','-e','display notification '+json.dumps(message)+' with title "SLT boundary validation"'],check=False)
