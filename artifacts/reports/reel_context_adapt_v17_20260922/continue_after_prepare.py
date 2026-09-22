"""Wait for the preparation process exit using macOS kqueue; never poll training."""
import json, select, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).parent
pid=int(sys.argv[1])
queue=select.kqueue()
event=select.kevent(pid,filter=select.KQ_FILTER_PROC,
                   flags=select.KQ_EV_ADD|select.KQ_EV_ONESHOT,
                   fflags=select.KQ_NOTE_EXIT)
try:
    queue.control([event],0,0)
except ProcessLookupError:
    pass
else:
    queue.control(None,1,None)
finally:
    queue.close()
if not (OUT/'cache_manifest.json').exists() or (OUT/'failure.json').exists():
    raise SystemExit('Preparation did not pass; training not started. Read failure.json.')
with (OUT/'train.log').open('w') as log:
    result=subprocess.run([str(ROOT/'venv/bin/python'),'-u',
                           str(ROOT/'scripts/train_reel_context_adapt_v17.py'),'--train'],
                          cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
if result.returncode:
    raise SystemExit(result.returncode)
result=json.loads((OUT/'completion.json').read_text())
entry=("\n## 2026-09-22 — Reel context head adaptation completed\n\n"
       f"Finished {result['epochs']}epochs in {result['training_seconds']/60:.2f}min fitting. "
       f"Selected epoch{result['selected_epoch']}; no promotion. Baseline WER"
       f"{result['baseline']['wer']:.2%}, best trained {result['best_trained']['wer']:.2%}. "
       "Full retention/isolated validation gates are in history.json; best trained does not "
       "necessarily qualify. Report artifacts/reports/reel_context_adapt_v17_20260922/REPORT.md. "
       "Next read report/confirmation before any deployment or further unfreezing.\n\n")
logpath=ROOT/'docs/ground_truth/live-streaming/log.md'
text=logpath.read_text();logpath.write_text(text.replace('# live-streaming — log\n','# live-streaming — log\n'+entry,1))
current=ROOT/'PROJECT_GROUND_TRUTH.md';text=current.read_text()
start=text.index('**Reel context adaptation PREPARING 2026-09-22:**')
end=text.index('\n\n',start)
replacement=("**Reel context adaptation COMPLETE 2026-09-22:** "
             f"{result['epochs']}epochs/{result['training_seconds']/60:.2f}min fitting, "
             f"selectedepoch{result['selected_epoch']}. No promotion. "
             "Report `artifacts/reports/reel_context_adapt_v17_20260922/REPORT.md`. "
             "Proposalclassifier/verifierfusion heads; frozenencoders/BIO/acceptance. "
             "Next inspect retention, isolatedvalidation and confirmation before deployment.")
current.write_text(text[:start]+replacement+text[end:])
subprocess.run([str(ROOT/'venv/bin/python'),str(ROOT/'scripts/index_large_artifacts_v17.py')],cwd=ROOT,check=True)
