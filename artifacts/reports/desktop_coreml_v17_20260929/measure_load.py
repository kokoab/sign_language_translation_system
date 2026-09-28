"""Desktop live components: load time, peak memory, Stage 3 latency (one process per variant)."""
import json, resource, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
started = time.perf_counter()
if '--lightweight' in sys.argv:
    sys.argv.remove('--lightweight')
    from active.v17.coreml_runtime_v17 import lightweight_imports
    lightweight_imports()
from scripts.app_shell_v17 import parser
from scripts.live_segmental_v17 import SegmentalRecognizer, render_utterance
args = parser().parse_args(['--no-speech', '--no-ollama'] + sys.argv[2:])
imported = time.perf_counter()
model = SegmentalRecognizer(args)
built = time.perf_counter()
words = [dict(gloss=g, score=s, start_seconds=i * .8, end_seconds=i * .8 + .6) for i, (g, s) in
         enumerate([('HELLO', .9), ('MY', .9), ('NAME', .9), ('fs-GELO', .8), ('I', .7), ('GO', .6), ('SCHOOL', .8), ('TOMORROW', .9)])]
t = time.perf_counter(); first = render_utterance(model.naturalizer, words); t1 = time.perf_counter()
runs = []
for _ in range(5):
    s = time.perf_counter(); value = render_utterance(model.naturalizer, words); runs.append(time.perf_counter() - s)
result = dict(variant=sys.argv[1], import_s=imported - started, build_s=built - imported,
              peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
              stage3_first_ms=1000 * (t1 - t), stage3_median_ms=1000 * sorted(runs)[2], sentence=value['sentence'],
              recognizer_device=model.runtime.recognizer.device,
              torch_loaded='torch' in sys.modules, transformers_loaded='transformers' in sys.modules)
(Path(__file__).parent / f'load_{sys.argv[1]}.json').write_text(json.dumps(result, indent=1))
print(json.dumps(result, indent=1))
