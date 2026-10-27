"""ATLAS processing-pipeline flowchart (camera frame -> English text and speech), drawn with the paper's diagram helper.

Every node is checked against the iPhone implementation: LiveReelViewController.captureOutput/processCapturedFrame
(latest-frame gate, 20 fps deadline), LiveReelEngine.process (Apple Vision landmarks, finish gesture, hand crops,
MobileCLIP2-S0, boundary model, span recognizer batch 8, segmental decoder), finishUtterance (flush + English + speech).
"""
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
R = OUT.parents[2]
sys.path.insert(0, str(R / 'docs/capstone_papers/review_assets'))
import build_review_assets as bra

bra.HERE = OUT          # write the figure and its Mermaid source here, not into the paper assets folder
bra.flow('pipeline_flowchart', 'ATLAS processing pipeline: live camera frame to English text and speech', {
    'camera':    (6, 29.0, 'Device camera frame (30 fps)', 'box'),
    'gate':      (6, 26.8, 'Engine free and 50 ms since last processed frame?', 'decision'),
    'drop':      (10.2, 26.8, 'Skip frame; keep only the newest frame', 'box'),
    'landmarks': (6, 24.6, 'Landmark extraction: 61 points, 5 values each', 'box'),
    'boundary':  (2.6, 22.2, 'Boundary model: outside / begin / inside per frame (0.3 s look-ahead)', 'box'),
    'crops':     (9.4, 22.2, 'Hand crops (left, right, both) to MobileCLIP2-S0 features', 'box'),
    'intervals': (2.6, 19.8, 'Candidate sign intervals, each resampled to 32 frames', 'box'),
    'recognize': (6, 17.4, 'Squeezeformer recognizer: scores for 100 signs', 'output'),
    'decode':    (6, 15.2, 'Segmental decoder: combine timing and sign scores', 'box'),
    'commit':    (6, 13.0, 'Commit conditions met?', 'decision'),
    'wait':      (10.2, 13.0, 'Keep observing next frames', 'box'),
    'buffer':    (6, 10.8, 'Gloss buffer: append committed sign', 'box'),
    'english':   (6, 8.6, 'Update English text from accumulated glosses', 'box'),
    'finish':    (6, 6.4, 'Finish button or one-second Finish gesture?', 'decision'),
    'more':      (10.2, 6.4, 'Continue signing', 'box'),
    'final':     (6, 4.2, 'Flush open sign; finalize English text', 'box'),
    'output':    (6, 2.0, 'Display English text and speak it on the device', 'output'),
}, [('camera', 'gate', ''), ('gate', 'drop', 'No'), ('gate', 'landmarks', 'Yes'),
    ('landmarks', 'boundary', ''), ('landmarks', 'crops', ''),
    ('boundary', 'intervals', ''), ('intervals', 'recognize', ''), ('crops', 'recognize', ''),
    ('recognize', 'decode', ''), ('decode', 'commit', ''), ('commit', 'wait', 'No'), ('commit', 'buffer', 'Yes'),
    ('buffer', 'english', ''), ('english', 'finish', ''), ('finish', 'more', 'No'), ('finish', 'final', 'Yes'),
    ('final', 'output', '')], (9.5, 21), (12.2, 30))
print('saved')
