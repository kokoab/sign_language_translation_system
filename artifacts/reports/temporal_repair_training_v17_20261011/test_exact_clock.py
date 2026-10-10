"""Regression for the actual first-frame/exclusive-end rounding errors found in MY cores."""
from fractions import Fraction
import json
from pathlib import Path
O=Path(__file__).resolve().parent
inside=lambda t,core:core[0]-1e-9<=t<core[1]-1e-9
start=float(Fraction(5,30));end=float(Fraction(7,30));exact=[start,end]
assert inside(start,exact) and not inside(end,exact)
rounded=[.166667,.23333333333333334]
assert not inside(start,rounded), 'Reproduce first-frame loss in prior rounded input'
short=[float(Fraction(7,30)),float(Fraction(9,30))]
assert not inside(.3,short) and inside(.3,[.233333,.3000003333333333]),'Reproduce erroneous end-frame inclusion'
rows=json.loads((O/'timestamp_correction.json').read_text())['records'];assert len(rows)==132
for row in rows:
 assert row['exact_core_seconds']==[float(Fraction(row['start_rational'])),float(Fraction(row['end_rational']))]
 assert row['frame_end_inclusive']>=row['frame_start']
(O/'exact_clock_tests.json').write_text(json.dumps({'first_frame_preserved':True,'exclusive_end_excluded':True,'prior_errors_reproduced':True,'source_events_checked':len(rows)})+'\n')
print('Exact-clock regressions and132reviewed frame mappings pass.')
