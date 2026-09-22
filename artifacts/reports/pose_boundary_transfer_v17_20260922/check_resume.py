"""Resume preserves inputs; reject altered source or cache identities."""
import ast
import copy
import json
from pathlib import Path
from prepare import HERE, ROOT, digest, resume_entries

source=ROOT/'active/v17/temporal_boundary_manifest_20260922.json'
manifest=json.loads(source.read_text())
previous=json.loads((HERE/'prepared_manifest_serial.json').read_text())
one=copy.deepcopy(previous);one['records']=one['records'][:1]
assert len(resume_entries(one,manifest,digest(source)))==1
for field in ('pose_sha256','video_sha256','item'):
    bad=copy.deepcopy(one);bad['records'][0][field]='invalid'
    try: resume_entries(bad,manifest,digest(source))
    except (ValueError,FileNotFoundError): pass
    else: raise AssertionError('accepted invalid '+field)
def extraction(path):
    return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='extract'))
assert extraction(HERE/'prepare.py')==extraction(HERE/'prepare_serial.py')
print('Resume identity/hash rejection and unchanged per-video extraction checks pass.')
