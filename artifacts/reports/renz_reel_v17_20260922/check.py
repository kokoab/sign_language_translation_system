"""Check temporal coverage, model outputs and source provenance after replay."""
import json,runpy
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent
m=runpy.run_path(str(HERE/'run.py'));m['self_check']()
for folder in [HERE,HERE/'edge_padded',HERE/'edge_padded_midpoint']:
 if not (folder/'results.json').exists():continue
 result=json.loads((folder/'results.json').read_text());assert result['complete'] and len(result['results'])==12
 provenance=json.loads((folder/'provenance.json').read_text())
 for path,sha in provenance['sha256'].items():
  if path.endswith('/run.py'):continue # preserved initial_runner.py for first-pass provenance
  assert m['digest'](m['ROOT']/path)==sha,path
 for cache in folder.glob('*.npz'):
  with np.load(cache) as z:
   n=int(z['nframes']);expected=n if folder.name=='edge_padded' else max(1,n-15)
   assert len(z['features'])==len(z['boundary_probability'])==expected
   assert np.isfinite(z['features']).all() and np.isfinite(z['boundary_probability']).all()
 for row in result['results']:
  assert row['edit_distance']==m['distance'](row['reference'],row['hypothesis'])
  assert all(c['end']>c['start']>=0 for c in row['candidates'])
 print(folder.name,'passed 12-source hashes, temporal coverage, finite outputs and edit counts')
base=json.loads((HERE/'provenance.json').read_text())
assert m['digest'](HERE/'initial_runner.py')==base['sha256'][str((HERE/'run.py').relative_to(m['ROOT']))]
edge=json.loads((HERE/'edge_padded/provenance.json').read_text())
assert m['digest'](HERE/'edge_runner.py')==edge['sha256'][str((HERE/'run.py').relative_to(m['ROOT']))]
latest=json.loads((HERE/'edge_padded_midpoint/provenance.json').read_text())
assert m['digest'](HERE/'run.py')==latest['sha256'][str((HERE/'run.py').relative_to(m['ROOT']))]
print('Span/edit self-checks and all three runner hashes passed')
