"""Lightweight Core ML loading for the desktop live app.

- `lightweight_imports()`: coremltools probes TensorFlow (and transformers) on import. The live app
  never uses TensorFlow; that probe alone cost ~9 s of a ~12 s model build. Call it before anything
  imports coremltools. transformers is blocked too unless the PyTorch Stage 3 renderer is wanted.
- `load(package)`: an MLModel from a cached compiled .mlmodelc (~/Library/Caches/slt_v17_coreml), so the
  .mlpackage is not recompiled on every start. The cache is keyed by the package manifest.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
# Internal disk: large compiled weights fail to plan from the exFAT project drive (error -5).
COMPILED = Path.home() / 'Library/Caches/slt_v17_coreml'


def lightweight_imports(allow_transformers: bool = False) -> None:
    blocked = ['tensorflow'] + ([] if allow_transformers else ['transformers'])
    for name in blocked:
        if name not in sys.modules:
            sys.modules[name] = None      # `import name` now raises ImportError


def _key(package: Path) -> str:
    manifest = package / 'Manifest.json'
    stamp = manifest.read_bytes() if manifest.exists() else str(package.stat().st_mtime_ns).encode()
    weights = sorted(package.rglob('*.bin')) + sorted(package.rglob('*.mlmodel'))
    for path in weights:
        info = path.stat()
        stamp += f'{path.name}:{info.st_size}:{info.st_mtime_ns}'.encode()
    return hashlib.sha256(stamp).hexdigest()[:12]


def load(package, compute_units: str = 'ALL'):
    import coremltools as ct
    package = Path(package)
    units = getattr(ct.ComputeUnit, compute_units)
    compiled = COMPILED / f'{package.stem}-{_key(package)}.mlmodelc'
    if not compiled.exists():
        COMPILED.mkdir(parents=True, exist_ok=True)
        ct.utils.compile_model(str(package), str(compiled))
    return ct.models.CompiledMLModel(str(compiled), compute_units=units)


def spec(package):
    import coremltools as ct
    return ct.utils.load_spec(str(package))
