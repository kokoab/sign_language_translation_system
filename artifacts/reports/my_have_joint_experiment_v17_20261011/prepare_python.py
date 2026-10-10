from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[2]
s=(R/'active/v17/segmental_runtime_v17.py').read_text()
s=s.replace('import json\n','import json\nimport os\n',1)
s=s.replace('ROOT = Path(__file__).resolve().parents[2]','ROOT = Path(__file__).resolve().parents[3]')
s=s.replace('self.early, self.shown_early = {}, []','self.early, self.shown_early = {}, []\n        self.pending_output = []')
a=s.index('                # Early identity must not publish');b=s.index("                if g['end']",a)
s=s[:a]+'''                if self.config['stream'].get('experiment_arm', 'current') == 'current':
'''+''.join('    '+x if x.strip() else x for x in s[a:b].splitlines(keepends=True))+s[b:]
needle='''        self.shown_early = [e for e in self.shown_early if e[0] >= self.origin - 2]
        self._trim()
        return committed'''
repl='''        if self.config['stream'].get('experiment_arm') == 'ordered':
            self.pending_output += committed
            barrier = float('inf') if final else min((g['start'] + origin for g in segs
                if g['end'] + origin >= self.origin
                and not any(abs(g['start'] + origin - a) <= 2 for a, _ in self.shown_early)), default=float('inf'))
            committed = sorted((w for w in self.pending_output if w['start_frame'] < barrier), key=lambda w: w['start_frame'])
            self.pending_output = [w for w in self.pending_output if w['start_frame'] >= barrier]
        self.shown_early = [e for e in self.shown_early if e[0] >= self.origin - 2]
        self._trim()
        return committed'''
assert needle in s;s=s.replace(needle,repl);(P/'runtime_experiment.py').write_text(s)
s=(R/'artifacts/reports/have_root_fix_v17_20261010/evaluate_production.py').read_text()
a=s.index('    assert set(names)');b=s.index('    original_logits',a)
s=s[:a]+'''    cfg = json.loads((ROOT/'artifacts/reports/have_root_fix_v17_20261010/config_fixed.json').read_text())
    built = build_runtime(cfg, device='mps')
    base = built.words if hasattr(built, 'words') else built
    spec = importlib.util.spec_from_file_location('runtime_experiment', HERE/'runtime_experiment.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    runtimes = {}
    for name in names:
        local = copy.deepcopy(cfg)
        stream=local['stream'];stream['experiment_arm']=name
        if name.startswith('rollback'):
            stream['have_contact_closure']=False
            if name not in ('rollback_geometry','rollback_both'):
                stream.pop('crop_pair',None);stream.pop('handshape_pair',None)
            if name not in ('rollback_roll','rollback_both'):stream.pop('tired_rescue_droop',None)
        runtimes[name] = mod.SegmentalRuntime(base.boundary.model, base.recognizer, local, mode='both', letter_gate=.95)
'''+s[b:]
s=s.replace("('production_parity_' if os.environ.get('CASES') else 'production_parity_')", "'development_'")
(P/'evaluate.py').write_text(s)
