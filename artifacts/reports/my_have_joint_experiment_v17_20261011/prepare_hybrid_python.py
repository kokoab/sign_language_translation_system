from pathlib import Path
P=Path(__file__).resolve().parent
s=(P/'runtime_ends.py').read_text()
g=(P/'geometry_probe.py').read_text();a=g.index('def features(');b=g.index("out={'validation'",a)
helper=g[a:b].replace('def features(', 'def experiment_geometry(')
s += '\n\n'+helper
needle='        words = len(self.recognizer.labels)\n        fist = self._fist_geometry()'
repl='''        words = len(self.recognizer.labels)
        my, like = self.recognizer.labels.index('MY'), self.recognizer.labels.index('LIKE')
        for j,(key,x) in enumerate(zip(keys,batch)):
            order=np.argsort(-logits[j][:words])
            if order[0] != like or my not in order[:3] or key[1]-key[0]+1 < 6: continue
            hands=experiment_geometry(x[0])
            if not hands or not all(h['straight']>=.98 and h['middle']>=.98 and h['pinch_min']>=.8 and h['pinch_change']<=.5 and h['wrist_motion']<=.75 and h['droop']<=.5 for h in hands): continue
            if int(self.recognizer.experiment_old_vote(x).argmax()) != my: continue
            logits[j]=np.array(logits[j],copy=True)
            logits[j][my],logits[j][like]=logits[j][like],logits[j][my]
        fist = self._fist_geometry()'''
assert needle in s;s=s.replace(needle,repl,1);(P/'runtime_hybrid.py').write_text(s)
s=(P/'evaluate_fast.py').read_text()
s=s.replace("    runtimes = {}",'''    spec3=importlib.util.spec_from_file_location('runtime_hybrid', HERE/'runtime_hybrid.py')
    hybrid=importlib.util.module_from_spec(spec3);spec3.loader.exec_module(hybrid)
    from active.v17.export_unified_multimodal_coreml_v17 import load_model
    old,meta=load_model(ROOT/'artifacts/models/span_recognizer_v17_local_a/best_model.pth')
    assert sorted(meta['label_to_index'],key=meta['label_to_index'].get)==list(base.recognizer.labels)
    old.to('mps').eval()
    def old_vote(x):
        batch=[torch.from_numpy(np.asarray(a)[None]).to('mps') for a in x]
        batch[2]=batch[2]>.5
        with torch.inference_mode():return old(*batch)[0].cpu().numpy()
    base.recognizer.experiment_old_vote=old_vote
    runtimes = {}''')
s=s.replace("'ordered' if name=='ordered_ends' else name", "'ordered' if name in ('ordered_ends','hybrid') else name")
s=s.replace("(ends if name=='ordered_ends' else mod).SegmentalRuntime", "(hybrid if name=='hybrid' else ends if name=='ordered_ends' else mod).SegmentalRuntime")
s=s.replace("'development_fast_'", "'development_hybrid_'").replace("'progress_fast_'", "'progress_hybrid_'")
(P/'evaluate_hybrid.py').write_text(s)
