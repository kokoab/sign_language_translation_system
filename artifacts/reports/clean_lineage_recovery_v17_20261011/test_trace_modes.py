from pathlib import Path
import shutil,subprocess,os,json,collections
O=Path(__file__).resolve().parent;d=O/'trace_checks';d.mkdir(exist_ok=True)
for p in (O/'traced').glob('*.swift'):shutil.copy2(p,d/p.name)
p=d/'LiveReelEngine.swift';s=p.read_text().replace('    private func resetSpellMode() {','    func diagnosticSpellingMode() { spellMode = .spelling }\n    private func resetSpellMode() {');p.write_text(s)
p=d/'main.swift';s=p.read_text().replace('        engine.reset()','''        engine.reset()
        if ProcessInfo.processInfo.environment["FORCE_SPELLING"] == "1" { engine.diagnosticSpellingMode() }''')
needle='    engine.startTrace()'
assert needle in s
s=s.replace(needle,'''    // A later sign can lock its identity before an earlier pending sign closes.
    let first = LiveWord(gloss: "MY", startFrame: 10, endFrame: 15, commitFrame: 30,
        startSeconds: 1, endSeconds: 1.5, commitSeconds: 3, score: 0.9, early: false)
    let later = LiveWord(gloss: "NAME", startFrame: 20, endFrame: 25, commitFrame: 28,
        startSeconds: 2, endSeconds: 2.5, commitSeconds: 2.8, score: 0.9, early: true)
    precondition(engine.runtime.merge([later, first], []).map(\\.gloss) == ["MY", "NAME"])
    engine.reset()
'''+needle);p.write_text(s)
with (O/'trace_checks_build.log').open('w') as f:
 subprocess.run(['xcrun','swiftc','-O','-o',str(O/'trace_checks_probe')]+[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','main.swift']],stdout=f,stderr=subprocess.STDOUT,check=True)
result={}
for name,extra in [('always',{'LETTERS':'always'}),('spelling',{'FORCE_SPELLING':'1'})]:
 out=O/('trace_'+name+'.json')
 with (O/('trace_'+name+'.log')).open('w') as f:
  subprocess.run([str(O/'trace_checks_probe'),str(O/'resources'),str(O/'trace_case.json'),str(out)],env=dict(os.environ,FINISH='0',LETTER_THRESHOLD='0.7',**extra),stdout=f,stderr=subprocess.STDOUT,check=True)
 rows=json.load(open(str(out)+'.trace.json'));frames=[r for r in rows if not r.get('start')]
 assert len(frames)==321 and all('e' in r for r in frames)
 assert any(r['mode']==name for r in frames)
 result[name]={'frames':len(frames),'modes':dict(collections.Counter(r['mode'] for r in frames)),'chronological_merge_assertion':'passed','forced_initial_mode':True}
frames=[r for r in json.load(open(O/'trace_replay.json.trace.json')) if not r.get('start')]
assert len(frames)==321 and all('e' in r for r in frames)
assert all(b['word_count']>=a['word_count'] for a,b in zip(frames,frames[1:]))
result['automatic_name']={'frames':len(frames),'modes':dict(collections.Counter(r['mode'] for r in frames)),'word_timeline_monotonic':True}
(O/'trace_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
