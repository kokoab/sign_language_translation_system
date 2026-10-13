import collections, hashlib, json, re, shutil, subprocess, xml.etree.ElementTree as ET
from pathlib import Path
root=Path(__file__).resolve().parent
records=json.loads((root/'selected_python.json').read_text())
for n,r in enumerate(records,1):
 r.update(id=f'PY-{n:02}',status='Passed',check=r['method'].removeprefix('test_').replace('_',' ').capitalize(),source=r['module_class'].rsplit('.',1)[0].replace('.','/')+'.py',evidence='python.log')
swift={'testLandmarkCaptureRotation':('WB-01','Camera-coordinate rotation'), 'testAllPracticeReferencesPlayAndLoop':('AI-04','Playback and looping of all 100 reference videos'), 'testCameraLayoutsLandmarkPreferenceAndReferenceOverlay':('AI-03','Preference persistence and reference display controls; secondary layout checks'), 'testLatestFrameDropsBacklogAndInvalidatesStoppedWork':('WB-02','Frame backlog handling and cancellation of stopped work'), 'testFinishPoseRightOpenLeftFistAboveShoulders':('WB-05','Finish gesture conditions, activation timing and state reset'), 'testWordModeBridgesRestDipInsideSign':('WB-04','Whole-sign candidate across a short boundary dip'), 'testTiredRescueMatchesPython':('WB-04','Native recognition correction agrees with reference calculations'), 'testCameraToolbarOmitsRemovedControls':('AI-02','Current toolbar controls remain visible; removed controls remain absent'), 'testLiveAndPracticeSupportPortraitAndLandscapeLayout':('AI-02','Portrait support and landscape control layout')}
for n,(method,(group,check)) in enumerate(swift.items(),1):
 evidence='swift_retest.log' if method in ['testCameraToolbarOmitsRemovedControls','testLiveAndPracticeSupportPortraitAndLandscapeLayout'] else 'swift.log'
 assert re.search(re.escape(method)+r"\]' passed",(root/evidence).read_text()),method
 records.append(dict(id=f'SW-{n:02}',group=group,framework='XCTest',method=method,check=check,status='Passed',source='ATLAS/mobile_app/slt_mobile_app/ios/RunnerTests/RunnerTests.swift',evidence=evidence))
events=[]
for l in (root/'flutter.jsonl').read_text().splitlines():
 try: events.append(json.loads(l))
 except ValueError: pass
starts={e['test']['id']:e['test'] for e in events if e.get('type')=='testStart'}
flutter_groups=['AI-01','AI-01','AI-01','AI-02','AI-05','WB-07','WB-07'];n=0
for e in events:
 if e.get('type')!='testDone' or e.get('hidden') or e['testID'] not in starts: continue
 t=starts[e['testID']]
 if t['name'].startswith('loading '):continue
 assert e['result']=='success' and not e['skipped'],e
 group=flutter_groups[n];n+=1
 records.append(dict(id=f'FL-{n:02}',group=group,framework='Flutter Test',method=t['name'],check=t['name'],status='Passed',source='ATLAS/mobile_app/slt_mobile_app/test/widget_test.dart',evidence='flutter.jsonl'))
assert n==7
xml=Path('ATLAS/mobile_app/slt_mobile_app/build/app/test-results/testDebugUnitTest/TEST-com.example.slt_mobile_app.live.LiveAlignmentTest.xml');shutil.copyfile(xml,root/'kotlin_results.xml')
t=ET.parse(xml).getroot();assert t.get('failures')=='0' and t.get('errors')=='0' and t.get('skipped')=='0'
for n,c in enumerate(t.findall('testcase'),1):records.append(dict(id=f'KT-{n:02}',group='WB-04',framework='JUnit',method=c.get('name'),check=c.get('name'),status='Passed',source='ATLAS/mobile_app/slt_mobile_app/android/app/src/test/kotlin/com/example/slt_mobile_app/live/LiveAlignmentTest.kt',evidence='kotlin_results.xml'))
assert len(records)==47
counts=collections.Counter(r['group'] for r in records)
assert sum(v for k,v in counts.items() if k.startswith('WB'))==38
files={r['source'] for r in records};hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(files)}
result=dict(date='2026-10-13',base_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),working_tree_source_hashes=hashes,counts=dict(sorted(counts.items())),total=47,passed=47,failed=0,skipped=0,records=records,scope='29 Python, 9 native XCTest, 7 Flutter and 2 JVM cases; broader Python suite 245 passed. Latest result per case, not a single simultaneous run.',corrections=['Outdated landscape-only expectation aligned to portrait-first requirement before initial native run.','Initial native toolbar test failed because removed Letters control was expected; aligned to recorded UI requirement and rerun.'],excluded='Experimental-only models, dataset gates, standalone profiles/replays, empty scaffold and unexecuted model-dependent native methods are not included in the 47-case total.')
(root/'results.json').write_text(json.dumps(result,indent=2))
lines=['# ATLAS software test case register — October 13, 2026','','47 selected cases; 47 passed, 0 failed, 0 skipped in their final recorded executions. Initial native failure and correction retained in logs.','','| ID | Group | Framework | Test method | Final outcome | Evidence |','|---|---|---|---|---|---|']
lines += [f"| {r['id']} | {r['group']} | {r['framework']} | {r['method']} | {r['status']} | {r['evidence']} |" for r in records]
(root/'CASE_REGISTER.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(counts=result['counts'],total=47)))
