from pathlib import Path
import json,hashlib,subprocess,sys
R=Path('/Volumes/secret/SLT/SLT');O=Path(__file__).resolve().parent
S=Path('/Volumes/secret/SLT/mobile_app/slt_mobile_app/ios/Runner/LiveReel');P=R/'artifacts/reports/lineage_runtime_investigation_v17_20261011'
manifest=[]
for variant in ['control','continuous','continuous_contact']:
 if len(sys.argv)>1 and variant not in sys.argv[1:]:continue
 d=O/variant;d.mkdir(exist_ok=True)
 for name in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift']:
  src=S/name
  if name=='LiveReelDecoder.swift' and variant=='continuous_contact':src=R/'artifacts/reports/my_have_phone_v17_20261011/candidate_LiveReelDecoder.swift'
  s=src.read_text();manifest.append({'variant':variant,'path':str(src),'sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
  if name=='LiveReelCore.swift':s=s.replace('private(set) var processed = 0','private(set) var processed = 0\n    func setDiagnosticPhase(_ phase: Int) { processed = phase }')
  if name=='LiveReelModels.swift':
   s=s.replace('name: String = "SpanRecognizerV17FragmentRejectLettersB8FP16"','name: String = ProcessInfo.processInfo.environment["MODEL"] ?? "SpanRecognizerV17CleanRecoveryLettersB8FP16"')
   s=s.replace('letterThreshold: Float = 0.5','letterThreshold: Float = Float(ProcessInfo.processInfo.environment["LETTER_THRESHOLD"] ?? "0.5") ?? 0.5')
  if variant!='control' and name=='LiveReelDecoder.swift':
   s=s.replace('words.rawCache = shared; letters.rawCache = shared','// Word and letter timelines may have different origins after NAME.\n        // Keep their span caches separate to prevent index-key collisions.')
   s=s.replace('private func merge(_ wordOut:', 'func merge(_ wordOut:')
   s=s.replace('$0.element.0.commitSeconds', '$0.element.0.startSeconds').replace('$1.element.0.commitSeconds', '$1.element.0.startSeconds')
  if variant!='control' and name=='LiveReelEngine.swift':
   a=s.index('        runtime = LiveDualRuntime(');b=s.index('\n    }',a)
   s=s[:a]+'''        wordsOnly = LiveSegmentalRuntime(boundary: wordBoundary, recognizer: recognizer, encoder: encoder,
                                         config: .v3Words, mode: .words)
        wordsGated = LiveSegmentalRuntime(boundary: wordBoundary, recognizer: recognizer, encoder: encoder,
                                          config: .v3Words, mode: .both, letterGate: Self.wordLetterGate)
        runtime = LiveDualRuntime(words: wordsGated,
            letters: LiveSegmentalRuntime(boundary: letterBoundary, recognizer: recognizer, encoder: encoder, mode: .letters))
'''+s[b:]
   s=s.replace('for w in out where w.startSeconds >= suppressUntil { shown += accept(w) }','for w in runtime.merge(out, []) where w.startSeconds >= suppressUntil { shown += accept(w) }')
   s=s.replace('for w in try runtime.observe(observation, image: image) { shown += accept(w); lastWord = w }','''let wordOut = try wordDecoder.observe(observation, image: image).filter { !$0.gloss.hasPrefix("FS_") }
                let letterOut = try runtime.letters.observe(observation, image: nil, hand: wordDecoder.hands.last)
                traceFrame(observation, wordDecoder.hands.last, held: false)
                for w in runtime.merge(wordOut, letterOut) { shown += accept(w); lastWord = w }''')
   a=s.index('            for w in try runtime.finish() { shown += accept(w) }');b=s.index('            spellMode = .words; event = "off:name"',a)
   s=s[:a]+'''            for w in runtime.merge([], try runtime.letters.finish()) { shown += accept(w) }
            shown += speller.flush()
            runtime.letters.reset(); backlog = []
'''+s[b:]
   a=s.index('        for w in try wordDecoder.finish() where w.endSeconds <= nameEnd');b=s.index('        spellMode = .name;',a)
   s=s[:a]+'''        runtime.letters.reset()
        let replay = backlog.filter { $0.0.seconds > nameEnd }
        backlog = []
        for (o, h) in replay {
            let letters = try runtime.letters.observe(o, image: nil, hand: h)
            for w in runtime.merge([], letters) { shown += accept(w) }
        }
'''+s[b:]
  (d/name).write_text(s)
 (d/'main.swift').write_text((P/'current/main.swift').read_text())
 cmd=['xcrun','swiftc','-O','-o',str(O/(variant+'_probe'))]+[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','main.swift']]
 with (O/(variant+'_build.log')).open('w') as f:
  p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 print(variant,'build',p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
(O/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
