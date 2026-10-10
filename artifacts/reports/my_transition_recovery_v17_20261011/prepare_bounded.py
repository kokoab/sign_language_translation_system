from pathlib import Path
import shutil,json,subprocess
D=Path(__file__).resolve().parent;d=D/'bounded_native';shutil.copytree(D/'native',d,dirs_exist_ok=True)
s=(D/'export_actual.py').read_text();s=s.replace('return words,self.letters(lf,hf,original_words)','return words,self.letters(lf,hf,original_words),original_words')
s=s.replace('w,l=wrapper(*x);','w,l,_=wrapper(*x);')
s=s.replace("ct.TensorType(name='letter_logits')],convert_to", "ct.TensorType(name='letter_logits'),ct.TensorType(name='original_word_logits')],convert_to")
s=s.replace("package=O/('SpanRecognizerV17'+''.join(w.title() for w in name.split('_'))+'B8FP16.mlpackage')", "package=O/'SpanRecognizerV17BoundedRepairB8FP16.mlpackage'").replace("O/f'{name}_export.json'","O/'bounded_export.json'");(D/'export_bounded.py').write_text(s)
p=d/'Experiment.swift';s=p.read_text().replace('    static var audit','''    static var maxDuration=0.35, floor=0.9
    static var rank=3, flat=false
    static var policy="pair"
    static func eligible(_ observations:[LiveObservation],start:Double,end:Double)->Bool {
        guard end-start<=maxDuration+1e-6 else {return false}
        if !flat {return true}
        let rows=LiveMyLikeMotion.features(observations).filter { $0.t>=start-0.1 && $0.t<=end && $0.flatValid && $0.straight>=0.97 && $0.pinch>=0.8 && abs($0.px)<=0.6 && $0.py>=(-0.15) && $0.py<=1.0 }
        return (0..<2).contains { slot in let rr=rows.filter { $0.slot==slot }; return rr.count>=2 && rr.last!.t>=end-0.100001 && rr.last!.t-rr.first!.t>=0.05 }
    }
    static var audit''');p.write_text(s)
p=d/'main.swift';s=p.read_text().replace('        Experiment.events = []','''        Experiment.events = []
        Experiment.maxDuration=item["max_duration"] as? Double ?? 0.35
        Experiment.floor=item["floor"] as? Double ?? 0.9
        Experiment.rank=item["rank"] as? Int ?? 3
        Experiment.flat=item["flat"] as? Bool ?? false
        Experiment.policy=item["policy"] as? String ?? "pair"''');p.write_text(s)
p=d/'LiveReelModels.swift';s=p.read_text().replace('let letters = outputs.keys.first(where: { $0 != "word_logits" })','let letters = outputs.keys.first(where: { $0 != "word_logits" && $0 != "original_word_logits" })')
s=s.replace('func logits(_ batch: [LiveSpanInput]) throws -> [[Float]] {','func logits(_ batch: [LiveSpanInput], repairEligible:[Bool]? = nil) throws -> [[Float]] {')
s=s.replace('            for slot in 0..<chunk.count {\n                out.append(Array(words[slot * 100 ..< slot * 100 + 100]) + Array(letters[slot * 27 ..< slot * 27 + 27]))','''            let original = result.featureValue(for: "original_word_logits")?.multiArrayValue?.floats()
            for slot in 0..<chunk.count {
                var chosen=Array(words[slot*100..<slot*100+100])
                if let original, let my=labels.firstIndex(of:"MY"), let like=labels.firstIndex(of:"LIKE") {
                    let adapted=chosen;chosen=Array(original[slot*100..<slot*100+100])
                    let order=chosen.indices.sorted { chosen[$0]>chosen[$1] }
                    let lp=LiveSegmental.logSoftmax(adapted)
                    if repairEligible?[i+slot] == true && order.first==like && order.prefix(Experiment.rank).contains(my) && adapted.indices.max(by:{adapted[$0]<adapted[$1]})==my && exp(lp[my])>=Experiment.floor {
                        if Experiment.policy == "swap" {chosen.swapAt(my,like)}
                        else {
                            let originalLP=LiveSegmental.logSoftmax(chosen)
                            let mass=exp(originalLP[my])+exp(originalLP[like])
                            let pair=LiveSegmental.logSoftmax([adapted[my],adapted[like]])
                            chosen=originalLP.map(Float.init);chosen[my]=Float(log(mass)+pair[0]);chosen[like]=Float(log(mass)+pair[1])
                        }
                    }
                }
                out.append(chosen + Array(letters[slot * 27 ..< slot * 27 + 27]))''');p.write_text(s)
p=d/'LiveReelDecoder.swift';s=p.read_text().replace('try recognizer.logits(batch)','try recognizer.logits(batch,repairEligible:keys.map { Experiment.eligible(obs,start:seconds($0.s),end:seconds($0.e)) })');p.write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','Experiment.swift']]
with (D/'bounded_build.log').open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(D/'bounded_probe')]+files+[str(d/'main.swift')],stdout=f,stderr=subprocess.STDOUT,check=True)
base=json.loads((D/'new_video_cases.json').read_text());arms=[dict(name=f'd{dur}_flat{flat}_rank{rank}',max_duration=dur,flat=flat,rank=rank,policy='pair',floor=.9) for dur in [.25,.35,.45] for flat in [False,True] for rank in [3,100]]
(D/'bounded_arms.json').write_text(json.dumps(arms,indent=2));(D/'bounded_screen_cases.json').write_text(json.dumps([dict(r,**{k:v for k,v in a.items() if k!='name'},id=a['name']+'__'+r['id']) for r in base if r['phase'] in [0,4] for a in arms],indent=2))
print('Built12boundedcandidate-repairarms')
