from pathlib import Path
import shutil,json,subprocess
D=Path(__file__).resolve().parent;d=D/'aux_native';shutil.copytree(D/'bounded_native',d,dirs_exist_ok=True)
p=d/'Experiment.swift';s=p.read_text().replace('    static var maxDuration','''    static var aux = 1
    static func flatStart(_ observations:[LiveObservation],start:Double,end:Double)->Bool {
        let rows=LiveMyLikeMotion.features(observations).filter { $0.t>=start-1e-6 && $0.t<=min(end,start+0.25)+1e-6 && $0.flatValid && $0.straight>=0.97 && $0.pinch>=0.8 && abs($0.px)<=0.6 && $0.py>=(-0.15) && $0.py<=1.0 }
        return (0..<2).contains { slot in let rr=rows.filter{$0.slot==slot};return rr.count>=2 && rr.last!.t-rr.first!.t>=0.05 }
    }
    static var maxDuration''');p.write_text(s)
p=d/'LiveReelDecoder.swift';s=p.read_text().replace('repairEligible:keys.map { Experiment.eligible(obs,start:seconds($0.s),end:seconds($0.e)) }','repairEligible:keys.map { Experiment.eligible(obs,start:seconds($0.s),end:seconds($0.e)) },flatStarts:keys.map { Experiment.flatStart(obs,start:seconds($0.s),end:seconds($0.e)) }');p.write_text(s)
p=d/'LiveReelModels.swift';s=p.read_text().replace('repairEligible:[Bool]? = nil','repairEligible:[Bool]? = nil,flatStarts:[Bool]? = nil');needle='''                    }
                }
                out.append(chosen + Array(letters''';replacement='''                    } else if Experiment.aux>0 && repairEligible?[i+slot] == true && flatStarts?[i+slot] == true && order.first != my && order.first != like && order.prefix(3).contains(my) && adapted.indices.max(by:{adapted[$0]<adapted[$1]})==my && exp(lp[my])>=0.98 {
                        chosen.swapAt(my,order.first!)
                    }
                }
                out.append(chosen + Array(letters''';assert needle in s;s=s.replace(needle,replacement);p.write_text(s)
p=d/'main.swift';s=p.read_text().replace('        Experiment.events = []','        Experiment.events = []\n        Experiment.aux=item["aux"] as? Int ?? 1');p.write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','Experiment.swift']]
with (D/'aux_build.log').open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(D/'aux_probe')]+files+[str(d/'main.swift')],stdout=f,stderr=subprocess.STDOUT,check=True)
base=json.loads((D/'bounded_new_cases.json').read_text());(D/'aux_screen_cases.json').write_text(json.dumps([dict(r,aux=aux,id=f'aux{aux}__'+r['id']) for r in base if r['phase'] in [0,4] for aux in [0,1]]))
print('Built flat-chest onset auxiliary correction ablation')
