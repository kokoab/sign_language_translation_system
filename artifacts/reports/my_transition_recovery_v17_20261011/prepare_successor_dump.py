from pathlib import Path
import json,shutil,subprocess
D=Path(__file__).resolve().parent;d=D/'successor_dump_native';shutil.copytree(D/'dump_native',d,dirs_exist_ok=True)
p=d/'Experiment.swift';s=p.read_text().replace('"label":"MY"','"label":a["label"]!');p.write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','Experiment.swift']]
with (D/'successor_dump_build.log').open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(D/'successor_dump_probe')]+files+[str(d/'main.swift')],stdout=f,stderr=subprocess.STDOUT,check=True)
cores=[('FRIEND_1','FRIEND',1.52,1.58),('FRIEND_2','FRIEND',5.22,5.28),('NAME_1','NAME',7.77,7.83),('NAME_2','NAME',10.17,10.23),('FRIEND_3','FRIEND',15.52,15.58),('EAT_1','EAT',17.17,17.23),('FRIEND_4','FRIEND',19.82,19.88),('NAME_3','NAME',21.67,21.73)]
ann=[dict(id=n,label=l,lo=a,hi=b) for n,l,a,b in cores];base=json.loads((D/'new_video_cases.json').read_text());raw=D/'successor_windows';raw.mkdir(exist_ok=True)
(D/'successor_cases.json').write_text(json.dumps([dict(r,annotations=ann,dump_path=str(raw/(r['id']+'.bin'))) for r in base if r['phase'] in [0,4]],indent=2))
(D/'successor_review.json').write_text(json.dumps({'method':'Manually inspected source-PTS ffmpeg frames in review/successor_review.jpg plus saved contact sheets; conservative interiors around visibly formed FRIEND, NAME and EAT. Labels are independent of candidate model outputs.','events':ann},indent=2))
s=(D/'encode_actual.py').read_text().replace('actual_window_dump.json','successor_dump.json').replace("D/'actual_windows'","D/'successor_windows'")
s=s.replace("target=labels.index('MY')","targets=np.array([labels.index(x['label']) for x in metadata])")
s=s.replace("targets=np.full(len(f[0]),target)","targets=targets").replace("pred[i].argmax()==target","pred[i].argmax()==targets[i]")
s=s.replace('actual_features.npz','successor_features.npz').replace('actual_metadata.json','successor_metadata.json').replace('actual_feature_summary.json','successor_feature_summary.json');(D/'encode_successor.py').write_text(s)
print('Prepared successor-input contract and exporter')
