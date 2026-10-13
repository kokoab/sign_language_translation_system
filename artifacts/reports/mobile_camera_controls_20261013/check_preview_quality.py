from pathlib import Path
import json,re,subprocess
O=Path(__file__).resolve().parent
scores={}
for original in sorted((O/'before/media').glob('*.mp4')):
 result=subprocess.run(['ffmpeg','-hide_banner','-i',str(original),'-i',str(O/'h264_previews'/original.name),'-lavfi','[0:v]settb=AVTB,setpts=N[a];[1:v]settb=AVTB,setpts=N[b];[a][b]ssim','-f','null','-'],capture_output=True,text=True,check=True)
 score=float(re.findall(r'All:([0-9.]+)',result.stderr)[-1]);scores[original.stem]=score
 assert score>0.99,(original,score)
(O/'media_quality.json').write_text(json.dumps({'metric':'decoded-frame SSIM','minimum':min(scores.values()),'mean':sum(scores.values())/len(scores),'clips':scores},indent=2)+'\n')
print(len(scores),'clips; min SSIM',min(scores.values()))
