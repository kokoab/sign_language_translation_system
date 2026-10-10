"""Extract timestamped review frames without model predictions or image distortion."""
from pathlib import Path
import json,subprocess,sys
import cv2,numpy as np
from PIL import Image,ImageDraw
O=Path(__file__).resolve().parent;R=O.parents[2]
def sheet(path,times,out,columns=6):
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(R/path)]))
 pts=np.array([float(f['best_effort_timestamp_time']) for f in probe['frames']]);pts-=pts[0]
 ids={int(np.argmin(abs(pts-t))) for t in times};frames={};cap=cv2.VideoCapture(str(R/path));i=0
 while True:
  ok,a=cap.read()
  if not ok:break
  if i in ids:frames[i]=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB))
  i+=1
 cap.release();assert len(frames)==len(ids)
 w,h=300,250;canvas=Image.new('RGB',(w*columns,h*((len(ids)+columns-1)//columns)),(18,18,18));draw=ImageDraw.Draw(canvas)
 for j,(idx,im) in enumerate(sorted(frames.items())):
  im.thumbnail((w,h-24));x=(j%columns)*w+(w-im.width)//2;y=(j//columns)*h
  canvas.paste(im,(x,y+24));draw.text(((j%columns)*w+5,y+4),f'{pts[idx]:.4f}s frame {idx}',fill='white')
 out=O/out;canvas.save(out);out.with_suffix('.json').write_text(json.dumps({'video':path,'frames':[{'index':i,'pts':float(pts[i])} for i in sorted(ids)]},indent=2)+'\n')
if __name__=='__main__':
 sheet('data/local/phone_session_pull_v17_20261009b/live_reel_videos/20261009_135024_11.mp4',np.arange(0,33,.5),'have_overview.jpg')
 sheet('data/local/phone_session_pull_v17_20261011/20261011_102647_03.mp4',np.arange(2.3,3.61,.1),'my_first_detail.jpg')
 print('Phone review sheets prepared')
