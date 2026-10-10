from pathlib import Path
import json,numpy as np
O=Path(__file__).resolve().parent

def measure(rows):
 out=[];body=None;bs=-1e9
 for r in rows:
  if 's' not in r:continue
  t=r['s'];w=r['w'];h=r['h'];b=np.array(r['b'])
  if min(b[:2,2])>=.3:body=b[:2,:2]*[w,h];bs=t
  for slot in ['l','r']:
   if slot not in r or body is None or t-bs>1.5:continue
   a=np.array(r[slot]);needed=[0,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
   if min(a[[0,4,9,12],2])<.3:continue
   flat_valid=min(a[needed,2])>=.3
   xy=a[:,:2]*[w,h];size=np.linalg.norm(xy[9]-xy[0]);sw=np.linalg.norm(body[0]-body[1]);mid=body.mean(0)
   if size<5 or sw<10:continue
   palm=xy[[0,5,9,13,17]].mean(0);straight=[]
   for m,tip in [(5,8),(9,12),(13,16),(17,20)]:
    chain=sum(np.linalg.norm(xy[j+1]-xy[j]) for j in range(m,tip));straight.append(np.linalg.norm(xy[tip]-xy[m])/chain if chain>0 else 0.)
   out.append(dict(t=t,slot=slot,pinch=float(np.linalg.norm(xy[4]-xy[12])/size),straight=float(min(straight)) if flat_valid else 0.,flat_valid=bool(flat_valid),size=float(size),px=float((palm[0]-mid[0])/sw),py=float((palm[1]-mid[1])/sw),x=float(palm[0]/sw),y=float(palm[1]/sw)))
 return out
if __name__=='__main__':
 p=O.parent/'my_like_regression_v17_20261011/first_miss_trace.json';rows=measure(json.loads(p.read_text())['frames']);(O/'first_miss_motion.json').write_text(json.dumps(rows,indent=2)+'\n')
 for r in rows:
  if 63.7<=r['t']<=64.85:print({k:round(v,3) if isinstance(v,float) else v for k,v in r.items()})
