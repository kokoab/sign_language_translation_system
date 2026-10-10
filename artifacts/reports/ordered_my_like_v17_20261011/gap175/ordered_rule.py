"""Ordered, confidence-aware motion confirmation; missing evidence never means MY."""
import math
DEFAULT=dict(flat=.97,open=.8,closed=.55,closure=.35,outward=.15,lookahead=.5,gap=.175)

def assess(features,start,end,available,policy=DEFAULT,next_start=None):
 # Features are original source-time samples, not the recognizer's resampled32frames.
 limit=min(end+policy['lookahead'],next_start-1e-5) if next_start is not None else end+policy['lookahead']
 if limit<end+.08 or available < limit-1e-6:return dict(decision='unknown',reason='incomplete_future')
 hands=[]
 for slot in ['l','r']:
  rr=[r for r in features if r['slot']==slot and start-.15-1e-6<=r['t']<=limit+1e-6]
  near=lambda r:abs(r['px'])<=.6 and -.15<=r['py']<=1.0
  contacts=[r for r in rr if r['t']<=end and near(r)]
  if not contacts:continue
  anchors=[r for r in contacts if r['flat_valid'] and r['straight']>=policy['flat'] and r['pinch']>=policy['open']]
  # LIKE's open-to-closed trajectory does not require an initially flat palm.
  like=False;closing=False
  for a in contacts:
   tail=[r for r in rr if r['t']>=a['t']]
   for i,b in enumerate(tail):
    if b['t']-a['t']<.05 or b['t']-a['t']>.6:continue
    if b['pinch']<=policy['closed'] and a['pinch']-b['pinch']>=policy['closure']:
     closing=True
     # 2D chest-relative departure or hand scale growth toward the camera.
     moved=math.hypot(b['px']-a['px'],b['py']-a['py'])>=policy['outward'] or b['size']/a['size']>=1.15
     contiguous=all(y['t']-x['t']<=policy['gap']+1e-6 for x,y in zip(tail[:i],tail[1:i+1]))
     if moved and contiguous:like=True
  if like:return dict(decision='LIKE',reason='ordered_closure_and_departure',slot=slot)
  if len(anchors)<2 or anchors[-1]['t']-anchors[0]['t']<.05:
   hands.append('unknown');continue
  a=anchors[0];tail=[r for r in rr if r['t']>=a['t']]
  complete=(len(tail)>=4 and tail[-1]['t']>=limit-.10 and all(y['t']-x['t']<=policy['gap']+1e-6 for x,y in zip(tail,tail[1:])))
  # Strong MY posture plus observed open fingers through the departure window.
  no_pinch=all(r['pinch']>=policy['closed']+.15 for r in tail)
  hands.append('MY' if complete and no_pinch and not closing else 'unknown')
 if hands and all(h=='MY' for h in hands):return dict(decision='MY',reason='observed_flat_contact_without_pinch')
 return dict(decision='unknown',reason='insufficient_or_ambiguous_motion')

def selftest():
 def frames(pinch,flat=True):return [dict(t=i*.05,slot='r',px=0. if i<3 else (i-2)*.08,py=.3,size=100.,straight=.99 if flat else .8,flat_valid=True,pinch=p) for i,p in enumerate(pinch)]
 a=frames([1.2]*16);assert assess(a,0.,.2,.7)['decision']=='MY'
 b=frames([1.2,1.2,1.1,.9,.7,.5,.4,.3,.3,.3,.3,.3,.3,.3,.3,.3]);assert assess(b,0.,.2,.7)['decision']=='LIKE'
 assert assess(list(reversed([dict(r,t=.75-r['t']) for r in b])),0.,.2,.7)['decision']!='LIKE'
 assert assess([r for i,r in enumerate(a) if i<3 or i>7],0.,.2,.7)['decision']=='unknown'
 assert assess(a,0.,.2,.3)['decision']=='unknown'
 # A pinch that happened before the chest contact must not confirm LIKE.
 assert assess(frames([.2,.2,.3,.6,.9,1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2]),0.,.2,.7)['decision']!='LIKE'
 print('Ordered cue tests passed: MY, LIKE, reversed motion, occlusion, incomplete future, prior pinch.')
if __name__=='__main__':selftest()
