from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse, Polygon, FancyArrowPatch
P=Path(__file__).parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
BLUE='#245b85'; TEAL='#14776f'
def canvas(w,h):
 f,a=plt.subplots(figsize=(w,h));a.set(xlim=(0,w),ylim=(0,h));a.axis('off');return f,a
def node(a,x,y,text,w=2.5,h=.7,kind='box'):
 color=TEAL if kind=='process' else BLUE
 if kind=='decision':p=Polygon([(x-w/2,y),(x,y+h/2),(x+w/2,y),(x,y-h/2)],fc='#fff3df',ec='#aa6e26')
 elif kind=='entity':p=Rectangle((x-w/2,y-h/2),w,h,fc='white',ec=color)
 elif kind=='store':
  a.plot([x-w/2,x+w/2],[y-h/2,y-h/2],color=color);a.plot([x-w/2,x+w/2],[y+h/2,y+h/2],color=color);p=None
 elif kind=='process':p=Ellipse((x,y),w,h,fc='#eaf4f1',ec=color)
 else:p=FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=.04,rounding_size=.08',fc='#edf3f7',ec=color)
 if p:a.add_patch(p)
 a.text(x,y,text,ha='center',va='center',fontsize=10)
def arrow(a,pts,label='',lp=None):
 for p,q in zip(pts[:-2],pts[1:-1]):a.plot([p[0],q[0]],[p[1],q[1]],color='#52616d',lw=1.2)
 a.add_patch(FancyArrowPatch(pts[-2],pts[-1],arrowstyle='-|>',mutation_scale=12,color='#52616d',lw=1.2))
 if label:
  x,y=lp or ((pts[0][0]+pts[-1][0])/2,(pts[0][1]+pts[-1][1])/2)
  a.text(x,y,label,ha='center',va='center',fontsize=8.5,bbox=dict(fc='white',ec='none',pad=1))
def save(f,n):
 f.tight_layout(pad=.3)
 for ext in ['png','svg']:f.savefig(P/(n+'.'+ext),dpi=200,bbox_inches='tight')
 plt.close(f)
# Live processing: separate continuous processing, gesture completion and button stop.
f,a=canvas(9,8.5)
for y,tx,k in [(8,'Start Live recognition','box'),(6.8,'Camera → landmarks + hand images','box'),(5.6,'Boundary estimates + recognizer\n→ decoder → accepted glosses','box'),(4.4,'Update English incrementally','box'),(3.2,'Complete message?','decision'),(1.9,'Finalize English and speak','box'),(.7,'Stop session?','decision')]:node(a,4,y,tx,w=4.4,h=.65,kind=k)
for y,z in [(8,6.8),(6.8,5.6),(5.6,4.4),(4.4,3.2)]:arrow(a,[(4,y-.35),(4,z+.35)])
arrow(a,[(4,2.87),(4,2.25)],'Finish button / gesture',(5.25,2.55));arrow(a,[(4,1.55),(4,1.05)])
arrow(a,[(1.8,3.2),(.7,3.2),(.7,6.8),(1.8,6.8)],'No',(.7,4.6))
arrow(a,[(1.8,.7),(.3,.7),(.3,6.8),(1.8,6.8)],'Gesture: lower hands\nand continue',(.85,1.9))
node(a,7.7,.7,'Save session;\nreturn to Ready',w=2,h=.7)
arrow(a,[(6.2,.7),(6.7,.7)],'Button',(6.45,1.1));save(f,'live_flow')
# Context diagram: one system process, external roles.
f,a=canvas(10,4)
node(a,1.6,2,'Signer',w=2.4,kind='entity');node(a,5,2,'0. ATLAS\nmobile application',w=3,h=1.3,kind='process');node(a,8.5,2,'Application user /\nmessage recipient',w=2.7,h=1,kind='entity')
arrow(a,[(2.8,2.3),(3.5,2.3)],'Signing video',(3.15,2.8));arrow(a,[(6.5,2.3),(7.15,2.3)],'Glosses, English, speech',(6.9,2.95))
arrow(a,[(8.5,1.5),(8.5,.65),(5,.65),(5,1.35)],'Selections and completion controls',(6.6,.65));save(f,'context')
# DFD with processes, external entity, data store, data-labelled edges.
f,a=canvas(10,8)
node(a,5,7.5,'User',kind='entity',w=2)
for x,y,tx in [(2.5,5.9,'1. Extract visual\nfeatures'),(7.5,5.9,'2. Recognize and\ndecode signs'),(7.5,3.6,'3. Maintain glosses\nand generate English'),(2.5,3.6,'4. Present output\nand manage sessions')]:node(a,x,y,tx,w=3.5,h=1,kind='process')
node(a,5,1.1,'D1  Local session history',w=4,h=.75,kind='store')
arrow(a,[(5,7.15),(5,6.85),(2.5,6.85),(2.5,6.4)],'Signing frames',(3.7,6.85))
arrow(a,[(4.25,5.9),(5.75,5.9)],'Visual features',(5,6.2))
arrow(a,[(7.5,5.4),(7.5,4.1)],'Accepted glosses',(7.5,4.75))
arrow(a,[(5.75,3.6),(4.25,3.6)],'Glosses and English',(5,3.9))
arrow(a,[(.75,3.6),(.2,3.6),(.2,7.5),(4,7.5)],'Visible / spoken output',(.7,4.7))
arrow(a,[(6,7.5),(9.7,7.5),(9.7,3.6),(9.25,3.6)],'Completion request',(9.25,4.65))
arrow(a,[(2.5,3.1),(2.5,1.1),(3,1.1)],'Session record',(2.5,2))
arrow(a,[(5,1.475),(5,2.4),(2.9,2.4),(2.9,3.11)],'Saved session',(4.3,2.4))
save(f,'data_flow')
# Use cases.
f,a=canvas(9,5.5);a.add_patch(Rectangle((3,.3),5.7,5,fc='none',ec=BLUE));a.text(5.85,5,'ATLAS',ha='center',weight='bold');node(a,1.25,2.7,'Application\nuser',w=2,kind='entity')
for y,tx in [(4.3,'Use Live recognition'),(3.45,'Complete a message'),(2.6,'Browse sign demonstrations'),(1.75,'Practice supported signs'),(.9,'Review saved sessions')]:node(a,5.9,y,tx,w=4.7,h=.6,kind='process');a.plot([2.25,3.55],[2.7,y],color='#52616d',lw=.9)
save(f,'use_cases')
# Compact navigation; functional loops are explicit.
f,a=canvas(15,9)
node(a,7.5,8.5,'Open ATLAS → Home',w=4)
for x,tx in [(1.6,'Live'),(4.5,'Glosses'),(7.5,'Practice'),(10.4,'History'),(13.4,'Manuals')]:node(a,x,7.2,tx,w=2.3);arrow(a,[(7.5,8.15),(7.5,7.9),(x,7.9),(x,7.55)])
node(a,1.6,5.8,'Start; sign and view\nrecognition / English',w=2.5,h=.9);node(a,1.6,4.1,'Complete message\n(button or gesture)',w=2.5,h=.9);node(a,1.6,2.4,'Button: save and stop\nGesture: resume signing',w=2.5,h=.9)
for y,z in [(7.2,5.8),(5.8,4.1),(4.1,2.4)]:arrow(a,[(1.6,y-.45),(1.6,z+.45)])
arrow(a,[(.35,2.4),(.12,2.4),(.12,5.8),(.35,5.8)],'Gesture',(.4,3.3))
node(a,4.5,5.8,'Search / select sign;\nwatch demonstration',w=2.5,h=.9);arrow(a,[(4.5,6.85),(4.5,6.25)])
node(a,7.5,5.8,'Select set; view target\nand attempt sign',w=2.5,h=.9);arrow(a,[(7.5,6.85),(7.5,6.25)])
node(a,7.5,4.1,'Correct or Skip?',w=2.6,h=.85,kind='decision');arrow(a,[(7.5,5.35),(7.5,4.53)])
arrow(a,[(6.2,4.1),(5.95,4.1),(5.95,5.8),(6.25,5.8)],'Retry',(6,4.8))
node(a,7.5,2.4,'More targets?',w=2.6,h=.85,kind='decision');arrow(a,[(7.5,3.68),(7.5,2.83)],'Yes / Skip',(8.25,3.25))
arrow(a,[(8.8,2.4),(9.05,2.4),(9.05,5.8),(8.75,5.8)],'Yes',(9.05,4.8))
node(a,7.5,.9,'Show practice summary',w=2.6);arrow(a,[(7.5,1.98),(7.5,1.25)],'No',(7.9,1.6))
node(a,10.4,5.8,'Select saved session;\nreview signs and English',w=2.5,h=.9);arrow(a,[(10.4,6.85),(10.4,6.25)])
node(a,13.4,5.8,'Choose topic / read guide',w=2.6,h=.9)
arrow(a,[(13.4,6.85),(13.4,6.25)])
node(a,13.4,4.1,'Watch referenced sign\nin Glosses (optional)',w=2.6,h=.9)
arrow(a,[(13.4,5.35),(13.4,4.55)])
node(a,13.4,2.4,'Back to guide / topics;\nreturn Home',w=2.6,h=.9)
arrow(a,[(13.4,3.65),(13.4,2.85)])
a.text(7.5,.1,'Back returns to Home. Live / Practice ? opens the relevant manual section.',ha='center',fontsize=10)
save(f,'app_flow')
# Streaming decisions.
f,a=canvas(9,6.5)
for y,tx,k in [(6,'Observe landmarks and hand images','box'),(4.8,'Boundary estimates → candidate intervals','box'),(3.6,'Recognizer scores → segmental decoding','box'),(2.4,'Acceptance conditions met?','decision'),(1.2,'Append gloss; update English','box')]:node(a,4.5,y,tx,w=5,h=.65,kind=k)
for y,z in [(6,4.8),(4.8,3.6),(3.6,2.4)]:arrow(a,[(4.5,y-.35),(4.5,z+.35)])
arrow(a,[(4.5,2.07),(4.5,1.55)],'Yes',(5,1.8));arrow(a,[(7,2.4),(8,2.4),(8,6),(7,6)],'No: observe next frames',(8,4.8));arrow(a,[(2,1.2),(1,1.2),(1,6),(2,6)],'Continue',(.8,3.6));save(f,'streaming')
# Deployment only.
f,a=canvas(10,3.8)
xs=[1.2,3.7,6.2,8.7]
for x,tx in zip(xs,['Export selected\nmodels','Check conversion\non matched inputs','Integrate and\ninstall application','Evaluate on\nthe device']):node(a,x,2,tx,w=2.1,h=1)
for x,y in zip(xs,xs[1:]):arrow(a,[(x+1.05,2),(y-1.05,2)])
a.text(5,.75,'FP16: visual / recognition components     FP32: English generation',ha='center',fontsize=10);save(f,'deployment')
