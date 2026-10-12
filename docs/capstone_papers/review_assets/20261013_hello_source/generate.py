"""Plot saved model intervals and unmodified midpoint frames from the verified source."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
O=Path(__file__).resolve().parent
meta=json.loads((O/'frames.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12})
fig=plt.figure(figsize=(8.5,5.5),facecolor='white')
gs=fig.add_gridspec(2,3,height_ratios=[1.25,1],left=.035,right=.985,top=.96,bottom=.08,hspace=.46,wspace=.09)
ax=fig.add_subplot(gs[0,:]);ax.set_xlim(-1.48,4.7);ax.set_ylim(-.55,1.55)
colors=['#176c68','#35739a','#80627f']
for r,c in zip(meta['frames'],colors):
 s,e=r['interval'];mid=(s+e)/2
 ax.barh(1,e-s,left=s,height=.46,color='#e5ebef',edgecolor='#627681',linewidth=1)
 ax.text(mid,1,f'{s:.2f}–{e:.2f} s',ha='center',va='center',fontsize=11)
 ax.barh(0,e-s,left=s,height=.46,color=c)
 ax.text(mid,0,r['gloss'],ha='center',va='center',color='white',weight='bold',fontsize=12)
ax.text(-.12,1,'Selected intervals\nBoundary + decoder',ha='right',va='center',fontsize=10)
ax.text(-.12,0,'Recognized glosses\nRecognizer + decoder',ha='right',va='center',fontsize=10)
ax.set_yticks([]);ax.set_xticks([0,1,2,3,4]);ax.set_xlabel('Position within the recording (seconds)',fontsize=11)
for side in ['top','right','left']:ax.spines[side].set_visible(False)
ax.spines['bottom'].set_bounds(0,4.6);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
for i,r in enumerate(meta['frames']):
 a=fig.add_subplot(gs[1,i]);a.imshow(Image.open(O/(r['gloss'].lower()+'.png')));a.axis('off');a.set_title(f"{r['gloss']} · {r['seconds']:.2f} s",fontsize=12,pad=7)
fig.savefig(O/'hello_timeline_with_frames.png',dpi=220,bbox_inches='tight',facecolor='white')
fig.savefig(O/'hello_timeline_with_frames.svg',bbox_inches='tight',facecolor='white')
plt.close(fig)
