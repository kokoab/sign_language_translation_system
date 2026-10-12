from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[4];O=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
def save(fig,name):
 fig.savefig(O/(name+'.png'),bbox_inches='tight');fig.savefig(O/(name+'.svg'),bbox_inches='tight');plt.close(fig)
r=json.loads((R/'artifacts/reports/current_lineage_components_v17_20261013/results.json').read_text())
keys=['landmark_only','hand_only','fusion','phrase','interval','current_primary'];names=['Landmarks only','Hand images only','Combined inputs','Phrase-adapted','Interval-adapted','Current primary recognizer'];v=[r['summary'][k]['accuracy_percent'] for k in keys]
fig,ax=plt.subplots(figsize=(10,4.8));bars=ax.barh(names[::-1],v[::-1],color=['#166b68']+['#7396ad']*5);ax.set_xlim(0,105);ax.set_xlabel('Validation accuracy (%)');ax.set_title('Recognition components on the same validation inputs',loc='left',pad=16)
for bar,x in zip(bars,v[::-1]):ax.text(x+1,bar.get_y()+bar.get_height()/2,f'{x:.2f}%',va='center')
ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);fig.tight_layout();save(fig,'component_accuracy')
h=json.loads((R/'artifacts/generated/kaggle_stage1_partwise_kokoab_pull_v1/stage1_v17_partwise_v2/history.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,4))
e=[x['epoch'] for x in h];axes[0].plot(e,[x['train_loss'] for x in h],label='Training loss',color='#166b68');axes[0].plot(e,[x['loss'] for x in h],label='Validation loss',color='#b56a39');axes[0].set_ylabel('Loss');axes[0].legend(frameon=False)
axes[1].plot(e,[x['top1'] for x in h],color='#166b68');axes[1].scatter([100],[96.8253968254],color='#b56a39',zorder=4);axes[1].annotate('Selected epoch 100: 96.83%',(100,96.8254),xytext=(35,65),arrowprops={'arrowstyle':'->','color':'#555'});axes[1].set_ylabel('Base validation accuracy (%)');axes[1].set_ylim(0,103)
for ax in axes:ax.set_xlabel('Epoch');ax.axvline(100,color='#777',linestyle=':',alpha=.6);ax.grid(alpha=.15)
fig.suptitle('Base landmark classifier — recorded training history',x=.08,ha='left');fig.tight_layout();save(fig,'base_training')
h=json.loads((R/'artifacts/reports/go_split_root_cause_v17_20261009/run_frag25_reject/span_recognizer/history.json').read_text())['history'];h=[x for x in h if x['epoch']>0]
fig,ax=plt.subplots(figsize=(8.5,4));ax.plot([x['epoch'] for x in h],[x['loss'] for x in h],marker='o',color='#166b68');ax.axvline(4,color='#b56a39',ls='--',label='Selected checkpoint: epoch 4');ax.set(xlabel='Epoch',ylabel='Training loss',title='Current primary recognizer — interval / partial-sign training');ax.set_xticks(range(1,9));ax.grid(alpha=.15);ax.legend(frameon=False);fig.tight_layout();save(fig,'current_training')
fig,axes=plt.subplots(1,2,figsize=(11,4.5),gridspec_kw={'width_ratios':[2,3]})
for ax,names,values,title in [(axes[0],['Before interval\nadaptation','After interval\nadaptation'],[35.84,29.65],'Adaptation experiment'),(axes[1],['Runtime\nbaseline','Partial-sign\nconfiguration','Transition/runtime\nrefinement'],[17.70,15.49,12.83],'Runtime development')]:
 bars=ax.bar(names,values,color='#7396ad');ax.set_ylim(0,45);ax.set_title(title);ax.set_ylabel('WER (%)');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 for i,(bar,x) in enumerate(zip(bars,values)):ax.text(bar.get_x()+bar.get_width()/2,x+1,f'{x:.2f}'+('\nretained' if ax==axes[1] and i==2 else ''),ha='center',va='bottom')
fig.suptitle('Recognition performance across development stages',x=.08,ha='left');fig.tight_layout();save(fig,'wer_stages')
(O/'sources.json').write_text(json.dumps({'accuracy':'artifacts/reports/current_lineage_components_v17_20261013/results.json','base_history':'artifacts/generated/kaggle_stage1_partwise_kokoab_pull_v1/stage1_v17_partwise_v2/history.json','current_history':'artifacts/reports/go_split_root_cause_v17_20261009/run_frag25_reject/span_recognizer/history.json','wer_evidence':'artifacts/reports/paper_metric_lineage_v17_20261013/verification.json','wer_training_history':'artifacts/reports/citizen_variant_local_filter_v17_20261008/run_v2/span_recognizer/history.json'},indent=2)+'\n')
