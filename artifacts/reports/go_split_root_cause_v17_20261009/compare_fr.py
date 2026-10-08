"""Installed vs fragment/reject recognizers in the installed decoder (dip bridging + HAVE/ANGRY check)."""
import json,collections,glob
def load(f,k):
    try:return json.load(open(f)).get(k)
    except FileNotFoundError:return None
V={'installed':{'citizen_val':('results_both.json','citizen_val:frag_fix_both'),'tune':('results_both.json','tune:frag_fix_both'),
                'pair_val':('results_pv_p1.json','pair_val:frag_fix_both'),'takes3':('results_t3_p0.json','takes3:inst'),
                'takes':('results_t3_p0.json','takes:inst'),'takes2':('results_t3_p0.json','takes2:inst')}}
for i,n in enumerate(['frag25','frag25_reject','frag25_reject35']):
    V[n]={s:(f'results_fr_p{i}.json',f'{s}:{n}_inst') for s in ('citizen_val','tune','pair_val','takes3','takes','takes2')}
for i,n in enumerate(['reject_own_th40','reject_own_th50']):
    V[n]={s:(f'results_own_p{i}.json',f'{s}:{n}') for s in ('citizen_val','tune','pair_val','takes3','takes','takes2')}
for i,n in enumerate(['frag25_reject_th40','frag25_reject_th35','installed_th40']):
    V[n]={s:(f'results_th_p{i}.json',f'{s}:{n}') for s in ('citizen_val','tune','pair_val','takes3','takes','takes2')}
watch=[('YEAR','MAKE'),('ANSWER','GO'),('DIFFERENT','GO'),('PLEASE','MY'),('WE','I'),('SORRY','I'),('DRINK','MOTHER'),('LEARN','SICK'),('MAN','FATHER'),('FAMILY','WAIT')]
for name,sets in V.items():
    out=[]
    r=load(*sets['tune']);out.append('tune WER %.3f (S%d D%d I%d)'%(r['summary']['wer'],r['summary']['S'],r['summary']['D'],r['summary']['I']) if r else 'tune -')
    r=load(*sets['citizen_val'])
    if r:
        rows=r['rows'];multi=sum(len(x['words'])>1 for x in rows);s=r['summary']
        ws=collections.Counter()
        for x in rows:
            h=[w['g'] for w in x['words']];t=x['target'][0]
            for a,b in watch:
                if t==a and b in h:ws[f'{a}+{b}']+=1
        out.append(f"citizen correct {s['correct']} exact {s['exact']} silent {s['silent']} extra {s['extra']} multi-word {multi} watch {dict(ws)}")
    r=load(*sets['pair_val'])
    if r:
        c=collections.Counter()
        for x in r['rows']:
            t=x['target'][0];h=[w['g'] for w in x['words']];c[t+'_n']+=1;c[t]+=t in h
        out.append('pair_val ANGRY %d/%d HAVE %d/%d TIRED %d/%d HAPPY %d/%d'%(c['ANGRY'],c['ANGRY_n'],c['HAVE'],c['HAVE_n'],c['TIRED'],c['TIRED_n'],c['HAPPY'],c['HAPPY_n']))
    t3=load(*sets['takes3'])
    if t3:v=t3['summary']['01'];out.append(f"20:18 session HAVE {v['HAVE']} ANGRY {v['ANGRY']} words {v['n']}")
    t=load(*sets['takes'])
    if t:out.append('takes01-03 GO %d words %d'%(sum(t['summary'][k]['GO'] for k in ('01','02','03')),sum(t['summary'][k]['n'] for k in ('01','02','03'))))
    t2=load(*sets['takes2'])
    if t2:v=t2['summary']['11'];out.append(f"take11 HAVE {v['HAVE']} ANGRY {v['ANGRY']}")
    print(f'== {name}\n   '+'\n   '.join(out))
